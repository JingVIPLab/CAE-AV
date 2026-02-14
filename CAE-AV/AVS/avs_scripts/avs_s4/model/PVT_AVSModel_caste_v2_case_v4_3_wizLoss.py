import torch
import torch.nn as nn
import torchvision.models as models
from model.pvt import pvt_v2_b5
from model.TPAVI import TPAVIModule
from ipdb import set_trace
import timm
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable
import numpy as np
import copy
import math
from ipdb import set_trace
import os

from torch import Tensor
from typing import Optional, Any
from einops import rearrange, repeat
from .models import EncoderLayer, Encoder, DecoderLayer, Decoder
from timm.models.vision_transformer import Attention
import timm
import loralib as lora
from transformers.activations import get_activation
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint

### VGGSound
from .htsat import HTSAT_Swin_Transformer
import model.esc_config as esc_config
from .utils import do_mixup, get_mix_lambda, do_mixup_label

from torch.nn import init


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
                                    bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                    dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output

    
class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output
    
    
class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature

class ExpertAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True, num_tk=87, is_multimodal=True):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = is_multimodal
        self.opt = opt
        self.num_tk = num_tk
        # self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)
        # self.fc = nn.Linear(linear_in, linear_out)
        # self.conv_dim_out = conv_dim_out

        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        if adapter_kind == "bottleneck" and self.is_multimodal:
            self.down_sample_size = input_dim // reduction_factor
            self.my_tokens = nn.Parameter(torch.rand((num_tk, input_dim)))

            self.gate_av = nn.Parameter(torch.zeros(1))
            self.activation = nn.ReLU(inplace=True)
            # self.down_sampler = nn.Linear(input_dim, self.down_sample_size, bias=False)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)
            # self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            ### -------> yb: add
            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "bottleneck":
            self.down_sample_size = input_dim // reduction_factor
            self.activation = nn.ReLU(inplace=True)
            self.num_head = 4
            self.head_dropout = 0.2
            if self.opt.is_self_attention:
                if self.opt.self_attention_version == "v1":
                    self.self_attention = MultiheadAttention(input_dim, num_heads=self.num_head, dropout=self.head_dropout)
                if self.opt.self_attention_version == "v2":
                    self.my_tokens = nn.Parameter(torch.rand((num_tk, input_dim)))
                    self.gate_self = nn.Parameter(torch.zeros(1))

            # self.down_sampler = nn.Linear(input_dim, self.down_sample_size, bias=False)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)
            # nn.init.zeros_(self.down_sampler)  # yb:for lora

            # self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            ### -------> yb: add
            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "basic":
            self.activation = nn.ReLU(inplace=True)
            # self.conv = nn.Conv2d(input_dim, output_dim, 1, bias=False)
            self.conv = nn.Linear(input_dim, output_dim, bias=False)

            if use_bn:
                self.bn = nn.BatchNorm1d(output_dim)

        else:
            raise NotImplementedError

    def forward(self, x, vis_token=None):
        if self.adapter_kind == "bottleneck" and self.is_multimodal:
            ### -------> high dim att
            rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
            att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))
            att_v2tk = F.softmax(att_v2tk, dim=-1)
            rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0, 2, 1))
            rep_token = rep_token + rep_token_res

            att_tk2x = torch.bmm(x.squeeze(-1).permute(0, 2, 1), rep_token.permute(0, 2, 1))

            att_tk2x = F.softmax(att_tk2x, dim=-1)
            x_res = torch.bmm(att_tk2x, rep_token).permute(0, 2, 1).unsqueeze(-1)

            x = x + self.gate_av * x_res.contiguous()
            ### <----------
            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "bottleneck":
            # x shape1:  torch.Size([20, 96, 4096, 1])

            if self.opt.is_self_attention:
                if self.opt.self_attention_version == "v1":
                    x = x.squeeze(-1).permute(0,2,1)
                    x, x_weights = self.self_attention(x, x, x)
                    x = x.permute(0, 2, 1).unsqueeze(-1)
                if self.opt.self_attention_version == "v2":
                    rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
                    att_v2tk = torch.bmm(rep_token, x.squeeze(-1))

                    att_v2tk = F.softmax(att_v2tk, dim=-1)
                    rep_token_res = torch.bmm(att_v2tk, x.squeeze(-1).permute(0, 2, 1))
                    rep_token = rep_token + rep_token_res

                    att_tk2x = torch.bmm(x.squeeze(-1).permute(0, 2, 1), rep_token.permute(0, 2, 1))
                    att_tk2x = F.softmax(att_tk2x, dim=-1)
                    x_res = torch.bmm(att_tk2x, rep_token).permute(0, 2, 1).unsqueeze(-1)

                    x = x + self.gate_self * x_res.contiguous()
            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "basic":
            output = self.conv(x)
            if self.use_bn:
                output = self.bn(rearrange(output, 'N C L -> N L C'))
                output = rearrange(output, 'N L C -> N C L')

        if self.opt.is_post_layernorm:
            output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

        if self.gate is not None:
            output = self.gate * output
        return output


class MoEAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, adapter_kind, dim_list, layer_idx, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True, num_tk=87, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
        super().__init__()
        self.opt = opt
        self.use_bn = use_bn
        self.num_tk = num_tk
        self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)
        self.fc = nn.Linear(linear_in, linear_out)
        self.num_multimodal_experts = self.opt.num_multimodal_experts
        self.num_singlemodal_experts = self.opt.num_singlemodal_experts
        self.multimodal_experts = nn.ModuleList([
            ExpertAdapter(input_dim, output_dim, adapter_kind, reduction_factor, opt, use_bn,
                          use_gate, num_tk, is_multimodal=True)
            for _ in range(self.num_multimodal_experts)
        ])
        self.singlemodal_experts = nn.ModuleList([
            ExpertAdapter(input_dim, output_dim, adapter_kind, reduction_factor, opt, use_bn,
                          use_gate, num_tk, is_multimodal=False)
            for _ in range(self.num_singlemodal_experts)
        ])

        self.router = nn.Sequential(
            nn.Linear(input_dim + linear_out, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_multimodal_experts + self.num_singlemodal_experts)
        )

    def forward(self, x, vis_token=None, is_training=True):
        vis_token = self.conv_adapter(vis_token.transpose(2, 1))
        vis_token_fc = self.fc(vis_token.squeeze(-1))
        vis_token = vis_token_fc.permute(0, 2, 1).unsqueeze(-1)
        modal_1 = x.squeeze(-1).permute(0, 2, 1)
        modal_2 = vis_token_fc
        modal_1 = modal_1.mean(dim=1, keepdim=True)
        modal_2 = modal_2.mean(dim=1, keepdim=True)

        multimodal_input = torch.cat((modal_1, modal_2), dim=-1)
        gating_logits = self.router(multimodal_input)
        if is_training:
            noise = torch.randn_like(gating_logits) * 0.01
            gating_logits = gating_logits + noise
            
        gating_probs = F.softmax(gating_logits, dim=-1)

        expert_indices = torch.argmax(gating_probs, dim=-1)

        expert_outputs = []
        for expert in self.multimodal_experts + self.singlemodal_experts:
            expert_output = expert(x, vis_token)
            expert_outputs.append(expert_output)
        expert_outputs_tensor = torch.concat(expert_outputs, dim=-1)
        final_expert_output = (expert_outputs_tensor * gating_probs.unsqueeze(-2)).sum(dim=-1, keepdim=True)
        if self.opt.use_load_balacing_loss==1:
            load_balancing_loss = self.compute_load_balancing_loss(gating_probs)
        else:
            load_balancing_loss = 0.
        return final_expert_output, expert_indices, gating_probs, load_balancing_loss
    
    def compute_load_balancing_loss(self, gating_probs):
        expert_probs_mean = torch.mean(gating_probs, dim=0)
        uniform_distribution = torch.full_like(expert_probs_mean, 1.0 / expert_probs_mean.size(0))
        load_balancing_loss = F.kl_div(expert_probs_mean.log(), uniform_distribution, reduction='batchmean')
        return load_balancing_loss
class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x

    
class TemporalAttention(nn.Module):
    def __init__(self):
        super(TemporalAttention, self).__init__()
        self.gamma = 0.05
        self.video_input_dim = 256
        self.audio_input_dim = 128

        self.video_fc_dim = 256
        self.audio_fc_dim = 128
        self.d_model = 256

        self.v_fc = nn.ModuleList([nn.Linear(self.video_input_dim, self.video_fc_dim) for i in range(4)])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.video_encoder = nn.ModuleList([InternalTemporalRelationModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
        self.video_decoder = nn.ModuleList([CrossModalRelationAttModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024) for i in range(4)]) 
        self.audio_encoder = nn.ModuleList([InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
        self.audio_decoder = nn.ModuleList([CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
        self.audio_visual_rnn_layer = nn.ModuleList([RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1) for i in range(4)])

        self.audio_gated = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    ) for i in range(4)])
        self.video_gated = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    ) for i in range(4)])
        
    def forward(self, visual_feature_list, audio_feature):
        # shape for pvt-v2-b5
        # BF x 256 x 56 x 56
        # BF x 256 x 28 x 28
        # BF x 256 x 14 x 14
        # BF x 256 x  7 x  7

        bs = audio_feature.size(0)
        x1, x2, x3, x4 = visual_feature_list
        x1_ = self.avgpool(x1)
        x1_ = x1_.squeeze()
        x2_ = self.avgpool(x2)
        x2_ = x2_.squeeze()
        x3_ = self.avgpool(x3)
        x3_ = x3_.squeeze()
        x4_ = self.avgpool(x4)
        x4_ = x4_.squeeze()
        
        x1_ = x1_.view(bs, 5, -1)
        x2_ = x2_.view(bs, 5, -1)
        x3_ = x3_.view(bs, 5, -1)
        x4_ = x4_.view(bs, 5, -1)

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.view(-1, audio_feature.size(-1))
        x1_, x2_, x3_, x4_ = [self.v_fc[i](x) for i, x in enumerate([x1_, x2_, x3_, x4_])]
        x1_, x2_, x3_, x4_ = [self.dropout(self.relu(x)) for x in [x1_, x2_, x3_, x4_]]
        
        visual_rnn_input = [x1_, x2_, x3_, x4_]

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer[0](audio_rnn_input, visual_rnn_input[0])
        audio_rnn_output2, visual_rnn_output2 = self.audio_visual_rnn_layer[1](audio_rnn_input, visual_rnn_input[1])
        audio_rnn_output3, visual_rnn_output3 = self.audio_visual_rnn_layer[2](audio_rnn_input, visual_rnn_input[2])
        audio_rnn_output4, visual_rnn_output4 = self.audio_visual_rnn_layer[3](audio_rnn_input, visual_rnn_input[3])
        
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [5, B, 256]
        audio_encoder_input2 = audio_rnn_output2.transpose(1, 0).contiguous()  # [5, B, 256]
        audio_encoder_input3 = audio_rnn_output3.transpose(1, 0).contiguous()  # [5, B, 256]
        audio_encoder_input4 = audio_rnn_output4.transpose(1, 0).contiguous()  # [5, B, 256]
        
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [5, B, 512]
        visual_encoder_input2 = visual_rnn_output2.transpose(1, 0).contiguous()  # [5, B, 512]
        visual_encoder_input3 = visual_rnn_output3.transpose(1, 0).contiguous()  # [5, B, 512]
        visual_encoder_input4 = visual_rnn_output4.transpose(1, 0).contiguous()  # [5, B, 512]

        # audio query
        video_key_value_feature1 = self.video_encoder[0](visual_encoder_input1)
        video_key_value_feature2 = self.video_encoder[1](visual_encoder_input2)
        video_key_value_feature3 = self.video_encoder[2](visual_encoder_input3)
        video_key_value_feature4 = self.video_encoder[3](visual_encoder_input4)
        
        audio_query_output1 = self.audio_decoder[0](audio_encoder_input1, video_key_value_feature1)
        audio_query_output2 = self.audio_decoder[1](audio_encoder_input2, video_key_value_feature2)
        audio_query_output3 = self.audio_decoder[2](audio_encoder_input3, video_key_value_feature3)
        audio_query_output4 = self.audio_decoder[3](audio_encoder_input4, video_key_value_feature4)
        
        # video query
        audio_key_value_feature1 = self.audio_encoder[0](audio_encoder_input1)
        audio_key_value_feature2 = self.audio_encoder[1](audio_encoder_input2)
        audio_key_value_feature3 = self.audio_encoder[2](audio_encoder_input3)
        audio_key_value_feature4 = self.audio_encoder[3](audio_encoder_input4)
        
        video_query_output1 = self.video_decoder[0](visual_encoder_input1, audio_key_value_feature1)
        video_query_output2 = self.video_decoder[1](visual_encoder_input2, audio_key_value_feature2)
        video_query_output3 = self.video_decoder[2](visual_encoder_input3, audio_key_value_feature3)
        video_query_output4 = self.video_decoder[3](visual_encoder_input4, audio_key_value_feature4)

        audio_gate1 = self.audio_gated[0](audio_key_value_feature1) # [5, B, 1]
        audio_gate2 = self.audio_gated[1](audio_key_value_feature2)
        audio_gate3 = self.audio_gated[2](audio_key_value_feature3)
        audio_gate4 = self.audio_gated[3](audio_key_value_feature4)
        
        video_gate1 = self.video_gated[0](video_key_value_feature1) # [5, B, 1]
        video_gate2 = self.video_gated[1](video_key_value_feature2)
        video_gate3 = self.video_gated[2](video_key_value_feature3)
        video_gate4 = self.video_gated[3](video_key_value_feature4)

        audio_gate1 = audio_gate1.transpose(1, 0)
        audio_gate1 = audio_gate1.reshape(bs*5, 1, 1, 1)
        audio_gate2 = audio_gate2.transpose(1, 0)
        audio_gate2 = audio_gate2.reshape(bs*5, 1, 1, 1)
        audio_gate3 = audio_gate3.transpose(1, 0)
        audio_gate3 = audio_gate3.reshape(bs*5, 1, 1, 1)
        audio_gate4 = audio_gate4.transpose(1, 0)
        audio_gate4 = audio_gate4.reshape(bs*5, 1, 1, 1)

        video_gate1 = video_gate1.transpose(1, 0)
        video_gate1 = video_gate1.reshape(bs*5, 1)
        video_gate2 = video_gate2.transpose(1, 0)
        video_gate2 = video_gate2.reshape(bs*5, 1)
        video_gate3 = video_gate3.transpose(1, 0)
        video_gate3 = video_gate3.reshape(bs*5, 1)
        video_gate4 = video_gate4.transpose(1, 0)
        video_gate4 = video_gate4.reshape(bs*5, 1)
        
        x1 = x1 + audio_gate1 * x1 * self.gamma
        x2 = x2 + audio_gate2 * x2 * self.gamma
        x3 = x3 + audio_gate3 * x3 * self.gamma
        x4 = x4 + audio_gate4 * x4 * self.gamma
        
        video_gate = (video_gate1 + video_gate2 + video_gate3 + video_gate4) / 4
        audio_feature = audio_feature + video_gate * audio_feature * self.gamma
        
        return [x1, x2, x3, x4], audio_feature
    
# ################## 映射器 ##################
class FeatureMapper(nn.Module):
    def __init__(self, clip_dim=768, hidden_dim=768, output_dim=768, dropout_r=0.2, head=8):
        super().__init__()
        self.mhatt = MHAtt(hidden_dim, dropout_r, head)
        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.norm1(x + self.dropout1(self.mhatt(x, x, x)))
        # NEW: 把 mapper 的输出以小比例注入
        return y


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, outdim=640):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_r)
        self.dense2 = nn.Linear(hidden_dim * 2, outdim)

    def forward(self, X):
        return self.dense2(self.dropout(self.relu(self.dense1(X))))



# ################## Adapter ##################
class Adapter(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=768//16, output_dim=None, activation="gelu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.bottleneck_dim = bottleneck_dim

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        # 下投影保持正常初始化
        nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.down_proj.bias)
        # NEW: 上投影零初始化（经典 Adapter 稳定化做法）
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        out = self.up_proj(self.activation(self.down_proj(x)))
        return out



class MHAtt(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(MHAtt, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / head)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim,hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask=None):
        b, n, s = q.shape

        v = self.linear_v(v).view(b, -1, self.head, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(b, -1, self.head, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(b, -1, self.head, self.head_size).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(b, -1, self.hidden_dim)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -65504.0)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ################## avact ##################
class att(nn.Module):
    def __init__(self, dim):
        super(att, self).__init__()

        self.attention = MHAtt(dim)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        return self.norm(self.dropout(self.attention(kv, kv, q)) + q)

class ffn(nn.Module):
    def __init__(self, dim):
        super(ffn, self).__init__()

        self.ffn = PositionWiseFFN(dim, 0.2, dim)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q):
        return self.norm(self.dropout(self.ffn(q)) + q)

class GumbelTopKMask(nn.Module):
    """
    可微 TopK：训练期用 Gumbel-Softmax 生成 soft mask；推理期退化为硬 top-k。
    ratio: 选取比例 (0,1]
    """
    def __init__(self, ratio: float = 0.3, hard: bool = False, tau: float = 1.0):
        super().__init__()
        self.ratio = ratio
        self.hard = hard
        self.tau = nn.Parameter(torch.tensor(tau))  # 可学习温度

    def forward(self, x):
        # x: [B*T, L, C] -> saliency by ||·||^2 over C
        sal = x.pow(2).sum(dim=-1)  # [BT, L]
        BT, L = sal.shape
        k = max(1, int(L * self.ratio))

        if self.training and not self.hard:
            # gumbel noise
            g = -torch.empty_like(sal).exponential_().log()
            logits = (sal + g) / self.tau.clamp(min=1e-3)
            # soft top-k via continuous relaxation
            # 这里用 softmax 近似 top-k 分配（实测稳定且足够）
            probs = F.softmax(logits, dim=-1)  # [BT, L]
            # 放大前 k 名权重：scale 使其更接近 one-hot topk
            topk_val, topk_idx = probs.topk(k, dim=1)
            thr = topk_val[:, -1:].detach()
            mask = (probs >= thr).float()
            # 连续化：mask 作为上界，probs 作为权重
            soft_mask = (probs * mask) / (probs * mask).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            return soft_mask.unsqueeze(-1), probs  # [BT,L,1], [BT,L]
        else:
            # inference: hard top-k binary mask
            topk_val, topk_idx = sal.topk(k, dim=1)
            mask = torch.zeros_like(sal)
            mask.scatter_(1, topk_idx, 1.0)
            mask = mask / mask.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            return mask.unsqueeze(-1), None


class TemporalDWConv1D(nn.Module):
    """
    只沿 T 做 depthwise + pointwise conv，捕捉 t-1,t,t+1 的局部依赖；保持 [BT,L,C] 接口。
    """
    def __init__(self, embed_dim: int, T: int = 10, k: int = 3, dilation: int = 1):
        super().__init__()
        self.T = T
        self.k = k
        self.dw = nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=dilation, dilation=dilation, groups=embed_dim)
        self.pw = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: [BT, L, C]
        BT, L, C = x.shape
        B = BT // self.T
        x_ = x.view(B, self.T, L, C).permute(0,2,3,1).contiguous()    # [B,L,C,T]
        y = x_.reshape(B*L, C, self.T)
        y = self.pw(self.dw(y)).reshape(B, L, C, self.T).permute(0,3,1,2).reshape(BT, L, C)
        return self.norm(y)


class CASE(nn.Module):
    """
    升级版 AvAct：
    - caption 强锚点（caption 做 query 更频繁）
    - 可微 TopK 注入
    - 投影头 + learnable temperature 用于对比损失
    - 轻量时序 depthwise conv
    - 注意力熵正则项输出
    """
    def __init__(self, v_dim, a_dim, proj_dim=512, T=10, topk_ratio=0.3):
        super().__init__()
        self.T = T
        self.topk_ratio = topk_ratio

        # 自注意 + 交叉注意
        self.sa_v = att(v_dim);   self.sa_a = att(a_dim)
        self.sa_vcap = att(v_dim); self.sa_acap = att(a_dim)

        self.ca_v_qcap = att(v_dim)  # caption->video (cap as Q)
        self.ca_a_qcap = att(a_dim)  # caption->audio (cap as Q)
        self.ca_vqa = att(v_dim)     # audio->video
        self.ca_aqv = att(a_dim)     # video->audio

        # FFN
        self.ffn_v = ffn(v_dim); self.ffn_a = ffn(a_dim)
        self.ffn_vcap = ffn(v_dim); self.ffn_acap = ffn(a_dim)

        # 可微 TopK
        self.topk = GumbelTopKMask(ratio=topk_ratio, hard=False, tau=1.0)

        # 双向线性对齐（维度互投）
        self.linear_v_from_a = nn.Linear(a_dim, v_dim)
        self.linear_a_from_v = nn.Linear(v_dim, a_dim)

        # 投影头（对比学习）
        self.proj_v = nn.Sequential(nn.LayerNorm(v_dim), nn.Linear(v_dim, proj_dim))
        self.proj_a = nn.Sequential(nn.LayerNorm(a_dim), nn.Linear(a_dim, proj_dim))
        self.proj_vcap = nn.Sequential(nn.LayerNorm(v_dim), nn.Linear(v_dim, proj_dim))
        self.proj_acap = nn.Sequential(nn.LayerNorm(a_dim), nn.Linear(a_dim, proj_dim))
        self.tau = nn.Parameter(torch.tensor(0.2))  # InfoNCE 温度

        # 时序轻模组
        self.tmp_v = TemporalDWConv1D(v_dim, T=T, k=3, dilation=1)
        self.tmp_a = TemporalDWConv1D(a_dim, T=T, k=3, dilation=1)

        self.norm_v = nn.LayerNorm(v_dim)
        self.norm_a = nn.LayerNorm(a_dim)

    def forward(self, v, a, v_cap, a_cap):
        """
        v, a: [BT, L, C]
        v_cap, a_cap: [BT, Lc, C]  (你外部已 repeat 到 BT)
        返回：更新后的 v, a, 以及对比/正则用的中间量
        """
        v0, a0 = v, a

        # (1) intra
        v = self.sa_v(v, v); v_cap = self.sa_vcap(v_cap, v_cap)
        a = self.sa_a(a, a); a_cap = self.sa_acap(a_cap, a_cap)

        # (2) 用“v作Q、v_cap作KV”的方式保持长度为L，并做残差注入
        v_cap2v = self.ca_v_qcap(v, v_cap)        # q=v, kv=v_cap, 输出长度= L
        a_cap2a = self.ca_a_qcap(a, a_cap)
        v = v_cap2v
        a = a_cap2a

        # (3) 再做跨模态细化，也走残差+小门控
        v_refined = self.ca_vqa(v, self.linear_v_from_a(a))
        a_refined = self.ca_aqv(a, self.linear_a_from_v(v))
        v = v_refined
        a = a_refined

        # (4) FFN + 轻时序
        v = self.ffn_v(v); a = self.ffn_a(a)
        v = self.tmp_v(v); a = self.tmp_a(a)
        v_cap = self.ffn_vcap(v_cap); a_cap = self.ffn_acap(a_cap)

        # (5) 帧级一致性门：由 cap 与帧的一致性确定注入强度
        with torch.no_grad():
            cap_v_sim = (F.normalize(v.mean(1), dim=-1) * F.normalize(v_cap.mean(1), dim=-1)).sum(-1, keepdim=True)  # [BT,1]
            cap_a_sim = (F.normalize(a.mean(1), dim=-1) * F.normalize(a_cap.mean(1), dim=-1)).sum(-1, keepdim=True)
            w_cap_v = torch.sigmoid(cap_v_sim).unsqueeze(-1)   # [BT,1,1]
            w_cap_a = torch.sigmoid(cap_a_sim).unsqueeze(-1)

        # (6) 可微 TopK 掩膜，选择注入位点（逐帧逐 token）
        mask_v, prob_v = self.topk(v)   # [BT,L,1], [BT,L] (prob for entropy reg)
        mask_a, prob_a = self.topk(a)

        # (7) 只对 TopK token 做残差注入（caption 驱动）
        v = v0 + (w_cap_v * v * mask_v + (1 - w_cap_v) * v)
        a = a0 + (w_cap_a * a * mask_a + (1 - w_cap_a) * a)

        v = self.norm_v(v)
        a = self.norm_a(a)

        # (8) 投影特征（给 contrastive）
        Pv = self.proj_v(v.mean(1))          # [BT, D]
        Pa = self.proj_a(a.mean(1))
        Pvcap = self.proj_vcap(v_cap.mean(1))
        Pacap = self.proj_acap(a_cap.mean(1))

        aux = {
            "Pv": Pv, "Pa": Pa, "Pvcap": Pvcap, "Pacap": Pacap,
            "prob_v": prob_v, "prob_a": prob_a,  # 用于熵正则
            "tau": self.tau
        }
        return v, a, v_cap, a_cap, aux




class MobileViTv2Attention(nn.Module):
    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input)  # (bs,nq,1)
        weight_i = torch.softmax(i, dim=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(input)  # bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True)  # bs,1,d_model
        v = self.fc_v(input) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model

        return out


# 空间富集v2
class SpatialEnrichmentLite(nn.Module):
    def __init__(self, embed_dim: int, T: int = 10, drop: float = 0.0):
        super().__init__()
        self.T = T
        self.attn = MobileViTv2Attention(embed_dim)  # 每个时间T都共享了参数，来减少显存
        self.norm = nn.LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.zeros(1))    # 残差门控，init=0
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # x: [B*T, L, C]
        BT, L, C = x.shape
        # assert BT % self.T == 0, "T 不整除，请确认每条视频是 T 帧"
        B = BT // self.T
        x_ = x.view(B, self.T, L, C)

        out = []
        for t in range(self.T):
            out.append(self.attn(x_[:, t]))          # [B, L, C] -> [B, L, C]
        out = torch.stack(out, dim=1)                # [B, T, L, C]
        out = out.view(BT, L, C)
        out = self.dropout(out)
        return x + self.gamma * self.norm(out)


# 时间富集v3
"""
使用： mode为tsm是只用 TSM（最省显存、最稳）；为dwconv是v2；为tsm+dw是先tsm再dw
self.vis_temporal_enrich_blocks_p1 = nn.ModuleList([
    TemporalEnrichmentLite(embed_dim=hidden_list[i], T=self.T, mode="tsm+dw", fold_div=8)
    for i in range(len(hidden_list))
])
self.audio_temporal_enrich_blocks_p1 = nn.ModuleList([
    TemporalEnrichmentLite(embed_dim=hidden_list_a[i], T=self.T, mode="tsm+dw", fold_div=8)
    for i in range(len(hidden_list_a))
])

"""
class TemporalEnrichmentLite(nn.Module):
    """
    x: [B*T, L, C]  (B: batch, T: 帧数, L: token数, C: 嵌入维)
    mode: "dwconv" | "tsm" | "tsm+dw"
    """
    def __init__(self, embed_dim: int, T: int = 10, mode: str = "dwconv", fold_div: int = 8):
        super().__init__()
        assert mode in ["dwconv", "tsm", "tsm+dw"]
        self.T = T
        self.mode = mode
        self.fold_div = fold_div

        # DW-Conv（仅在需要时用到）
        if mode in ["dwconv", "tsm+dw"]:
            self.dw = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
            self.pw = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        self.norm = nn.LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # 残差门控，init=0

    @torch.no_grad()
    def _tsm_inplace(self, x_bt_l_c):
        """
        原地 TSM：将一部分通道向前/向后平移一帧
        x_bt_l_c: [B*T, L, C]
        """
        BT, L, C = x_bt_l_c.shape
        assert BT % self.T == 0, "T 不整除，请确认帧数"
        B = BT // self.T
        x = x_bt_l_c.view(B, self.T, L, C)

        fold = max(1, C // self.fold_div)
        out = x.clone()

        # 向后移（t -> t+1）
        out[:, 1:, :, :fold] = x[:, :-1, :, :fold]
        out[:, 0,  :, :fold] = 0

        # 向前移（t -> t-1）
        out[:, :-1, :, fold:2*fold] = x[:, 1:, :, fold:2*fold]
        out[:, -1,  :, fold:2*fold] = 0

        # 其余通道不动
        return out.view(BT, L, C)

    def _dwconv_time(self, x_bt_l_c):
        """
        DW-Conv1D 沿时间：保证只用 t-1, t, t+1
        """
        BT, L, C = x_bt_l_c.shape
        assert BT % self.T == 0
        B = BT // self.T
        x_ = x_bt_l_c.view(B, self.T, L, C).permute(0, 2, 3, 1).contiguous()  # [B, L, C, T]
        y = x_.view(B * L, C, self.T)                                         # [B*L, C, T]
        y = self.pw(self.dw(y))                                               # [B*L, C, T]
        y = y.view(B, L, C, self.T).permute(0, 3, 1, 2).contiguous()          # [B, T, L, C]
        return y.view(BT, L, C)

    def forward(self, x):
        # x: [B*T, L, C]
        if self.mode == "tsm":
            y = self._tsm_inplace(x)
        elif self.mode == "dwconv":
            y = self._dwconv_time(x)
        else:  # "tsm+dw"
            y = self._tsm_inplace(x)
            y = self._dwconv_time(y)
        return x + self.gamma * self.norm(y)


class AgreementGate(nn.Module):
    """
    逐帧跨模态一致性门控：
    将 V/A 映射到共享维度后做帧级 prototype（mean-pool），
    计算 cosine 一致性，再用小 MLP 输出 [w_spatial, w_temporal]。
    """
    def __init__(self, v_dim: int, a_dim: int, proj_dim: int = 128, T: int = 10):
        super().__init__()
        self.T = T
        self.v_proj = nn.Linear(v_dim, proj_dim)
        self.a_proj = nn.Linear(a_dim, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * proj_dim, proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim // 2, 2)
        )

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor):
        """
        x_v: [B*T, L_v, C_v]
        x_a: [B*T, L_a, C_a]
        return: w ∈ [B*T, 2]（softmax），第一维=空间权重，第二维=时间权重
        """
        BT, Lv, Cv = x_v.shape
        _,  La, Ca = x_a.shape
        assert BT == x_a.size(0), "V/A batch不匹配"
        assert BT % self.T == 0, "T 不整除，请确认帧数"

        v_proto = self.v_proj(x_v.mean(dim=1))   # [BT, D]
        a_proto = self.a_proj(x_a.mean(dim=1))   # [BT, D]

        v_norm = F.normalize(v_proto, dim=-1)
        a_norm = F.normalize(a_proto, dim=-1)
        agree = (v_norm * a_norm).sum(-1, keepdim=True)  # [BT,1], cosine ∈ [-1,1]

        # 更丰富的门控输入：拼接 v，a，差分，Hadamard 积
        feat = torch.cat([v_proto, a_proto, v_proto - a_proto, v_proto * a_proto], dim=-1)  # [BT, 4D]
        logits = self.mlp(feat)  # [BT,2]
        # 轻度偏置：一致性越高越偏空间（稳定），越低越偏时间（稳健）
        logits = logits + torch.cat([agree, -agree], dim=-1)
        w = F.softmax(logits, dim=-1)
        return w  # [BT, 2]


def token_topk_mask(x: torch.Tensor, ratio: float = 0.3):
    """
    x: [B*T, L, C]
    返回逐帧逐 token 的权重 mask ∈ [B*T, L, 1]，Top-K 为1，其余为0（或软权重）
    """
    BT, L, C = x.shape
    k = max(1, int(L * ratio))
    # 简单用 L2 能量做 saliency
    sal = x.pow(2).sum(dim=-1)  # [BT, L]
    topk_val, topk_idx = sal.topk(k, dim=1)
    mask = torch.zeros_like(sal, dtype=x.dtype)
    mask.scatter_(1, topk_idx, 1.0)
    return mask.unsqueeze(-1)  # [BT, L, 1]


class CASTEBlock(nn.Module):
    """
    Cross-modal Agreement-guided SpatioTemporal Enrichment
    复用你已有的 SpatialEnrichmentLite / TemporalEnrichmentLite，
    用 AgreementGate 产生空间/时间的混合系数，并做 Top-K 选择性注入。
    """
    def __init__(self, v_dim: int, a_dim: int, T: int = 10,
                 topk_ratio: float = 0.3,
                 use_tsm: bool = False):
        super().__init__()
        self.T = T
        self.topk_ratio = topk_ratio
        self.gate = AgreementGate(v_dim=v_dim, a_dim=a_dim, proj_dim=128, T=T)

        # 你代码里最后定义的 TemporalEnrichmentLite 支持 mode
        self.spatial = SpatialEnrichmentLite(embed_dim=v_dim, T=T, drop=0.0)
        self.temporal = TemporalEnrichmentLite(embed_dim=v_dim, T=T,
                                               mode=("tsm+dw" if use_tsm else "dwconv"),
                                               fold_div=8)
        # 额外的残差门控，0 初始化
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor):
        """
        x_v: 当前分支的特征 [B*T, L_v, C_v]
        x_a: 另一模态特征   [B*T, L_a, C_a]
        返回：residual，shape 同 x_v
        """
        BT, Lv, Cv = x_v.shape
        w = self.gate(x_v, x_a)                   # [BT,2]
        v_sp = self.spatial(x_v)                  # [BT, L, C]（已带 LayerNorm + gamma）
        v_tm = self.temporal(x_v)                 # [BT, L, C]（已带 LayerNorm + gamma）

        # 帧级混合
        w_sp = w[:, 0].view(BT, 1, 1)
        w_tm = w[:, 1].view(BT, 1, 1)
        mixed = w_sp * v_sp + w_tm * v_tm         # [BT, L, C]

        # 只对 Top-K token 注入
        mask = token_topk_mask(x_v, ratio=self.topk_ratio)  # [BT, L, 1]
        residual = mixed * mask

        # 额外一层全局残差门控（0-init，训练中慢慢学）
        return self.gamma * residual



class Pred_endecoder(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, channel=256,opt=None, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True):
        super(Pred_endecoder, self).__init__()
        self.cfg = config
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag
        self.vis_dim = vis_dim

        self.opt = opt
        
        self.relu = nn.ReLU(inplace=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[3])
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[2])
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[1])
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[0])

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)



        self.x1_linear = nn.Linear(192,64)
        self.x2_linear = nn.Linear(384,128)
        self.x3_linear = nn.Linear(768,320)
        self.x4_linear = nn.Linear(1536,512)

        self.x1_linear_ = nn.Linear(192,256)
        self.x2_linear_ = nn.Linear(384,256)
        self.x3_linear_ = nn.Linear(768,256)
        self.x4_linear_ = nn.Linear(1536,256)
        
        self.audio_linear = nn.Linear(768,128)

        self.encoder_backbone = pvt_v2_b5()
        self.temporal_attn = TemporalAttention()
        
        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
        if opt.backbone_type == "esc-50":
            esc_config.dataset_path = "your processed ESC-50 folder"
            esc_config.dataset_type = "esc-50"
            esc_config.loss_type = "clip_ce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320
            esc_config.classes_num = 50
            esc_config.checkpoint_path = "/home/hyz/workspace/codes/AV/AVMOE/AVMOE/checkpoints/ESC-50/"
            esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
        elif opt.backbone_type == "audioset":
            esc_config.dataset_path = "your processed audioset folder"
            esc_config.dataset_type = "audioset"
            esc_config.balanced_data = True
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320
            esc_config.classes_num = 527
            esc_config.checkpoint_path = "/home/hyz/workspace/codes/AV/AVMOE/AVMOE/checkpoints/AudioSet/"
            esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
        elif opt.backbone_type == "scv2":
            esc_config.dataset_path = "your processed SCV2 folder"
            esc_config.dataset_type = "scv2"
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 16000
            esc_config.hop_size = 160
            esc_config.classes_num = 35
            esc_config.checkpoint_path = "/home/hyz/workspace/codes/AV/AVMOE/AVMOE/checkpoints/SCV2/"
            esc_config.checkpoint = "HTSAT_SCV2_Saved_2.ckpt"
        else:
            raise NotImplementedError
    
        self.htsat = HTSAT_Swin_Transformer(
            spec_size=esc_config.htsat_spec_size,
            patch_size=esc_config.htsat_patch_size,
            in_chans=1,
            num_classes=esc_config.classes_num,
            window_size=esc_config.htsat_window_size,
            config = esc_config,
            depths = esc_config.htsat_depth,
            embed_dim = esc_config.htsat_dim,
            patch_stride=esc_config.htsat_stride,
            num_heads=esc_config.htsat_num_head
        )
        
        checkpoint_path = os.path.join(esc_config.checkpoint_path, esc_config.checkpoint)
        tmp = torch.load(checkpoint_path, map_location='cpu')
        tmp = {k[10:]:v for k, v in tmp['state_dict'].items()}
        self.htsat.load_state_dict(tmp, strict=True)

        hidden_list, hidden_list_a = [], []
        down_in_dim, down_in_dim_a = [], []
        down_out_dim, down_out_dim_a = [], []
        conv_dim, conv_dim_a = [], []
        
        ## ------------> for swin and htsat 
        for idx_layer, (my_blk, my_blk_a) in enumerate(zip(self.swin.layers, self.htsat.layers)):
            if self.opt.num_skip>1:
                if (idx_layer+1) % self.opt.num_skip == 0:
                    continue
            conv_dim_tmp = (my_blk.input_resolution[0]*my_blk.input_resolution[1])
            conv_dim_tmp_a = (my_blk_a.input_resolution[0]*my_blk_a.input_resolution[1])
            if not isinstance(my_blk.downsample, nn.Identity):
                down_in_dim.append(my_blk.downsample.reduction.in_features)
                down_out_dim.append(my_blk.downsample.reduction.out_features)
            if my_blk_a.downsample is not None:
                down_in_dim_a.append(my_blk_a.downsample.reduction.in_features)
                down_out_dim_a.append(my_blk_a.downsample.reduction.out_features)
            
            for blk, blk_a in zip(my_blk.blocks, my_blk_a.blocks):
                hidden_d_size = blk.norm1.normalized_shape[0]
                hidden_list.append(hidden_d_size)
                conv_dim.append(conv_dim_tmp)
                hidden_d_size_a = blk_a.norm1.normalized_shape[0]
                hidden_list_a.append(hidden_d_size_a)
                conv_dim_a.append(conv_dim_tmp_a)


        self.audio_moe_adapter_blocks_p1 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck",dim_list=hidden_list_a, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i], linear_in=hidden_list[i], linear_out=hidden_list_a[i])
            for i in range(len(hidden_list_a))])

        self.vis_moe_adapter_blocks_p1 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i], linear_in=hidden_list_a[i], linear_out=hidden_list[i])
            for i in range(len(hidden_list))])

        self.audio_moe_adapter_blocks_p2 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck", dim_list=hidden_list_a, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i], linear_in=hidden_list[i], linear_out=hidden_list_a[i])
            for i in range(len(hidden_list_a))])

        self.vis_moe_adapter_blocks_p2 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i], linear_in=hidden_list_a[i], linear_out=hidden_list[i] )
            for i in range(len(hidden_list))])

        for i in self.tpavi_stages:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel, mode='dot'))
            print("==> Build TPAVI block...")

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        if self.training:
            self.initialize_pvt_weights()

        # 帧数 T
        self.T = 5

        # 映射器
        self.mapper = FeatureMapper()

        # Adapter
        self.frame_adapter_v = Adapter(output_dim=1536)
        self.wav_adapter_a = Adapter(output_dim=768)

        # case
        self.case = CASE(1536, 768, proj_dim=512, T=self.T, topk_ratio=0.3)


        # ====== 新增：收集“按 layer”的维度（只保留未被 num_skip 跳过的 layer）======
        layer_hidden_list, layer_hidden_list_a = [], []
        for idx_layer, (my_blk, my_blk_a) in enumerate(zip(self.swin.layers, self.htsat.layers)):
            # 四层都加
            layer_hidden_list.append(my_blk.blocks[0].norm1.normalized_shape[0])
            layer_hidden_list_a.append(my_blk_a.blocks[0].norm1.normalized_shape[0])

        print(layer_hidden_list)

        # ====== 用 CASTE 替换原先的 p1 空/时 + TTGate ======
        self.vis_caste_blocks_p1 = nn.ModuleList([
            CASTEBlock(v_dim=layer_hidden_list[i],
                    a_dim=layer_hidden_list_a[i],
                    T=self.T,
                    topk_ratio=0.3,       # 可调：0.25~0.35 较稳，默认0.3
                    use_tsm=False)        # 先禁 TSM，后面再尝试 True
            for i in range(len(layer_hidden_list))
        ])
        self.audio_caste_blocks_p1 = nn.ModuleList([
            CASTEBlock(v_dim=layer_hidden_list_a[i],
                    a_dim=layer_hidden_list[i],
                    T=self.T,
                    topk_ratio=0.3,
                    use_tsm=False)
            for i in range(len(layer_hidden_list_a))
        ])



    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        try:
            x = x.reshape(-1, 5, C, H, W)
        except:
            print("pre_reshape_for_tpavi: ", x.shape)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, _ = tpavi_b(x) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x

    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        try:
            audio = audio.view(-1, 5, audio.shape[-1]) # [B, T, 128]
        except:
            print("tpavi_va: ", audio.shape)
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, audio_feature, frame_cap, wave_cap, mixup_lambda=None, is_training=False):
        B, frame, C, H, W = x.shape
        x = x.view(B*frame, C, H, W)
        audio = audio_feature
        audio = audio.view(audio.size(0)*audio.size(1), -1)
        waveform = audio

        x = F.interpolate(x, mode='bicubic',size=[192,192])
        f_v = self.swin.patch_embed(x)
        audio = self.htsat.spectrogram_extractor(audio)
        audio = self.htsat.logmel_extractor(audio)        
        audio = audio.transpose(1, 3)
        audio = self.htsat.bn0(audio)
        audio = audio.transpose(1, 3)
        if self.htsat.training:
            audio = self.htsat.spec_augmenter(audio)
        if self.htsat.training and mixup_lambda is not None:
            audio = do_mixup(audio, mixup_lambda)

        if audio.shape[2] > self.htsat.freq_ratio * self.htsat.spec_size:
            audio = self.htsat.crop_wav(audio, crop_size=self.htsat.freq_ratio * self.htsat.spec_size)
            audio = self.htsat.reshape_wav2img(audio)
        else: # this part is typically used, and most easy one
            audio = self.htsat.reshape_wav2img(audio)
        frames_num = audio.shape[2]
        f_a = self.htsat.patch_embed(audio)
        if self.htsat.ape:
            f_a = f_a + self.htsat.absolute_pos_embed
        f_a = self.htsat.pos_drop(f_a)   
        
        idx_layer = 0
        multi_scale = []
        out_idx_layer = 0
        idx_block = 0
        adapter_index_dict = {'audio': {'p1': [], 'p2': []}, 'video': {'p1': [], 'p2': []}}
        adapter_probs_dict = {'audio': {'p1': [], 'p2': []}, 'video': {'p1': [], 'p2': []}}
        total_load_balancing_loss = 0

        enrich_idx = 0

        for layer_index, (my_blk, htsat_blk) in enumerate(zip(self.swin.layers, self.htsat.layers)) :

            if len(my_blk.blocks) == len(htsat_blk.blocks):
                aud_blocks = htsat_blk.blocks
            else:
                aud_blocks = [None, None, htsat_blk.blocks[0], None, None, htsat_blk.blocks[1], None, None, htsat_blk.blocks[2], None, None, htsat_blk.blocks[3], None, None, htsat_blk.blocks[4], None, None, htsat_blk.blocks[5]]
                assert len(aud_blocks) == len(my_blk.blocks)

            block_first_layer = True
                
            for (blk, blk_a) in zip(my_blk.blocks, aud_blocks):
                if blk_a is not None:
                    if self.opt.num_skip>1 and (((layer_index+1) % self.opt.num_skip) == 0):
                        f_v_ori, f_a_ori = f_v, f_a
                        f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))

                        f_a, _ = blk_a(f_a)

                        if block_first_layer:
                            # ====> 换成 CASTE（逐层一次，逐帧门控 + TopK 注入）
                            block_first_layer = False
                            v_res = self.vis_caste_blocks_p1[enrich_idx](f_v_ori, f_a_ori)
                            a_res = self.audio_caste_blocks_p1[enrich_idx](f_a_ori, f_v_ori)
                            f_v = f_v + v_res
                            f_a = f_a + a_res

                            enrich_idx = enrich_idx + 1


                        f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                    else:
                        f_v_ori, f_a_ori = f_v, f_a
                        if block_first_layer:
                            # ====> 换成 CASTE（逐层一次，逐帧门控 + TopK 注入）
                            block_first_layer = False
                            v_res = self.vis_caste_blocks_p1[enrich_idx](f_v, f_a)
                            a_res = self.audio_caste_blocks_p1[enrich_idx](f_a, f_v)
                            f_v = f_v + v_res
                            f_a = f_a + a_res

                            enrich_idx = enrich_idx + 1

                        f_a_res, f_a_moe_adapter_index_p1, f_a_moe_adapter_probs_p1, f_a_load_balancing_loss_p1 = self.audio_moe_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1), is_training)
                        f_v_res, f_v_moe_adapter_index_p1, f_v_moe_adapter_probs_p1, f_v_load_balancing_loss_p1= self.vis_moe_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1), is_training)
                        total_load_balancing_loss += f_a_load_balancing_loss_p1
                        total_load_balancing_loss += f_v_load_balancing_loss_p1

                        adapter_index_dict['audio']['p1'].append(f_a_moe_adapter_index_p1.squeeze().tolist())
                        adapter_index_dict['video']['p1'].append(f_v_moe_adapter_index_p1.squeeze().tolist())
                        adapter_probs_dict['audio']['p1'].append(f_a_moe_adapter_probs_p1.squeeze().tolist())
                        adapter_probs_dict['video']['p1'].append(f_v_moe_adapter_probs_p1.squeeze().tolist())
        
                        f_v = f_v_ori + blk.drop_path1(blk.norm1(blk._attn(f_v_ori)))
                        f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

                        f_a, _ = blk_a(f_a_ori)
                        f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
                    
                        f_a_res, f_a_moe_adapter_index_p2, f_a_moe_adapter_probs_p2, f_a_load_balancing_loss_p2 = self.audio_moe_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1), is_training)
                        f_v_res, f_v_moe_adapter_index_p2, f_v_moe_adapter_probs_p2, f_v_load_balancing_loss_p2 = self.vis_moe_adapter_blocks_p2[idx_layer]( f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1), is_training)
                        total_load_balancing_loss += f_a_load_balancing_loss_p2
                        total_load_balancing_loss += f_v_load_balancing_loss_p2

                        adapter_index_dict['audio']['p2'].append(f_a_moe_adapter_index_p2.squeeze().tolist())
                        adapter_index_dict['video']['p2'].append(f_v_moe_adapter_index_p2.squeeze().tolist())
                        adapter_probs_dict['audio']['p2'].append(f_a_moe_adapter_probs_p2.squeeze().tolist())
                        adapter_probs_dict['video']['p2'].append(f_v_moe_adapter_probs_p2.squeeze().tolist())

                        f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                        f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

                        f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
                        
                        idx_layer = idx_layer +1
                else:
                    f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                    f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
            if idx_block != 3:
                multi_scale.append(f_v)
            else:
                multi_scale.append(self.swin.norm(f_v))
            idx_block += 1                
         
            f_v = my_blk.downsample(f_v)
            if htsat_blk.downsample is not None:
                f_a = htsat_blk.downsample(f_a)


        ## case
        frame_cap = self.mapper(frame_cap)
        frame_cap_v = self.frame_adapter_v(frame_cap)
        wave_cap = self.mapper(wave_cap)
        wav_cap_a = self.wav_adapter_a(wave_cap)
        wav_cap_a = torch.repeat_interleave(wav_cap_a, repeats=self.T, dim=0)
        frame_cap_v = torch.repeat_interleave(frame_cap_v, repeats=self.T, dim=0)
        a_cap = wav_cap_a
        v_cap = frame_cap_v
        f_v, f_a, v_cap, a_cap, case_aux = self.case(f_v, f_a, v_cap, a_cap)


        f_v = self.swin.norm(f_v)

        

        audio_feature = rearrange(f_a.mean(dim=1), '(b t) d -> b t d', t=5)
        # audio_feature = rearrange(torch.bmm(f_a_spatial_att_maps, f_a).squeeze(dim=1), '(b t) d -> b t d', t=5)
        audio_feature = self.audio_linear(audio_feature)
        visual_feature = f_v.mean(dim=1, keepdim=True)
        # print("f_v.shape: ", f_v.shape)
        audio_feature_reshape = audio_feature.view(audio_feature.size(1), audio_feature.size(0), -1)

        x1 = multi_scale[0].view(multi_scale[0].size(0),48,48,-1)
        x2 = multi_scale[1].view(multi_scale[1].size(0),24,24,-1)
        x3 = multi_scale[2].view(multi_scale[2].size(0),12,12,-1)
        x4 = multi_scale[3].view(multi_scale[3].size(0),6,6,-1)
        x1 = self.x1_linear_(x1)
        x2 = self.x2_linear_(x2)
        x3 = self.x3_linear_(x3)
        x4 = self.x4_linear_(x4)

        x1 = F.interpolate(rearrange(x1, 'BF w h c -> BF c w h'), mode='bicubic',size=[56,56])
        x2 = F.interpolate(rearrange(x2, 'BF w h c -> BF c w h'), mode='bicubic',size=[28,28])
        x3 = F.interpolate(rearrange(x3, 'BF w h c -> BF c w h'), mode='bicubic',size=[14,14])
        x4 = F.interpolate(rearrange(x4, 'BF w h c -> BF c w h'), mode='bicubic',size=[7,7])

        conv1_feat = x1    # BF x 256 x 56 x 56
        conv2_feat = x2    # BF x 256 x 28 x 28
        conv3_feat = x3    # BF x 256 x 14 x 14
        conv4_feat = x4    # BF x 256 x  7 x  7
 
        ############## Temporal Attention #################
        feature_map_list, audio_feature = self.temporal_attn([x1, x2, x3, x4], audio_feature)
        
        # feature_map_list = [conv1_feat, conv2_feat, conv3_feat, conv4_feat]
        a_fea_list = [None] * 4

        if len(self.tpavi_stages) > 0:
            if (not self.tpavi_vv_flag) and (not self.tpavi_va_flag):
                raise Exception('tpavi_vv_flag and tpavi_va_flag cannot be False at the same time if len(tpavi_stages)>0, \
                    tpavi_vv_flag is for video self-attention while tpavi_va_flag indicates the standard version (audio-visual attention)')
            for i in self.tpavi_stages:
                tpavi_count = 0
                conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
                if self.tpavi_vv_flag:
                    conv_feat_vv = self.tpavi_vv(feature_map_list[i], stage=i)
                    conv_feat += conv_feat_vv
                    tpavi_count += 1
                if self.tpavi_va_flag:
                    conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], audio_feature, stage=i)
                    conv_feat += conv_feat_va
                    tpavi_count += 1
                    a_fea_list[i] = a_fea
                conv_feat /= tpavi_count
                feature_map_list[i] = conv_feat # update features of stage-i which conduct non-local

        conv4_feat = self.path4(feature_map_list[3])            # BF x 256 x 14 x 14
        conv43 = self.path3(conv4_feat, feature_map_list[2])    # BF x 256 x 28 x 28
        conv432 = self.path2(conv43, feature_map_list[1])       # BF x 256 x 56 x 56
        conv4321 = self.path1(conv432, feature_map_list[0])     # BF x 256 x 112 x 112

        pred = self.output_conv(conv4321)   # BF x 1 x 224 x 224
        return pred, feature_map_list, a_fea_list, adapter_index_dict, adapter_probs_dict, total_load_balancing_loss, visual_feature, audio_feature_reshape, audio_feature, case_aux


    def initialize_pvt_weights(self,):
        pvt_model_dict = self.encoder_backbone.state_dict()
        pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_PVTV2_PATH)
        state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(state_dict)
        self.encoder_backbone.load_state_dict(pvt_model_dict)
        print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
        # pdb.set_trace()


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    audio = torch.randn(2, 5, 128)
    # model = Pred_endecoder(channel=256)
    model = Pred_endecoder(channel=256, tpavi_stages=[0,1,2,3], tpavi_va_flag=True,)
    # output = model(imgs)
    output = model(imgs, audio)
    pdb.set_trace()