import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
from ipdb import set_trace
import os

from torch import Tensor
from typing import Optional, Any
from einops import rearrange, repeat

from timm.models.vision_transformer import Attention
import timm
import loralib as lora

from transformers.activations import get_activation
from .models import *
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint

### VGGSound
from nets import Resnet_VGGSound
from .htsat import HTSAT_Swin_Transformer
import nets.esc_config as esc_config
from .utils import do_mixup, get_mix_lambda, do_mixup_label

from torch.nn import init


# torch.manual_seed(0)
# np.random.seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)

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


class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class + 1, kernel_size=1, stride=1, padding=0,
                      bias=False)
        )

    def forward(self, content):
        content = content.permute(0, 2, 1)

        out = self.classifier(content)
        out = out.permute(0, 2, 1)
        return out


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        # self.cas_model = CAS_Module(d_model=d_model, num_class=28)

    # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        # class_logits = self.event_classifier(fused_content.transpose(1,0))
        # sorted_scores_base,_ = class_logits.sort(descending=True, dim=1)
        # topk_scores_base = sorted_scores_base[:, :4, :]
        # class_logits = torch.mean(topk_scores_base, dim=1)
        class_scores = class_logits

        return logits, class_scores


class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim  # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1)  # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores


class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output


class TemporalAttention(nn.Module):
    def __init__(self, opt):
        super(TemporalAttention, self).__init__()
        self.opt = opt
        self.beta = 0.4
        self.video_input_dim = 512
        self.audio_input_dim = 128

        self.video_fc_dim = 512
        self.audio_fc_dim = 128
        self.d_model = 256

        if self.opt.model_size=="large":
            input_dim = 1536
        else:
            input_dim = 1024
        self.v_fc = nn.Linear(input_dim, self.video_fc_dim)
        self.a_fc = nn.Linear(768, self.audio_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_input_dim, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim,
                                                 d_model=self.d_model, num_layers=1)

        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

        self.alpha = 0.1
        self.gamma = 0.1

    def forward(self, visual_feature, audio_feature):
        audio_feature = self.a_fc(audio_feature)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        visual_rnn_input = visual_feature

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]

        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)

        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)

        audio_visual_gate = audio_gate * video_gate

        video_query_output = video_query_output + audio_gate * video_query_output * self.gamma
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.gamma

        return video_query_output, audio_query_output, audio_visual_gate


class CMBS(nn.Module):
    def __init__(self, config):
        super(CMBS, self).__init__()
        self.config = config
        self.beta = 0.4
        self.d_model = 256
        if self.config.is_inter_in_cmbs == 1:
            self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
            self.VAInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.localize_module = SupvLocalizeModule(self.d_model)
        self.audio_cas = nn.Linear(self.d_model, 28)
        self.video_cas = nn.Linear(self.d_model, 28)
        self.alpha = 0.1
        self.gamma = 0.3

    def forward(self, visual_feature, audio_feature):
        video_cas = self.video_cas(visual_feature)  # [10, 32, 28]
        audio_cas = self.audio_cas(audio_feature)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        topk_scores_video = sorted_scores_video[:, :4, :]
        score_video = torch.mean(topk_scores_video, dim=1)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 28]

        av_score = (score_video + score_audio) / 2

        if self.config.is_inter_in_cmbs == 1:
            video_query_output = self.AVInter(visual_feature, audio_feature)
            audio_query_output = self.VAInter(audio_feature, visual_feature)
            visual_feature = video_query_output
            audio_feature = audio_query_output
        is_event_scores, event_scores = self.localize_module((visual_feature + audio_feature) / 2)
        event_scores = event_scores + self.gamma * av_score

        return is_event_scores, event_scores, av_score, visual_feature, audio_feature


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
                self.self_attention = MultiheadAttention(input_dim, num_heads=self.num_head, dropout=self.head_dropout)

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
                x = x.squeeze(-1).permute(0,2,1)

                x, x_weights = self.self_attention(x, x, x)
                x = x.permute(0, 2, 1).unsqueeze(-1)
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

    def forward(self, x, vis_token=None):
        vis_token = self.conv_adapter(vis_token.transpose(2, 1))
        vis_token_fc = self.fc(vis_token.squeeze(-1))
        vis_token = vis_token_fc.permute(0, 2, 1).unsqueeze(-1)
        modal_1 = x.squeeze(-1).permute(0, 2, 1)
        modal_2 = vis_token_fc
        modal_1 = modal_1.mean(dim=1, keepdim=True)
        modal_2 = modal_2.mean(dim=1, keepdim=True)
        multimodal_input = torch.cat((modal_1, modal_2), dim=-1)
        gating_logits = self.router(multimodal_input)
        gating_probs = F.softmax(gating_logits, dim=-1)
        expert_indices = torch.argmax(gating_probs, dim=-1)

        expert_outputs = []
        for expert in self.multimodal_experts + self.singlemodal_experts:
            expert_output = expert(x, vis_token)
            expert_outputs.append(expert_output)
        expert_outputs_tensor = torch.concat(expert_outputs, dim=-1)
        final_expert_output = (expert_outputs_tensor * gating_probs.unsqueeze(-2)).sum(dim=-1, keepdim=True)
        return final_expert_output, expert_indices

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




class MMIL_Net(nn.Module):
    def __init__(self, opt):
        super(MMIL_Net, self).__init__()
        self.opt = opt
        if opt.model_size == "large":
            model_dim = 1536
            swin_name = "swinv2_large_window12_192_22k"
        else:
            model_dim = 1024
            swin_name = "swinv2_base_window12_192_22k"
        if self.opt.is_cmbs:
            self.CMBS = CMBS(self.opt)
            # self.temporal_attn = TemporalAttention()
            self.d_model = 256
            if self.opt.is_temporal_att:
                self.temporal_attn = TemporalAttention(self.opt)
            else:
                self.v_fc = nn.Linear(model_dim, self.d_model)
                self.a_fc = nn.Linear(768, self.d_model)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
        else:
            self.mlp_class = nn.Linear(model_dim + 768, 512)  # swinv2-Large
            self.mlp_class_2 = nn.Linear(512, 29)
        self.swin = timm.create_model(swin_name, pretrained=True)
        # self.swin = timm.create_model('swinv2_base_window12_192_22k', pretrained=True)

        if opt.backbone_type == "esc-50":
            esc_config.dataset_path = "your processed ESC-50 folder"
            esc_config.dataset_type = "esc-50"
            esc_config.loss_type = "clip_ce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320
            esc_config.classes_num = 50
            esc_config.checkpoint_path = "../checkpoints/ESC-50/"
            esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
        elif opt.backbone_type == "audioset":  # go this part
            esc_config.dataset_path = "your processed audioset folder"
            esc_config.dataset_type = "audioset"
            esc_config.balanced_data = True
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320
            esc_config.classes_num = 527
            esc_config.checkpoint_path = "../checkpoints/AudioSet/"
            esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
        elif opt.backbone_type == "scv2":
            esc_config.dataset_path = "your processed SCV2 folder"
            esc_config.dataset_type = "scv2"
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 16000
            esc_config.hop_size = 160
            esc_config.classes_num = 35
            esc_config.checkpoint_path = "../checkpoints/SCV2/"
            esc_config.checkpoint = "HTSAT_SCV2_Saved_3.ckpt"
        else:
            raise NotImplementedError

        self.htsat = HTSAT_Swin_Transformer(
            spec_size=esc_config.htsat_spec_size,
            patch_size=esc_config.htsat_patch_size,
            in_chans=1,
            num_classes=esc_config.classes_num,
            window_size=esc_config.htsat_window_size,
            config=esc_config,
            depths=esc_config.htsat_depth,
            embed_dim=esc_config.htsat_dim,
            patch_stride=esc_config.htsat_stride,
            num_heads=esc_config.htsat_num_head
        )

        checkpoint_path = os.path.join(esc_config.checkpoint_path, esc_config.checkpoint)
        tmp = torch.load(checkpoint_path, map_location='cpu')
        tmp = {k[10:]: v for k, v in tmp['state_dict'].items()}
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
            conv_dim_tmp = (my_blk.input_resolution[0] * my_blk.input_resolution[1])
            conv_dim_tmp_a = (my_blk_a.input_resolution[0] * my_blk_a.input_resolution[1])
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
        #
        # self.adapter_token_downsampler = nn.ModuleList([
        #     nn.Linear(down_out_dim[i] // (self.opt.Adapter_downsample * 2),
        #               down_out_dim[i] // self.opt.Adapter_downsample, bias=False)
        #     for i in range(len(down_in_dim))])
        # self.adapter_token_downsampler.append(nn.Identity())
        ## <--------------

        if self.opt.is_audio_adapter_p1:
            self.audio_moe_adapter_blocks_p1 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i],
                              adapter_kind="bottleneck", dim_list=hidden_list_a, layer_idx=i,
                              reduction_factor=self.opt.Adapter_downsample,
                              opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate,
                              num_tk=opt.num_tokens, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
                              linear_in=hidden_list[i], linear_out=hidden_list_a[i]
                              )
                for i in range(len(hidden_list_a))])

            self.vis_moe_adapter_blocks_p1 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list[i],
                              output_dim=hidden_list[i], adapter_kind="bottleneck",
                              dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample,
                              opt=opt, use_bn=self.opt.is_bn, use_gate=True,
                              num_tk=opt.num_tokens, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                              linear_in=hidden_list_a[i], linear_out=hidden_list[i]
                              )
                for i in range(len(hidden_list))])

        if self.opt.is_audio_adapter_p2:
            self.audio_moe_adapter_blocks_p2 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck",
                              dim_list=hidden_list_a, layer_idx=i, reduction_factor=self.opt.Adapter_downsample,
                              opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate,
                              num_tk=opt.num_tokens, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
                              linear_in=hidden_list[i], linear_out=hidden_list_a[i]
                              )
                for i in range(len(hidden_list_a))])

            self.vis_moe_adapter_blocks_p2 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                              dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample,
                              opt=opt, use_bn=self.opt.is_bn, use_gate=True,
                              num_tk=opt.num_tokens, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                              linear_in=hidden_list_a[i], linear_out=hidden_list[i]
                              )
                for i in range(len(hidden_list))])

        # 帧数 T
        self.T = 10

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
        if self.opt.is_audio_adapter_p1:
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


    def forward_swin(self, audio, vis, frame_cap, wave_cap, mixup_lambda, rand_train_idx=12, stage='eval'):

        audio = audio[0]
        audio = audio.view(audio.size(0) * audio.size(1), -1)
        waveform = audio
        bs = vis.size(0)
        vis = rearrange(vis, 'b t c w h -> (b t) c w h')
        f_v = self.swin.patch_embed(vis)

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
        else:  # this part is typically used, and most easy one
            audio = self.htsat.reshape_wav2img(audio)
        frames_num = audio.shape[2]
        f_a = self.htsat.patch_embed(audio)
        if self.htsat.ape:
            f_a = f_a + self.htsat.absolute_pos_embed
        f_a = self.htsat.pos_drop(f_a)

        idx_layer = 0
        out_idx_layer = 0

        adapter_index_dict = {'audio': {'p1': [], 'p2': []}, 'video': {'p1': [], 'p2': []}}

        enrich_idx = 0

        draw_ori_f_v, draw_ori_f_a = f_v, f_a


        for layer_index, (my_blk, htsat_blk) in enumerate(zip(self.swin.layers, self.htsat.layers)):
            if len(my_blk.blocks) == len(htsat_blk.blocks):
                aud_blocks = htsat_blk.blocks
            else:
                aud_blocks = [None, None, htsat_blk.blocks[0], None, None, htsat_blk.blocks[1], None, None,
                              htsat_blk.blocks[2], None, None, htsat_blk.blocks[3], None, None, htsat_blk.blocks[4],
                              None, None, htsat_blk.blocks[5]]
                assert len(aud_blocks) == len(my_blk.blocks)

            block_first_layer = True

            for (blk, blk_a) in zip(my_blk.blocks, aud_blocks):
                if blk_a is not None:
                    # f_a shape:  torch.Size([10, 4096, 96]) BNC
                    # f_v shape:  torch.Size([10, 2304, 128])
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

                        if self.opt.is_audio_adapter_p1:
                            f_a_res, f_a_moe_adapter_index_p1 = self.audio_moe_adapter_blocks_p1[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1), f_v.permute(0, 2, 1).unsqueeze(-1))
                            f_v_res, f_v_moe_adapter_index_p1 = self.vis_moe_adapter_blocks_p1[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1), f_a.permute(0, 2, 1).unsqueeze(-1))

                            adapter_index_dict['audio']['p1'].append(f_a_moe_adapter_index_p1.squeeze().tolist())
                            adapter_index_dict['video']['p1'].append(f_v_moe_adapter_index_p1.squeeze().tolist())
                            f_v = f_v_ori + blk.drop_path1(blk.norm1(blk._attn(f_v_ori)))
                            f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)
                        f_a, _ = blk_a(f_a_ori)
                        if self.opt.is_audio_adapter_p1:
                            f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                        if self.opt.is_audio_adapter_p2:
                            f_a_res, f_a_moe_adapter_index_p2 = self.audio_moe_adapter_blocks_p2[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1), f_v.permute(0, 2, 1).unsqueeze(-1))
                            f_v_res, f_v_moe_adapter_index_p2 = self.vis_moe_adapter_blocks_p2[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1), f_a.permute(0, 2, 1).unsqueeze(-1))
                            adapter_index_dict['audio']['p2'].append(f_a_moe_adapter_index_p2.squeeze().tolist())
                            adapter_index_dict['video']['p2'].append(f_v_moe_adapter_index_p2.squeeze().tolist())

                        f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                        if self.opt.is_audio_adapter_p2:
                            f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                            f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                        idx_layer = idx_layer + 1

                else:
                    f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                    f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))

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
        f_v = f_v.mean(dim=1, keepdim=True)
        f_a = f_a.mean(dim=1, keepdim=True)
        ########## Temporal Attention ##########
        if self.opt.is_cmbs:
            if self.opt.is_temporal_att:
                f_v = f_v.view(bs, 10, -1)
                f_a = f_a.view(bs, 10, -1)
                visual_feature, audio_feature, audio_visual_gate = self.temporal_attn(f_v, f_a)
            else:
                f_v = f_v.view(10, bs, -1)
                f_a = f_a.view(10, bs, -1)
                visual_feature = self.v_fc(f_v)
                visual_feature = self.dropout(self.relu(visual_feature))
                audio_feature = self.a_fc(f_a)
                audio_feature = self.dropout(self.relu(audio_feature))
                
            draw_swin_f_v, draw_swin_f_a = visual_feature, audio_feature

            is_event_scores, event_scores, av_score, draw_final_f_v, draw_final_f_a  = self.CMBS(visual_feature, audio_feature)


            return is_event_scores, event_scores, av_score, adapter_index_dict, case_aux, draw_ori_f_v.mean(dim=1, keepdim=True).mean(dim=0).squeeze(), draw_ori_f_a.mean(dim=1, keepdim=True).mean(dim=0).squeeze(), draw_swin_f_v.mean(dim=0).squeeze(), draw_swin_f_a.mean(dim=0).squeeze(), draw_final_f_v.mean(dim=0).squeeze(), draw_final_f_a.mean(dim=0).squeeze()
        else:
            out_av = torch.cat((f_v, f_a), dim=-1)
            out_av = rearrange(out_av, 'b t p -> (b t) p')

            p_av = self.mlp_class(out_av)
            p_av = self.mlp_class_2(p_av)

            # due to BCEWithLogitsLoss
            p_av = F.softmax(p_av, dim=-1)
            return p_av

    def forward(self, audio, vis, frame_cap, wave_cap, mixup_lambda=None, rand_train_idx=12, stage='eval'):
        return self.forward_swin(audio, vis, frame_cap, wave_cap, mixup_lambda, rand_train_idx=12, stage='eval')

