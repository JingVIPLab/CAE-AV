import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18

from ipdb import set_trace
import timm
from einops import rearrange, repeat
import os

import loralib as lora
from transformers.activations import get_activation
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint

from htsat import HTSAT_Swin_Transformer
import esc_config as esc_config
from utils import do_mixup, get_mix_lambda, do_mixup_label


from torch.nn import init
import math


class VisualAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = opt.is_multimodal
        self.opt = opt
        self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)
        self.fc = nn.Linear(linear_in, linear_out)

        d_model = linear_out // 2
        self.fc_affine_audio_1 = nn.Linear(linear_out, linear_out)
        self.fc_affine_video_1 = nn.Linear(linear_out, linear_out)
        self.fc_affine_bottleneck = nn.Linear(linear_out, d_model)
        self.fc_affine_video_2 = nn.Linear(linear_out, d_model)
        self.fc_affine_audio_2 = nn.Linear(linear_out, d_model)
        self.fc_affine_v_s_att = nn.Linear(d_model, 1)
        self.fc_tanh = nn.Tanh()
        self.fc_softmax = nn.Softmax(dim=-1)
        self.fc_affine_v_c_att = nn.Linear(d_model, linear_out)

        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        if adapter_kind == "bottleneck" and self.is_multimodal:
            self.down_sample_size = input_dim // reduction_factor

            self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))
            self.gate_av = nn.Parameter(torch.zeros(1))

            ### <------

            self.activation = nn.ReLU(inplace=True)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

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

            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)

            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "basic":
            self.activation = nn.ReLU(inplace=True)
            self.conv = nn.Linear(input_dim, output_dim, bias=False)

            if use_bn:
                self.bn = nn.BatchNorm1d(output_dim)

        else:
            raise NotImplementedError

    def forward(self, x, vis_token=None):
        vis_token = self.conv_adapter(vis_token.transpose(2, 1))
        vis_token = self.fc(vis_token.squeeze(-1))
        vis_token = vis_token.permute(0, 2, 1).unsqueeze(-1)

        spatial_att_maps = None
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

            # ============================== Channel Attention ====================================
            audio = vis_token.mean(dim=2).squeeze(-1)  # [B*10, dim]
            audio_query_1 = F.relu(self.fc_affine_audio_1(audio)).unsqueeze(-2)
            video_query_1 = F.relu(
                self.fc_affine_video_1(x.squeeze(-1).permute(0, 2, 1)))  # [*, grid ** 2, width]
            audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)  # [*, width]
            audio_video_query = F.relu(self.fc_affine_bottleneck(audio_video_query_raw))
            channel_att_maps = self.fc_affine_v_c_att(audio_video_query).sigmoid().reshape(x.size(0), 1, -1)
            c_att_visual_feat = (x.squeeze(-1).permute(0, 2, 1) * (channel_att_maps + 1))  # [B*10, 36, 768]

            # ============================== Spatial Attention =====================================
            # channel attended visual feature: [batch * 10, 36, v_dim]
            c_att_visual_query = F.relu(self.fc_affine_video_2(c_att_visual_feat))
            audio_query_2 = F.relu(self.fc_affine_audio_2(audio)).unsqueeze(-2)
            audio_video_query_2 = c_att_visual_query * audio_query_2
            spatial_att_maps_tmp = self.fc_affine_v_s_att(audio_video_query_2)
            spatial_att_maps_sigmoid = spatial_att_maps_tmp.transpose(2, 1).sigmoid()
            spatial_att_maps_sigmoid = spatial_att_maps_sigmoid.transpose(2, 1)
            spatial_att_maps = self.fc_softmax(self.fc_tanh(spatial_att_maps_tmp).transpose(2, 1))
            c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat)

            alpha, beta = 0.3, 0.05
            x = x.squeeze(-1).permute(0, 2, 1) * (
                    alpha * channel_att_maps + beta * spatial_att_maps_sigmoid + 1 - alpha)
            x = x.permute(0, 2, 1).unsqueeze(-1)

            # =======================================================================================

            ### <----------
            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            ## <----

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "bottleneck":

            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
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

        return output, spatial_att_maps


class ExpertAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True, is_multimodal=True):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = is_multimodal
        self.opt = opt
        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        if adapter_kind == "bottleneck" and self.is_multimodal:
            self.down_sample_size = input_dim // reduction_factor
            self.my_tokens = nn.Parameter(torch.rand((self.opt.num_tokens, input_dim)))

            self.gate_av = nn.Parameter(torch.zeros(1))
            self.activation = nn.ReLU(inplace=True)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)
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

            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)

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
                 use_bn=True, use_gate=True, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
        super().__init__()
        self.opt = opt
        self.use_bn = use_bn
        self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)
        self.fc = nn.Linear(linear_in, linear_out)
        self.num_multimodal_experts = self.opt.num_multimodal_experts
        self.num_singlemodal_experts = self.opt.num_singlemodal_experts
        self.multimodal_experts = nn.ModuleList([
            ExpertAdapter(input_dim, output_dim, adapter_kind, dim_list,
                          layer_idx, reduction_factor, opt, use_bn,
                          use_gate, is_multimodal=True)
            for _ in range(self.num_multimodal_experts)
        ])
        self.singlemodal_experts = nn.ModuleList([
            ExpertAdapter(input_dim, output_dim, adapter_kind, dim_list,
                          layer_idx, reduction_factor, opt, use_bn,
                          use_gate, is_multimodal=False)
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


def batch_organize(out_match_posi, out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0

    return out_match, batch_labels


# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):
        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)  # 2 for hidden and cell states

    def forward(self, question):
        qst_vec = self.word2vec(question)  # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)  # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)  # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)  # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]

        return qst_feature


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




class AVQA_Fusion_Net(nn.Module):

    def __init__(self, opt):
        super(AVQA_Fusion_Net, self).__init__()

        self.opt = opt

        # for features
        self.fc_a1 = nn.Linear(768, 1536)
        self.fc_a2 = nn.Linear(1536, 1536)

        self.fc_a1_pure = nn.Linear(768, 1536)
        self.fc_a2_pure = nn.Linear(1536, 1536)

        self.fc_fusion = nn.Linear(1536 + 1536, 1536)

        self.linear11 = nn.Linear(1536, 1536)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(1536, 1536)

        self.linear21 = nn.Linear(1536, 1536)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(1536, 1536)
        self.norm1 = nn.LayerNorm(1536)
        self.norm2 = nn.LayerNorm(1536)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(1536)

        self.attn_a = nn.MultiheadAttention(1536, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(1536, 4, dropout=0.1)

        # question
        self.question_encoder = QstEncoder(93, 1536, 1536, 1, 1536)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc_ans = nn.Linear(1536, self.opt.avqa_fc_class) # 42

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl = nn.Linear(1536 + 1536, 1536)

        # combine
        self.fc1 = nn.Linear(1536 + 1536, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()

        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)

        if opt.backbone_type == "esc-50":
            esc_config.dataset_path = "your processed ESC-50 folder"
            esc_config.dataset_type = "esc-50"
            esc_config.loss_type = "clip_ce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320
            esc_config.classes_num = 50
            esc_config.checkpoint_path = "./../checkpoints/ESC-50/"
            esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
        elif opt.backbone_type == "audioset":
            esc_config.dataset_path = "your processed audioset folder"
            esc_config.dataset_type = "audioset"
            esc_config.balanced_data = True
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320
            esc_config.classes_num = 527
            esc_config.checkpoint_path = "./../checkpoints/AudioSet/"
            esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
        elif opt.backbone_type == "scv2":
            esc_config.dataset_path = "your processed SCV2 folder"
            esc_config.dataset_type = "scv2"
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 16000
            esc_config.hop_size = 160
            esc_config.classes_num = 35
            esc_config.checkpoint_path = "./../checkpoints/SCV2/"
            esc_config.checkpoint = "HTSAT_SCV2_Saved_2.ckpt"
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

        # self.nce_av = InfoNCELoss(margin=opt.tmp_av)
        # self.nce_tv = InfoNCELoss(margin=opt.tmp_tv)

        hidden_list, hidden_list_a = [], []
        down_in_dim, down_in_dim_a = [], []
        down_out_dim, down_out_dim_a = [], []
        conv_dim, conv_dim_a = [], []

        ### ------------> for swin and htsat
        for idx_layer, (my_blk, my_blk_a) in enumerate(zip(self.swin.layers, self.htsat.layers)):
            if self.opt.num_skip>1:
                if (idx_layer+1)%self.opt.num_skip ==0:
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
        ### <--------------

        self.audio_adapter_blocks_p1 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i],
                       adapter_kind="bottleneck", dim_list=hidden_list_a, layer_idx=i,
                       reduction_factor=self.opt.Adapter_downsample,
                       opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i],
                       conv_dim_out=conv_dim_a[i],
                       linear_in=hidden_list[i], linear_out=hidden_list_a[i])
            for i in range(len(hidden_list_a))])

        self.vis_adapter_blocks_p1 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                       dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                       use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                       linear_in=hidden_list_a[i], linear_out=hidden_list[i])
            for i in range(len(hidden_list))])

        self.audio_adapter_blocks_p2 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck",
                       dim_list=hidden_list_a, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                       use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i],
                       conv_dim_out=conv_dim_a[i],
                       linear_in=hidden_list[i], linear_out=hidden_list_a[i])
            for i in range(len(hidden_list))])

        self.vis_adapter_blocks_p2 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                       dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                       use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                       linear_in=hidden_list_a[i], linear_out=hidden_list[i])
            for i in range(len(hidden_list_a))])
        

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

    def forward(self, audio, visual_posi, visual_nega, question, frame_cap, wave_cap, mixup_lambda, stage='eval'):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
            input visual_nega shape: [B, T, C, H, W]
        '''

        bs, t, c, h, w = visual_posi.shape

        audio = audio.view(audio.size(0) * audio.size(1), -1)
        waveform = audio
        bs = visual_posi.size(0)

        visual_posi = rearrange(visual_posi, 'b t c w h -> (b t) c w h')
        f_v = self.swin.patch_embed(visual_posi)
        # f_v_neg = self.swin.patch_embed(visual_nega)

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
        multi_scale = []

        idx_block = 0
        adapter_index_dict = {'audio': {'p1': [], 'p2': []}, 'video': {'p1': [], 'p2': []}}

        enrich_idx = 0

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
                    if self.opt.num_skip > 1 and (((layer_index+1) % self.opt.num_skip)==0):
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
                            f_a_res, f_a_moe_adapter_index_p1 = self.audio_adapter_blocks_p1[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1), f_v.permute(0, 2, 1).unsqueeze(-1))
                            f_v_res, f_v_moe_adapter_index_p1  = self.vis_adapter_blocks_p1[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1), f_a.permute(0, 2, 1).unsqueeze(-1))

                            adapter_index_dict['audio']['p1'].append(f_a_moe_adapter_index_p1.squeeze().tolist())
                            adapter_index_dict['video']['p1'].append(f_v_moe_adapter_index_p1.squeeze().tolist())
                            f_v = f_v_ori + blk.drop_path1(blk.norm1(blk._attn(f_v_ori)))
                            f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)
                        f_a, _ = blk_a(f_a_ori)
                        if self.opt.is_audio_adapter_p1:
                            f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)


                        if self.opt.is_audio_adapter_p2:
                            f_a_res, f_a_moe_adapter_index_p2 = self.audio_adapter_blocks_p2[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1), f_v.permute(0, 2, 1).unsqueeze(-1))
                            f_v_res, f_v_moe_adapter_index_p2 = self.vis_adapter_blocks_p2[idx_layer](
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

            #####
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

        with torch.no_grad():

            visual_nega = rearrange(visual_nega, 'b t c h w -> (b t) c h w')
            visual_nega = self.swin.forward_features(visual_nega)

        ############## <----------

        visual_posi = rearrange(f_v, '(b t) (h w) c -> b t c h w', b=bs, t=t, h=6, w=6)
        visual_nega = rearrange(visual_nega, '(b t) (h w) c -> b t c h w', b=bs, t=t, h=6, w=6)

        f_a = f_a.mean(dim=1)
        # f_a = torch.bmm(f_a_spatial_att_maps, f_a).squeeze(dim=1)
        audio = rearrange(f_a, '(b t) c -> b t c', b=bs, t=t)
        ### <-----

        # visual_posi, f_a = self.temporal_attn(visual_posi, f_a)

        ## question features
        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        ## audio features  [2*B*T, 128]
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)
        audio_feat_pure = audio_feat
        B, T, C = audio_feat.size()  # [B, T, C]
        audio_feat = audio_feat.view(B * T, C)  # [B*T, C]

        ## visual posi [2*B*T, C, H, W]
        B, T, C, H, W = visual_posi.size()
        temp_visual = visual_posi.view(B * T, C, H, W)  # [B*T, C, H, W]
        v_feat = self.avgpool(temp_visual)  # [B*T, C, 1, 1]
        visual_feat_before_grounding_posi = v_feat.squeeze()  # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)  # [B, HxW, C]
        visual_feat_posi = nn.functional.normalize(v_feat, dim=2)  # [B, HxW, C]

        ## audio-visual grounding posi
        audio_feat_aa = audio_feat.unsqueeze(-1)  # [B*T, C, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)  # [B*T, C, 1]

        x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze()  # [B*T, HxW]

        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)  # [B*T, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat_posi)
        visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()  # [B*T, C]

        visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi), dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_posi = self.fc_gl(visual_feat_grd)  # [B*T, C]

        feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)  # [B*T, C*2], [B*T, 3072]

        feat = F.relu(self.fc1(feat))  # (3072, 512)
        feat = F.relu(self.fc2(feat))  # (512, 256)
        feat = F.relu(self.fc3(feat))  # (256, 128)
        out_match_posi = self.fc4(feat)  # (128, 2)

        ###############################################################################################
        # visual nega
        B, T, C, H, W = visual_nega.size()
        temp_visual = visual_nega.view(B * T, C, H, W)
        v_feat = self.avgpool(temp_visual)
        visual_feat_before_grounding_nega = v_feat.squeeze()  # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)  # [B, HxW, C]
        visual_feat_nega = nn.functional.normalize(v_feat, dim=2)

        ##### av grounding nega
        x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze()
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)  # [B*T, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat_nega)
        visual_feat_grd_after_grounding_nega = visual_feat_grd.squeeze()  # [B*T, C]

        visual_gl = torch.cat((visual_feat_before_grounding_nega, visual_feat_grd_after_grounding_nega), dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_nega = self.fc_gl(visual_feat_grd)  # [B*T, C]

        # combine a and v
        feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)  # [B*T, C*2], [B*T, 1024]

        feat = F.relu(self.fc1(feat))  # (1024, 512)
        feat = F.relu(self.fc2(feat))  # (512, 256)
        feat = F.relu(self.fc3(feat))  # (256, 128)
        out_match_nega = self.fc4(feat)  # (128, 2)

        ###############################################################################################

        # out_match=None
        # match_label=None

        B = xq.shape[1]
        visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 1536)  # [B, T, 512]
        visual_feat_grd = visual_feat_grd_be.permute(1, 0, 2)

        ## attention, question as query on visual_feat_grd
        visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[
            0].squeeze(0)
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = visual_feat_att + self.dropout2(src)
        visual_feat_att = self.norm1(visual_feat_att)

        # attention, question as query on audio
        audio_feat_be = audio_feat_pure.view(B, -1, 1536)
        audio_feat = audio_feat_be.permute(1, 0, 2)
        audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = audio_feat_att + self.dropout4(src)
        audio_feat_att = self.norm2(audio_feat_att)

        feat = torch.cat((audio_feat_att + audio_feat_be.mean(dim=-2).squeeze(),
                          visual_feat_att + visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

        ## fusion with question
        combined_feature = torch.mul(feat, qst_feature)
        combined_feature = self.tanh(combined_feature)
        out_qa = self.fc_ans(combined_feature)  # [batch_size, ans_vocab_size]

        return out_qa, out_match_posi, out_match_nega, adapter_index_dict, case_aux

