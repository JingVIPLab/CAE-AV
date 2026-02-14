import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nets.grouping import ModalityTrans
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
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint
from .models import EncoderLayer, DecoderLayer, Decoder
from .models import Encoder as CMBS_Encoder

from .htsat import HTSAT_Swin_Transformer
import nets.esc_config as esc_config
from .utils import do_mixup, get_mix_lambda, do_mixup_label

from torch.nn import init
import math


def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ExpertAdapter(nn.Module):
	"""Conventional Adapter layer, in which the weights of up and down sampler modules
	are parameters and are optimized."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, opt=None, is_multimodal=True):
		super().__init__()
		self.opt = opt
		self.adapter_kind = adapter_kind
		self.reduction_factor = self.opt.Adapter_downsample
		self.use_bn = self.opt.is_bn
		self.use_gate=self.opt.is_gate
		self.is_multimodal = is_multimodal
		self.num_tk = self.opt.num_tokens

		if self.use_gate:
			self.gate = nn.Parameter(torch.zeros(1))
		else:
			self.gate = None

		# bottleneck, multi_modal_adapter
		if adapter_kind == "bottleneck" and self.is_multimodal:
			self.down_sample_size = input_dim // self.reduction_factor
			self.my_tokens = nn.Parameter(torch.rand((self.num_tk, input_dim)))

			self.gate_av = nn.Parameter(torch.zeros(1))
			self.activation = nn.ReLU(inplace=True)
			
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, 
										groups=self.opt.num_conv_group, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, 
							   			groups=self.opt.num_conv_group, bias=False)

			if self.use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
		
		# bottleneck, single_modal_adapter
		elif adapter_kind == "bottleneck":
			self.down_sample_size = input_dim // self.reduction_factor
			self.gate_av = nn.Parameter(torch.zeros(1))
			self.activation = nn.ReLU(inplace=True)

			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, 
										groups=self.opt.num_conv_group, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, 
							   			groups=self.opt.num_conv_group, bias=False)

			if self.use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
		
		# error
		else:
			raise NotImplementedError

	def forward(self, x, vis_token=None):
		if self.adapter_kind == "bottleneck" and self.is_multimodal:
			rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
			att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))			
			att_v2tk = F.softmax(att_v2tk, dim=-1)			
			rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0, 2, 1))			
			rep_token = rep_token + rep_token_res

			# cross-modal attention
			att_tk2x = torch.bmm(x.squeeze(-1).permute(0, 2, 1), rep_token.permute(0, 2, 1))
			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(att_tk2x, rep_token).permute(0, 2, 1).unsqueeze(-1)

			x = x + self.gate_av * x_res.contiguous()
			
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
			# self attention
			x_squ = x.squeeze(-1)
			att_tk2x = torch.bmm(x_squ.permute(0, 2, 1), x_squ)
			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(x_squ, att_tk2x).unsqueeze(-1)
			
			x = x + self.gate_av * x_res.contiguous()

			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

			z = self.down_sampler(x)

			if self.use_bn:
				z = self.bn1(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
		
		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)
		
		if self.gate is not None:
			output = self.gate * output
		
		return output


class MoEAdapter(nn.Module):
	def __init__(self, input_dim, output_dim, adapter_kind, dim_list, layer_idx, opt=None, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
		super().__init__()
		self.opt = opt
		self.num_multimodal_experts = self.opt.num_multimodal_experts
		self.num_singlemodal_experts = self.opt.num_singlemodal_experts
		
		self.multimodal_experts = nn.ModuleList([
			ExpertAdapter(input_dim, output_dim, adapter_kind, dim_list,
						  layer_idx, opt, is_multimodal=True)
			for _ in range(self.num_multimodal_experts)
		])
		
		self.singlemodal_experts = nn.ModuleList([
			ExpertAdapter(input_dim, output_dim, adapter_kind, dim_list,
						  layer_idx, opt, is_multimodal=False)
			for _ in range(self.num_singlemodal_experts)
		])

		self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)
		self.fc = nn.Linear(linear_in, linear_out)
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

		expert_outputs = []
		for expert in self.multimodal_experts:
			expert_output = expert(x, vis_token)
			expert_outputs.append(expert_output)
		for expert in self.singlemodal_experts:
			expert_output = expert(x, vis_token)
			expert_outputs.append(expert_output)
		expert_outputs_tensor = torch.concat(expert_outputs, dim=-1)

		multimodal_input = torch.cat((modal_1, modal_2), dim=-1)
		gating_logits = self.router(multimodal_input)
		gating_probs = F.softmax(gating_logits, dim=-1)
		final_expert_output = (expert_outputs_tensor * gating_probs.unsqueeze(-2)).sum(dim=-1, keepdim=True)

		if self.opt.use_load_balacing_loss==1:
			load_balancing_loss = self.compute_load_balancing_loss(gating_probs)
		else:
			load_balancing_loss = 0.

		return final_expert_output, load_balancing_loss

	def compute_load_balancing_loss(self, gating_probs):
		expert_probs_mean = torch.mean(gating_probs, dim=0)
		uniform_distribution = torch.full_like(expert_probs_mean, 1.0 / expert_probs_mean.size(0))
		load_balancing_loss = F.kl_div(expert_probs_mean.log(), uniform_distribution, reduction='batchmean')
		return load_balancing_loss


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




class MGN_Net(nn.Module):

    def __init__(self, args):
        super(MGN_Net, self).__init__()

        opt = args
        self.opt = opt
        self.fc_a =  nn.Linear(768, args.dim)
        self.fc_v = nn.Linear(1536, args.dim)
        self.fc_st = nn.Linear(512, args.dim)
        self.fc_fusion = nn.Linear(args.dim * 2, args.dim)

        # hard or soft assignment
        self.unimodal_assgin = args.unimodal_assign
        self.crossmodal_assgin = args.crossmodal_assign

        unimodal_hard_assignment = True if args.unimodal_assign == 'hard' else False
        crossmodal_hard_assignment = True if args.crossmodal_assign == 'hard' else False

        # learnable tokens
        self.audio_token = nn.Parameter(torch.zeros(25, args.dim))
        self.visual_token = nn.Parameter(torch.zeros(25, args.dim))

        # class-aware uni-modal grouping
        self.audio_cug = ModalityTrans(
                            args.dim,
                            depth=args.depth_aud,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=True
                        )

        self.visual_cug = ModalityTrans(
                            args.dim,
                            depth=args.depth_vis,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=False
                        )

        # modality cross-modal grouping
        self.av_mcg = ModalityTrans(
                            args.dim,
                            depth=args.depth_av,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=crossmodal_hard_assignment,
                            use_han=False                        
                        )

        # prediction
        self.fc_prob = nn.Linear(args.dim, 1)
        self.fc_prob_a = nn.Linear(args.dim, 1)
        self.fc_prob_v = nn.Linear(args.dim, 1)

        self.fc_cls = nn.Linear(args.dim, 25)

        self.apply(self._init_weights)

        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
        self.checkpoint_path = opt.checkpoint_path
        if opt.backbone_type == "esc-50":
            esc_config.dataset_path = "your processed ESC-50 folder"
            esc_config.dataset_type = "esc-50"
            esc_config.loss_type = "clip_ce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320 
            esc_config.classes_num = 50
            esc_config.checkpoint_path =  self.checkpoint_path + "/ESC-50/"
            esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
        elif opt.backbone_type == "audioset":
            esc_config.dataset_path = "your processed audioset folder"
            esc_config.dataset_type = "audioset"
            esc_config.balanced_data = True
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320 
            esc_config.classes_num = 527
            esc_config.checkpoint_path = self.checkpoint_path + "/AudioSet/"
            esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
        elif opt.backbone_type == "scv2":
            esc_config.dataset_path = "your processed SCV2 folder"
            esc_config.dataset_type = "scv2"
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 16000
            esc_config.hop_size = 160
            esc_config.classes_num = 35
            esc_config.checkpoint_path = self.checkpoint_path + "/SCV2/"
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


        if self.opt.is_audio_adapter_p1:
            self.audio_adapter_blocks_p1 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], 
                            adapter_kind="bottleneck",dim_list=hidden_list_a, layer_idx=i, opt=opt,  
							conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
                            linear_in=hidden_list[i], linear_out=hidden_list_a[i]       
                            )
                for i in range(len(hidden_list_a))])

            self.vis_adapter_blocks_p1 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], 
                            adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i, opt=opt,  
							conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                            linear_in=hidden_list_a[i], linear_out=hidden_list[i]       
                            )
                for i in range(len(hidden_list))])

        if self.opt.is_audio_adapter_p2:
            self.audio_adapter_blocks_p2 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], 
                            adapter_kind="bottleneck",dim_list=hidden_list_a, layer_idx=i, opt=opt,  
							conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
                            linear_in=hidden_list[i], linear_out=hidden_list_a[i]       
                            )
                for i in range(len(hidden_list_a))])

            self.vis_adapter_blocks_p2 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], 
                            adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i, opt=opt,  
							conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
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

            
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, audio, visual, visual_st, frame_cap, wave_cap, mixup_lambda=None):
        b, t, d = visual_st.size()
        
        audio = audio.view(audio.size(0)*audio.size(1), -1)
        waveform = audio
        bs = visual.size(0)
        vis = rearrange(visual, 'b t c w h -> (b t) c w h')
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
        else: # this part is typically used, and most easy one
            audio = self.htsat.reshape_wav2img(audio)
        frames_num = audio.shape[2]
        f_a = self.htsat.patch_embed(audio)
        if self.htsat.ape:
            f_a = f_a + self.htsat.absolute_pos_embed
        f_a = self.htsat.pos_drop(f_a)
        
        idx_layer = 0
        out_idx_layer = 0
        total_load_balancing_loss = 0

        enrich_idx = 0

        for _, (my_blk, htsat_blk) in enumerate(zip(self.swin.layers, self.htsat.layers)) :

            if len(my_blk.blocks) == len(htsat_blk.blocks):
                aud_blocks = htsat_blk.blocks
            else:
                aud_blocks = [None, None, htsat_blk.blocks[0], None, None, htsat_blk.blocks[1], None, None, htsat_blk.blocks[2], None, None, htsat_blk.blocks[3], None, None, htsat_blk.blocks[4], None, None, htsat_blk.blocks[5]]
                assert len(aud_blocks) == len(my_blk.blocks)
                
            block_first_layer = True

            for (blk, blk_a) in zip(my_blk.blocks, aud_blocks):
                if blk_a is not None:

                    f_v_ori, f_a_ori = f_v, f_a
                    if block_first_layer:
                        # ====> 换成 CASTE（逐层一次，逐帧门控 + TopK 注入）
                        block_first_layer = False
                        v_res = self.vis_caste_blocks_p1[enrich_idx](f_v, f_a)
                        a_res = self.audio_caste_blocks_p1[enrich_idx](f_a, f_v)
                        f_v = f_v + v_res
                        f_a = f_a + a_res

                        enrich_idx = enrich_idx + 1

                        
                    f_a_res, load_balancing_loss_a = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
                    f_v_res, load_balancing_loss_v = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))
                    total_load_balancing_loss += load_balancing_loss_a + load_balancing_loss_v

                    f_v = f_v_ori + blk.drop_path1(blk.norm1(blk._attn(f_v_ori)))
                    f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)


                    f_a_ori, _ = blk_a(f_a_ori)
                    f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
            
                    f_a_res, load_balancing_loss_a = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
                    f_v_res, load_balancing_loss_v = self.vis_adapter_blocks_p2[idx_layer]( f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))
                    total_load_balancing_loss += load_balancing_loss_a + load_balancing_loss_v

                    f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                    f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

                    f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
                    
                    idx_layer = idx_layer +1
                    
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
        
        # ly: Processing similar to spatial attention mechanisms
        f_v = f_v.mean(dim=1, keepdim=True).permute(1, 0, 2) # [B, 10, 1536]
        f_a = f_a.mean(dim=1, keepdim=True).permute(1, 0, 2) # [B, 10, 768]
        
        x1_0 = self.fc_a(f_a)

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(f_v)
        vid_st = self.fc_st(visual_st)

        x2_0 = torch.cat((vid_s, vid_st), dim=-1)
        x2_0 = self.fc_fusion(x2_0)

        # visual uni-modal grouping
        x2, attn_visual_dict, _ = self.visual_cug(x2_0, self.visual_token, return_attn=True)

        # audio uni-modal grouping
        x1, attn_audio_dict, _ = self.audio_cug(x1_0, self.audio_token, x2_0, return_attn=True)

        # modality-aware cross-modal grouping
        x, _, _ = self.av_mcg(x1, x2, return_attn=True)

        
        # prediction
        av_prob = torch.sigmoid(self.fc_prob(x))                                # [B, 25, 1]
        global_prob = av_prob.sum(dim=-1)                                       # [B, 25]

        # cls token prediction
        aud_cls_prob = self.fc_cls(self.audio_token)                            # [25, 25]
        vis_cls_prob = self.fc_cls(self.visual_token)                           # [25, 25]

        # attentions
        attn_audio = attn_audio_dict[self.unimodal_assgin].squeeze(1)                    # [25, 10]
        attn_visual = attn_visual_dict[self.unimodal_assgin].squeeze(1)                  # [25, 10]

        # audio prediction
        a_prob = torch.sigmoid(self.fc_prob_a(x1))                                # [B, 25, 1]
        a_frame_prob = (a_prob * attn_audio).permute(0, 2, 1)                     # [B, 10, 25]
        a_prob = a_prob.sum(dim=-1)                                               # [B, 25]

        # visual prediction
        v_prob = torch.sigmoid(self.fc_prob_v(x2))                                # [B, 25, 1]
        v_frame_prob = (v_prob * attn_visual).permute(0, 2, 1)                    # [B, 10, 25]
        v_prob = v_prob.sum(dim=-1)                                               # [B, 25]

        # print("aud_cls_prob: ", aud_cls_prob)
        # print("vis_cls_prob: ", vis_cls_prob)
        # print("global_prob: ", global_prob)
        # print("a_prob: ", a_prob)
        # print("v_prob: ", v_prob)
        # print("a_frame_prob: ", a_frame_prob)
        # print("v_frame_prob: ", v_frame_prob)

        return aud_cls_prob, vis_cls_prob, global_prob, a_prob, v_prob, a_frame_prob, v_frame_prob, total_load_balancing_loss, case_aux

