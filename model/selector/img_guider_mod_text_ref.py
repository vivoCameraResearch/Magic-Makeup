import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from typing import Dict, List, Optional, Set, Tuple, Union
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
import math
from diffusers.models.embeddings import FluxPosEmbed
import torch.utils.checkpoint as ckpt
import ipdb


def timestep_embedding(t: torch.Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, ln_bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6, bias=ln_bias)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x

class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

class CustomAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = query.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class ImgGuider(nn.Module):
    def __init__(
        self, dim: int, 
        attention_head_dim: int, 
        text_out_dim: int, 
        img_out_dim: int, 
        eps: float = 1e-6, 
        condition_lens: int = 64
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,  # inner dim = head dim * head (8)
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor2_0(),
            eps=eps,
            pre_only=False,
        )

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            dim_head=attention_head_dim,
            bias=True,
            processor=CustomAttnProcessor2_0(),
            eps=eps,
            pre_only=False,
        )

        self.attn3 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            dim_head=attention_head_dim,
            bias=True,
            processor=CustomAttnProcessor2_0(),
            eps=eps,
            pre_only=False,
        )
        
        self.norm_txt = AdaLayerNorm(dim)
        self.norm_img = AdaLayerNorm(dim)
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=(8, 28, 28))
        self.condition_lens = condition_lens

        self.img_norm_out = AdaLayerNorm(dim)
        self.img_proj_out = nn.Linear(dim, img_out_dim, bias=True)

        self.text_norm_out = AdaLayerNorm(dim)
        self.text_proj_out = nn.Linear(dim, text_out_dim, bias=True)

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def gen_image_rotary_emb(self, img_emb, text_emb) -> Tuple[torch.Tensor, torch.Tensor]:
        img_token_len = img_emb.shape[1]
        token_h = int(img_token_len ** 0.5)
        img_ids = torch.zeros(token_h, token_h, 3, device=img_emb.device, dtype=img_emb.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(token_h, device=img_emb.device, dtype=img_emb.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(token_h, device=img_emb.device, dtype=img_emb.dtype)[None, :]
        img_ids = img_ids.flatten(0, 1)

        txt_ids = torch.zeros(text_emb.shape[1], 3, device=text_emb.device, dtype=text_emb.dtype)

        token_ids = torch.cat([txt_ids, img_ids], dim=0)
        return self.pos_embed(token_ids)

    def _ckpt_attn1(self, norm_last_layer_emb, norm_text_emb, image_rotary_emb):
        # 用一个小函数封装 attn1 前向，便于 checkpoint
        return self.attn1(
            hidden_states=norm_last_layer_emb,
            encoder_hidden_states=norm_text_emb,
            image_rotary_emb=image_rotary_emb,
        )

    def _ckpt_attn2(self, init_q, enc_states):
        return self.attn2(
            hidden_states=init_q,
            encoder_hidden_states=enc_states,
        )

    def _ckpt_attn3(self, init_q, text_emb):
        return self.attn3(
            hidden_states=init_q,
            encoder_hidden_states=text_emb,
        )

    def forward(
        self,
        img_emb: torch.Tensor,
        text_emb: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        # img_emb 是 list/tuple，拼接前先保留 dtype/device
        bsz = img_emb[0].shape[0] // 2
        temb = timestep_embedding(temb, 1152)

        img_emb = torch.cat(img_emb, dim=1)

        norm_img_emb = self.norm_img(img_emb, timestep_embedding=temb)
        norm_text_emb = self.norm_txt(text_emb, timestep_embedding=temb)

        last_layer_emb = img_emb[:, -729:, :]

        image_rotary_emb = self.gen_image_rotary_emb(last_layer_emb, text_emb)

        norm_third_last_layer_emb, norm_second_last_layer_emb, norm_last_layer_emb = norm_img_emb.chunk(3, dim=1)
        # attn1: 需要返回 (img_update, text_update)
        if self.training and self.gradient_checkpointing:
            img_update, text_update = ckpt.checkpoint(
                self._ckpt_attn1,
                norm_last_layer_emb,
                norm_text_emb,
                image_rotary_emb,
                use_reentrant=False,
            )
        else:
            img_update, text_update = self._ckpt_attn1(
                norm_last_layer_emb, norm_text_emb, image_rotary_emb
            )

        last_layer_emb = last_layer_emb + img_update
        text_emb = text_update + text_emb

        # 选 top-k 作为 init_q
        sim = last_layer_emb @ text_emb.transpose(1, 2)  # (b, m, n)
        # 可视化sim
        _, top_idx = torch.topk(sim.max(dim=2).values, k=self.condition_lens, dim=1, largest=True, sorted=True)  # (b,k)
        top_idx = top_idx.unsqueeze(-1).expand(-1, -1, last_layer_emb.size(-1))  # [b, k, c]
        init_q = torch.gather(last_layer_emb, dim=1, index=top_idx)  # [b, k, c]

        enc_states = torch.cat([norm_third_last_layer_emb, norm_second_last_layer_emb, last_layer_emb], dim=1)

        # attn2: 只返回 img_update
        if self.training and self.gradient_checkpointing:
            img_update = ckpt.checkpoint(
                self._ckpt_attn2,
                init_q,
                enc_states,
                use_reentrant=False,
            )
        else:
            img_update = self._ckpt_attn2(init_q, enc_states)

        init_q = init_q + img_update

        # attn3
        if self.training and self.gradient_checkpointing:
            img_update = ckpt.checkpoint(
                self._ckpt_attn3,
                init_q,
                text_emb,
                use_reentrant=False,
            )
        else:
            img_update = self._ckpt_attn3(init_q, text_emb)

        img_fea_out = init_q + img_update

        img_fea_out = self.img_norm_out(img_fea_out, temb)
        img_fea_out = self.img_proj_out(img_fea_out)

        text_emb = self.text_norm_out(text_emb, temb)
        text_emb = self.text_proj_out(text_emb)

        concept1_text_emb = []
        concept2_text_emb = []
        img_emb1 = []
        img_emb2 = []
        for i in range(bsz):
            concept1_text_emb.append(text_emb[i: i+1])
            concept2_text_emb.append(text_emb[i+bsz: i+bsz+1])
            img_emb1.append(img_fea_out[i: i+1])
            img_emb2.append(img_fea_out[i+bsz: i+bsz+1])
        concept1_text_emb = torch.cat(concept1_text_emb, dim=0)
        concept2_text_emb = torch.cat(concept2_text_emb, dim=0)
        img_emb1 = torch.cat(img_emb1, dim=0)
        img_emb2 = torch.cat(img_emb2, dim=0)

        if img_fea_out.dtype == torch.float16:
            img_fea_out = img_fea_out.clip(-65504, 65504)
        if text_emb.dtype == torch.float16:
            text_emb = text_emb.clip(-65504, 65504)

        return img_emb1, img_emb2, concept1_text_emb.max(1).values, concept2_text_emb.max(1).values

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class ImgGuiderCrossAttn(nn.Module):
    def __init__(
        self, 
        dim: int, 
        attention_head_dim: int, 
        vit_dim: int, 
        ff_mult: int,
        eps: float = 1e-6,
        use_gradient_checkpointing: bool = False,  # 新增开关
    ):
        super().__init__()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=vit_dim,
            dim_head=attention_head_dim,
            bias=True,
            processor=CustomAttnProcessor2_0(),
            eps=eps,
            pre_only=False,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim, bias=False),
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing

    def _attn_block(self, img_emb: torch.Tensor, guider_emb: torch.Tensor):
        # 仅包含 attention + 残差
        img_update = self.attn(
            hidden_states=img_emb,
            encoder_hidden_states=guider_emb,
        )
        return img_update + img_emb

    def _ff_block(self, img_emb: torch.Tensor):
        # 仅包含 FFN + 残差
        return self.ff(img_emb) + img_emb

    def forward(
        self,
        img_emb: torch.Tensor,
        guider_emb: torch.Tensor,
    ) -> torch.Tensor:

        # 定义包装函数，满足 checkpoint 对函数签名的要求（只接收 Tensor）
        def attn_ckpt(img_emb_, guider_emb_):
            return self._attn_block(img_emb_, guider_emb_)

        def ff_ckpt(img_emb_):
            return self._ff_block(img_emb_)

        if self.training and self.use_gradient_checkpointing:
            # 对注意力块进行 checkpoint
            img_emb = checkpoint.checkpoint(attn_ckpt, img_emb, guider_emb)
            # 对 FF 块进行 checkpoint
            img_emb = checkpoint.checkpoint(ff_ckpt, img_emb)
        else:
            img_emb = self._attn_block(img_emb, guider_emb)
            img_emb = self._ff_block(img_emb)

        return img_emb

        

if __name__ == '__main__':
    model = ImgGuider(dim=1152, attention_head_dim=64, text_out_dim=3072, img_out_dim=3072, condition_lens=10)

    batch = 4

    img_emb = tuple(3*[torch.randn((batch*2,729,1152))])
    # vae_img_emb = torch.randn((batch*2,4096,64))
    text_emb = torch.randn((batch*2,64,1152))
    temb = torch.randn((batch*2,))
    # image_rotary_emb = torch.randn((729*2+64,3))
    ref1_concept_idx = batch*[1]  # 两个reference的提示词，在句子中的长度
    ref2_concept_idx = batch*[1]  # 两个reference的提示词，在句子中的长度
    img_fea_out, img_fea2_out, text1_update, text2_update = model(img_emb, text_emb, temb)
    print(img_fea_out.shape)
    print(img_fea2_out.shape)
    print(text1_update.shape)
    print(text2_update.shape)

    model2 = ImgGuiderCrossAttn(dim=3072, attention_head_dim=64, vit_dim=1152, ff_mult=2)
    img_emb = torch.randn((batch,4096,3072))
    guider_emb = torch.randn((batch,100,1152))
    img_emb = model2(img_emb, guider_emb)
    print(img_emb.shape)