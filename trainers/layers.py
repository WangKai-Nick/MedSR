import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Standard deterministic cross-attention used by the PVL adapter.
    """

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, context):
        B, Tq, C = query.shape
        _, Tk, _ = context.shape

        Q = self.q_proj(query)      # [B, Tq, C]
        K = self.k_proj(context)    # [B, Tk, C]
        V = self.v_proj(context)    # [B, Tk, C]

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(C)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, V)
        fused = self.out_proj(out)       
        return fused

class TwoWayTransformerLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cross_attn_img_to_txt = CrossAttention(embed_dim)
        self.cross_attn_txt_to_img = CrossAttention(embed_dim)

    def forward(self, img_tokens, txt_tokens):
        img_tokens = self.cross_attn_img_to_txt(img_tokens, txt_tokens)
        txt_tokens = self.cross_attn_txt_to_img(txt_tokens, img_tokens)
        return img_tokens, txt_tokens

class PVL_Adapter(nn.Module):
    def __init__(self,
                 in_channels_vis: int,
                 in_channels_txt: int,
                 adapter_channels: int):
        
        super().__init__()

        # Down projection
        self.proj_vis_down = nn.Sequential(nn.Linear(in_channels_vis, adapter_channels, bias=False))
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)

        # Up projection
        self.proj_vis_up = nn.Linear(adapter_channels, in_channels_vis, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)

        # Cross-modal interaction
        self.two_way = TwoWayTransformerLayer(adapter_channels)

    def forward(self, vis, text):

        v = self.proj_vis_down(vis)
        t = self.proj_txt_down(text)

        v_fused, t_fused = self.two_way(v, t)

        vis_out = self.proj_vis_up(v_fused)
        txt_out = self.proj_txt_up(t_fused)

        return vis_out, txt_out
