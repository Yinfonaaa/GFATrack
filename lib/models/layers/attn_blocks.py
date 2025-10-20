import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    消除用于减少计算和消除噪音的潜在背景候选项。
    Args:
       
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
                            保持搜索区域标记（候选）的比率
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights
                                   用来积累注意力权重的模板掩码

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
                                   保留的搜索区域标记的索引
        removed_index (torch.Tensor): indices of removed search region tokens
                                      被删除的搜索区域标记的索引
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape  #torch.Size([32, 12, 320, 320])

    lens_keep = math.ceil(keep_ratio * lens_s)  #向上取整
    if lens_keep == lens_s:
        return tokens, global_index, None,None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue 由于速度问题，使用sort而不是topk。
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
#   x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)  #从原tensor中获取指定dim和指定index的数据
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index,topk_attn


class CEBlock(nn.Module):  #在Block框架的中间加了一个CE

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  #在mlp模块中第一个全连接层节点个数
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search #1.0

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None,kongjian=False):#global_index_template tempalte特征向量长度, global_index_search search特征向量长度
        lens_t = global_index_template.shape[1]
        lens_s=global_index_search.shape[1]
        x_attn, attn = self.attn(self.norm1(x), mask, True,kongjian,lens_t,lens_s)#通过多头，x_attn为注意力后的参数，attn为注意力权重（k*q）#先计算，然后得出attn再筛除
        x = x + self.drop_path(x_attn)#残差连接
        lens_t = global_index_template.shape[1]  #64
        attn_t=None
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search,attn_t = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn_t


class Block(nn.Module):  #对应vit 中的Encoder Block

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
