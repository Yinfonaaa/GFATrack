import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):  #多头注意力模块
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,  #dim:输入token的dim
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  #使用全连接层有助于并行化
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  #->Wo
        self.proj_drop = nn.Dropout(proj_drop)

        #作者在vit的基础上改进的代码
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False,kongjian=False,lens_t=None,lens_s=None):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        #[batch_size, num_patches+1, total_embed_dim] ；batch_size为训练时这一批图片传入的数目，num_patches为切割后的图片面片数目，+1为cls token(本文未加)，total_embed_dim为768
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
      

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if kongjian==True:
            bool_w = mask_drop_outS(attn.flatten(0,1).detach().clone(), lens_t,lens_s,0.1)#空间抑制
            attn[bool_w.view(-1, self.num_heads, attn.shape[-2], attn.shape[-1])] = -float('inf')

        #作者改的------
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)
        #----------------

        attn = attn.softmax(dim=-1)  #对每一行数据进行softmax处理
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  #针对每一个v进行加权求和操作
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x



#作者改进部分
class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            #相对位置索引
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def mask_drop_outS(attn_weight, index_t,index_s,P):

    len_x=index_s
    len_z=index_t
    prob_w = torch.zeros_like(attn_weight)
    iden_w = torch.ones_like(attn_weight)
    N, L, L = attn_weight.shape
   

    # sample queries; based on the softmax attention matrix
    attn_weight_softmax = attn_weight.softmax(dim=-1)#n,320,320
    # 先获取互相关
    cross_z = attn_weight_softmax[:, -len_x:,0:len_z]#互相关
    cross_zmax=cross_z.max(dim=-1).values + 1e-6
   
    # for max
    att_query_max = cross_zmax # bs, L#取最大值

    # sample specific dropout positions for each query
    sample_num = int(P * (int(L//2)-1)) * L #p=0.1
    #两个自相关
    selfx_att = attn_weight_softmax[:, -len_x:, -len_x:]  # bs, L//2, L//2 #自相关
    selfx_att = selfx_att + torch.relu(-selfx_att.min(dim=-1).values).unsqueeze(-1) + 1e-6    # bs, L//2, L//2
    for i in range(N):
        selfx_att[i].fill_diagonal_(0)#对角线设0 对角线是自己和自己，没有意义
    spatial_att_all = selfx_att # bs, L, L//2
    spatial_att_all = spatial_att_all / spatial_att_all.sum(dim=-1).unsqueeze(-1) # bs, L, L//2#算平均比重
    att_final = spatial_att_all * att_query_max.unsqueeze(-1) # bs, L, L//2             att_query表示与别人之间最大的
   #他对两方都


    prob_flag = torch.zeros_like(att_final).view(N, -1)#bs,L*L//2
    sam_indices = torch.multinomial(att_final.view(N, -1), sample_num)#对每行进行M次抽样概率高的优先被采样
    prob_flag.scatter_(dim=1, index=sam_indices, value=1)#抽中的进入index
    
    prob_flag = prob_flag.view(N, len_x, len_x)
    prob_w[:, -len_x:, -len_x:] = prob_flag
    return (prob_w == 1) * (iden_w == 1) # 1 is dropout 