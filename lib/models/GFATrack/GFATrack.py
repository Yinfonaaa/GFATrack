
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.gfatrack.vit import vit_base_patch16_224
from lib.models.gfatrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.layers.MaeDecoder import build_maedecoder

class gfatrack(nn.Module):


    def __init__(self, selftransformer, transformer, finaltransformer, maedecoder, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.  初始化模型
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
                      如果要使用辅助解码损耗（每个解码器层的损耗），则为 True。
        """
        super().__init__()
        self.selfbackbone = selftransformer
        self.backbone = transformer
        self.finalbackbone = finaltransformer
        self.maedecoder=maedecoder
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)  #通过_get_clones()函数进行简单的子层堆叠定义self.layers=_get_clones(encoder_layer,num_layers)

   

    def ori_forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        pre_x, x, ori_aux_dict,none = self.backbone(z=template, x=search,sta = True,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        ori_x = pre_x
    
        return ori_x, ori_aux_dict


    #更新模板与模板标注框的框架，输入为模板与标注框的模板部分（或许可以是更新模板）
    def self_forward(self, template_gt:torch.Tensor,  
                template: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        self.template = template   

        self.template = torch.nn.functional.interpolate(template, size=(256, 256), mode='bilinear', align_corners=True) #(bt,3,256,256)


        pre_x, x, aux_dict ,index= self.selfbackbone( z=template_gt,x=self.template,sta = True,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )
        
        mae_x=pre_x
        B = pre_x.shape[0]
        pre_x = pre_x.view([B,16,-1,768])
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        pre_x = maxpool(pre_x.permute(0, 3, 1, 2))
        pre_x = pre_x.permute(0, 2, 3, 1)
        self_x = pre_x.reshape([B,-1,768])
        # print("更新模板与模板标注框的框架已运行完成，准备返回值self_out的大小：------",self_x.shape)
        

        return self_x,aux_dict,index,self.template,mae_x
                                    

    #综合两个框架的总框架，输入为两个子框架输出的特征
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_gt:torch.Tensor,
                target_in_search=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                use_mae=False   
                ):

        # self_x =self.self_forward(template_gt,template,ce_template_mask=None, ce_keep_rate=None,return_last_attn=False,)
        # ori_x = self.ori_forward(template,search,ce_template_mask=None,ce_keep_rate=None,return_last_attn=False,)

        # self_x  =self.self_forward(template_gt,template,ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,return_last_attn=return_last_attn,)
        # self_x  =self.self_forward(template_gt,template[1],ce_template_mask= None, ce_keep_rate=ce_keep_rate,return_last_attn=return_last_attn,)
        self_x ,template_dict,template_index,Dtemplate,mae_x=self.self_forward(template_gt,template[1],ce_template_mask= None, ce_keep_rate=ce_keep_rate,return_last_attn=return_last_attn,)
        ori_x, ori_aux_dict = self.ori_forward(template[0],search,ce_template_mask=ce_template_mask,ce_keep_rate=ce_keep_rate,return_last_attn=return_last_attn,)

        # print("开始进入综合框架：------")
        pre_x, x, final_aux_dict ,global_index_s= self.finalbackbone(z=self_x, x=ori_x,sta = False,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )
                                    



        # 合并字典
        a={'removed_indexes_t':template_dict['removed_indexes_s']}
        del template_dict['removed_indexes_s']
        del template_dict['keep_indexs']
        del template_dict['attn']
        template_dict.update(a)
        aux_dict = {}

        if use_mae:
           # mae_x=torch.cat([ori_x,pre_x],dim=2)
            mae_dict=self.maedecoder(pre_x,search,global_index_s)
            for k, v in mae_dict.items():
                aux_dict[k] = v#把值存入
            tmae_dict=self.maedecoder(mae_x,Dtemplate,template_index)
            for k, v in tmae_dict.items():
                        aux_dict['t'+k] = v#把值存入
        for d in [ori_aux_dict, final_aux_dict]:
            for k, v in d.items():
        
                if k in aux_dict:
                    aux_dict[k] += v # 列表相加
                else:
                    aux_dict[k] = v

        # print("看看aux_dict----：",aux_dict)


        # Forward head
        feat_last = x
        if isinstance(x, list):   
            feat_last = x[-1]
        # print("即将进入forward_head函数，feat_last的大小：------",feat_last.shape)
        out = self.forward_head(feat_last, None)

        # print("forward_head函数已结束，out为dict------")
        
        out.update(aux_dict)
        out.update(template_dict)
        out['backbone_feat'] = x
        return out
    
    
    
    
    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
                    主干的输出嵌入，它可以是（HW1+HW2,，B，C）或（HW2, B，C）
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)  搜索区域的编码器输出
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()  #torch.Size([32, 1, 768, 256]);bs=32, Nq=1, C=256, HW=768
        opt_feat = opt. view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map , max_score = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'out_score': max_score}
            return out
        else:
            raise NotImplementedError


def build_gfatrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('gfatrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''


    if cfg.MODEL.SELFBACKBONE.TYPE == 'vit_base_patch16_224':
        selfbackbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)   #vit_base_patch16_224预训练模型？？
        hidden_dim = selfbackbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.SELFBACKBONE.TYPE == 'vit_base_patch16_224_ce':
        selfbackbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.SELFBACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.SELFBACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = selfbackbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.SELFBACKBONE.TYPE == 'vit_large_patch16_224_ce':
        selfbackbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.SELFBACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.SELFBACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = selfbackbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError


    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)   #vit_base_patch16_224预训练模型？？
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError


    if cfg.MODEL.FinalBACKBONE.TYPE == 'vit_base_patch16_224':
        finalbackbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)   #vit_base_patch16_224预训练模型？？
        hidden_dim = finalbackbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.FinalBACKBONE.TYPE == 'vit_base_patch16_224_ce':
        finalbackbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.FinalBACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.FinalBACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = finalbackbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.FinalBACKBONE.TYPE == 'vit_large_patch16_224_ce':
        finalbackbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.FinalBACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.FinalBACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = finalbackbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError
    
    if cfg.MODEL.DECODER.TYPE == 'vit_base_patch16_224':
        maeattn = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)   #vit_base_patch16_224预训练模型？？
        hidden_dim = finalbackbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    selfbackbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)  #微调
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)  #微调
    finalbackbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)  #微调
    maeattn.finetune_track(cfg=cfg, patch_start_index=patch_start_index) 
    maedecoder=build_maedecoder(cfg=cfg,block=maeattn.blocks,p_emd=maeattn.pos_embed_x)
    box_head = build_box_head(cfg, hidden_dim)

    model = gfatrack(
        selfbackbone,
        backbone,
        finalbackbone,
        maedecoder,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'gfatrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
