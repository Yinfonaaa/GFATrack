import math

from lib.models.gfatrack import build_gfatrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from PIL import Image
from lib.test.tracker.vis_utils import gen_visualization,heatmap
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target , sample_target_gt
from copy import deepcopy
# for debug
import cv2
import os

import numpy as np
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond



class GFATrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(GFATrack, self).__init__(params)
        network = build_gfatrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.z_dict2={}

        #-----------------------
        self.z_dict_list = []
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)
        # self.num_extra_template = len(self.update_intervals)
        self.num_extra_template = 1

        self.cached_frames = []

        # self.z_patch_arr2 = self.z_dict_list[1]

    def initialize(self, image, info: dict):
        #-----------------------
        # initialize z_dict_list
        self.z_dict_list = []
        #-----------------------

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        self.z_patch_arr2 = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        z_gt_patch_arr, z_gt_amask_arr = sample_target_gt(image, info['init_bbox'], output_sz=self.params.template_size)
        self.z_gt_patch_arr = z_gt_patch_arr
        self.last_template_update_frame = self.frame_id
        template_gt = self.preprocessor.process(z_gt_patch_arr, z_gt_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template.tensors
            self.z_dict2 = template_gt

            #-----------------------
            self.z_dict_list.append(self.z_dict1)
            for i in range(self.num_extra_template):
                self.z_dict_list.append(deepcopy(self.z_dict1))
            #-----------------------

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, len_seq, seq_name,info: dict = None):
        update_point = len_seq//3
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            # out_dict = self.network.forward(
            #     template=self.z_dict1.tensors, search=x_dict.tensors,template_gt=self.z_dict2.tensors , ce_template_mask=self.box_mask_z)
            # self.z_dict_list = [torch.stack(nt_list) for nt_list in torch.nested_tensor.utils.nested_unbind(self.z_dict_list)]
            
            out_dict = self.network.forward(
                template=self.z_dict_list, search=x_dict.tensors,template_gt=self.z_dict2.tensors , ce_template_mask=self.box_mask_z)
  
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes , max_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # get confidence score (whether the search region is reliable)
        conf_score = out_dict['out_score'].view(-1).sigmoid().item()

        
        # # update template(got)
        # if self.frame_id <= update_point:
        #     if self.frame_id == 1:
        #         self.conf_threshold = conf_score
        #     else:
        #         self.conf_threshold = (self.conf_threshold * (self.frame_id - 1) + conf_score) / self.frame_id 

        # if self.frame_id > update_point and conf_score > self.conf_threshold:
        #     # check if enough frames have passed since the last template update
        #     if self.frame_id - self.last_template_update_frame >= 40:
        #         # update template with current frame's information
        #         z_patch_arr2, _, z_amask_arr2 = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
        #         template_t = self.preprocessor.process(z_patch_arr2, z_amask_arr2)
        #         with torch.no_grad():
        #             z_dict_t = template_t.tensors
        #         self.z_dict_list[1] = z_dict_t
                    
        #         # # update z_dict_list with new template
        #         # if self.signal_frame == 0:
        #         #     self.z_dict_list[1] = z_dict_t
        #         #     self.signal_frame = 1
        #         # elif self.signal_frame == 1:
        #         #     self.z_dict_list[2] = z_dict_t
        #         #     self.signal_frame = 0
                    
        #         # update last template update frame
        #         self.last_template_update_frame = self.frame_id
        #         print("Updated template with current frame's information:",self.frame_id)
        #     else:
        #         pass
        
        # update template(lasot)
        if self.frame_id <= 200:
            if self.frame_id == 1:
                self.conf_threshold = conf_score
            else:
                self.conf_threshold = (self.conf_threshold * (self.frame_id - 1) + conf_score) / self.frame_id 

        if self.frame_id > 200 and conf_score > self.conf_threshold:
            # check if enough frames have passed since the last template update
            if self.frame_id - self.last_template_update_frame >= 100:
                # update template with current frame's information
                z_patch_arr2, _, z_amask_arr2 = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
                self.z_patch_arr2 = z_patch_arr2
                template_t = self.preprocessor.process(z_patch_arr2, z_amask_arr2)
                with torch.no_grad():
                    z_dict_t = template_t.tensors
                self.z_dict_list[1] = z_dict_t

                # update the confidence threshold
                self.conf_threshold = (self.conf_threshold + conf_score) / 2
                
                # update last template update frame
                self.last_template_update_frame = self.frame_id
                print("Updated template with current frame's information:",self.frame_id)
            else:
                pass

        """
        # update template
        if conf_score <= 0.5:
            # check if we have at least one frame with conf_score > 0.65 in the cache
            max_conf = -1
            max_conf_frame = None
            for cached_frame in self.cached_frames:
                if cached_frame['conf_score'] > 0.5 and cached_frame['conf_score'] > max_conf:
                    max_conf = cached_frame['conf_score']
                    max_conf_frame = cached_frame

            if max_conf_frame is not None:
                # update template with the frame with max conf_score
                z_patch_arr2, _, z_amask_arr2 = sample_target(max_conf_frame['image'], max_conf_frame['state'], 
                                                            self.params.template_factor, output_sz=self.params.template_size)
                self.z_patch_arr2 = z_patch_arr2
                template_t = self.preprocessor.process(z_patch_arr2, z_amask_arr2)
                with torch.no_grad():
                    z_dict_t = template_t.tensors
                self.z_dict_list[1] = z_dict_t
                self.last_template_update_frame = self.frame_id
                # print("按照分数更新了",self.frame_id,"分数为：", max_conf)

        else:
            # check if the template needs to be updated
            if self.frame_id - self.last_template_update_frame >= self.update_intervals:
                # update template
                max_conf = -1
                max_conf_frame = None
                for cached_frame in self.cached_frames:
                    if cached_frame['conf_score'] > 0.6 and cached_frame['conf_score'] > max_conf:
                        max_conf = cached_frame['conf_score']
                        max_conf_frame = cached_frame

                if max_conf_frame is not None:
                    # update template with the frame with max conf_score
                    z_patch_arr2, _, z_amask_arr2 = sample_target(max_conf_frame['image'], max_conf_frame['state'], 
                                                                self.params.template_factor, output_sz=self.params.template_size)
                    self.z_patch_arr2 = z_patch_arr2
                    template_t = self.preprocessor.process(z_patch_arr2, z_amask_arr2)
                    with torch.no_grad():
                        z_dict_t = template_t.tensors
                    self.z_dict_list[1] = z_dict_t
                    self.last_template_update_frame = self.frame_id
                    # print("按照间隔更新了",self.frame_id)

        # cache the current frame info
        if len(self.cached_frames) == 8:
            self.cached_frames.pop(0)
        self.cached_frames.append({'frame_id': self.frame_id, 'conf_score': conf_score, 'image': image, 'state': self.state})
        """

       
        #
        # for debug
        if self.debug:
             
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                #热力图
                heat=heatmap(x_patch_arr,out_dict['attn'],out_dict['keep_indexs'],out_dict["removed_indexes_s"],seq_name,self.frame_id)
                self.visdom.register(torch.from_numpy(heat).permute(2, 0, 1), 'image', 1, 'heat_map')
                ##
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(torch.from_numpy(self.z_patch_arr2).permute(2, 0, 1), 'image', 1, 'template2')
                # self.visdom.register(torch.from_numpy(self.template_pro).permute(2, 0, 1), 'image', 1, 'template_pro')
                self.visdom.register(torch.from_numpy(self.z_gt_patch_arr).permute(2, 0, 1), 'image', 1, 'template_gt')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                self.visdom.register(torch.from_numpy(cv2.resize(self.z_patch_arr2,dsize=(256,256),interpolation=cv2.INTER_LINEAR)).permute(2, 0, 1), 'image', 1, 'Dtemplate2')
                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                    removed_indexes_t = out_dict['removed_indexes_t']
                    removed_indexes_t = [removed_indexes_t_i.cpu().numpy() for removed_indexes_t_i in removed_indexes_t]
                    masked_template = gen_visualization(cv2.resize(self.z_patch_arr2,dsize=(256,256),interpolation=cv2.INTER_LINEAR), removed_indexes_t)
                    self.visdom.register(torch.from_numpy(masked_template).permute(2, 0, 1), 'image', 1, 'masked_template')
                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        # if self.save_all_boxes:
        #     '''save all predictions'''
        #     all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
        #     all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
        #     return {"target_bbox": self.state,
        #             "all_boxes": all_boxes_save}
        # else:
        #     return {"target_bbox": self.state}
        
        #-----------------------
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score}
        #-----------------------

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return GFATrack
