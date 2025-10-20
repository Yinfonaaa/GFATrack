import numpy as np
from lib.models.layers import utils
import torch
import os
from PIL import Image
############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg
# def temp_heatmap(search_img,input,keep_index,,videoname,id):#serach_img ndnpaary 128,128,3
#     search=Image.fromarray(search_img)
#     search.save('/root/workspace/dyf/code/0924_0603_138/heatmap/search.jpg')
#     num_stages = len(input)
#     temp=torch.rand(1,128).cuda()
#     viz=[]
#     for i in range(1, 3):
#          remove_index[i]=torch.cat( [remove_index[i-1],remove_index[i]],dim=1)
#     for i in range(0, 3):#前三
#             map = torch.zeros_like(temp).scatter_(dim=1, index=keep_index[i].to(torch.int64), src=input[i])
#             input_=map.view(1,16,16)
#             viz.append(utils.heat_map(input_,videoname,id,i))
#     for i in range(3,num_stages):
#             map = torch.zeros_like(temp).scatter_(dim=1, index=keep_index[i].to(torch.int64), src=input[i])
#             map[0][remove_index[2][0].long()]=0
#             input_=map.view(1,16,16)
#             viz.append(utils.heat_map(input_,videoname,id,i))
#     viz=np.concatenate(viz,axis=1)
#     heat_jpg=Image.fromarray(viz)
#     if not os.path.exists('/root/workspace/dyf/code/0924_0603_138/heatmap/'+videoname):
#              os.mkdir('/root/workspace/dyf/code/0924_0603_138/heatmap/'+videoname)
#     save_path = '/root/workspace/dyf/code/0924_0603_138/heatmap/'+videoname+'/'+str(id)+'.jpg'
#     heat_jpg.save(save_path)
#     return viz

def heatmap(search_img,input,keep_index,videoname,id):#serach_img ndnpaary 256,256,3
    search=Image.fromarray(search_img)
    search.save('/root/workspace/dyf/code/0924_0603_138/heatmap/search.jpg')
    num_stages = len(input)
    temp=torch.rand(1,256).cuda()
    viz=[]
    for i in range(1, 3):
         remove_index[i]=torch.cat( [remove_index[i-1],remove_index[i]],dim=1)
    for i in range(0, 3):#前三
            map = torch.zeros_like(temp).scatter_(dim=1, index=keep_index[i].to(torch.int64), src=input[i])
            input_=map.view(1,16,16)
            viz.append(utils.heat_map(input_,videoname,id,i))
    for i in range(3,num_stages):
            map = torch.zeros_like(temp).scatter_(dim=1, index=keep_index[i].to(torch.int64), src=input[i])
            map[0][remove_index[2][0].long()]=0
            input_=map.view(1,16,16)
            viz.append(utils.heat_map(input_,videoname,id,i))
    viz=np.concatenate(viz,axis=1)
    heat_jpg=Image.fromarray(viz)
    if not os.path.exists('/root/workspace/dyf/code/0924_0603_138/heatmap/'+videoname):
             os.mkdir('/root/workspace/dyf/code/0924_0603_138/heatmap/'+videoname)
    save_path = '/root/workspace/dyf/code/0924_0603_138/heatmap/'+videoname+'/'+str(id)+'.jpg'
    heat_jpg.save(save_path)
    return viz
def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    # if num_stages>3:
    #     for i in range(1, 3):
    #         mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)
    #     for i in range(4, num_stages):
    #         mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)
    # else:
    for i in range(1, num_stages):
            mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)
                                #      (16,16,16,16,3) 转置1和2,                           (256,16,16,3)                                    
    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)##numpy.ndarray 256,256,3
    return viz
