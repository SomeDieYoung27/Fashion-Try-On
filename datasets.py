import json
from os import path as osp

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils import data
from torchvision import transforms

class VITONDataset(data.Dataset):
    def __init__(self,opt):
        super(VITONDataset,self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataset_dir,opt.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        #load data list
        img_names = []
        c_names = []

        with open(osp.join(opt.dataset_dir,opt.dataset_list),'r')as f :
            for line in f.readlines():
                img_name,c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)

            self.img_names = img_names
            self.c_names = dict()
            self.c_names['unpaired'] = c_names


    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        #mask arms

        for parse_id,pose_ids in [(14,[2,5,6,7] ),(15,[5,2,3,4])]:
            mask_arm = Image.new('L',(self.load_width,self.load_height),'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)

            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx,pointy = pose_data[i]
                radius = r*4 if i ==pose_ids[-1] else r *15