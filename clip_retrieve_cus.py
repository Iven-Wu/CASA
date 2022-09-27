import os
import pdb

import torch
from PIL import Image
import numpy as np
import sys

import clip

from tqdm import tqdm
from utils.split_utils import val_split_list
#


def retrieve():
    embed_dir = '/home/yuefanw/yuefanw/CASA_code/clip_tmp/data_new'
    info_dir = '/home/yuefanw/scratch/version9'


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    retrival_list = []
    for real_animal in val_split_list:

        min_distance = 1000
        query_embed = np.load(os.path.join(embed_dir, real_animal + '.npy'))

        # pdb.set_trace()
        for animal_voxel in tqdm(os.listdir(embed_dir)):
            if animal_voxel[:-4] in val_split_list:
                continue

            voxel_data = np.load(os.path.join(embed_dir, animal_voxel))
            for query_embed_i in query_embed:

                sorted_distance = np.sort(np.linalg.norm(voxel_data-query_embed_i,axis=1))
                # pdb.set_trace()
                distance = sorted_distance[:5].mean()

                if distance<min_distance:
                    min_distance = distance
                    animal_choose = animal_voxel[:-4]
            # pdb.set_trace()

        print("For animal {}, Nearest animal is {}".format(real_animal.split('.')[0], animal_choose))

        retrival_list.append(animal_choose)

    return val_split_list,retrival_list

if __name__ =='__main__':
    retrieve()
