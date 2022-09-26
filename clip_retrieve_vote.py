import os
import pdb

import torch
from PIL import Image
import numpy as np
import sys

import clip

from tqdm import tqdm
from collections import Counter

from utils.split_utils import val_split_list

def retrieve():

    retrival_list = []
    query_dir = '/home/yuefanw/yuefanw/CASA_code/clip_tmp/data_v9_30'
    embed_dir_list = ['/home/yuefanw/yuefanw/CASA_code/clip_tmp/data_new']

    result_dict = {}

    for query_animal in val_split_list:
        query_embed = np.load(os.path.join(query_dir,query_animal+'.npy'))
        per_frame_save_number = 5
        vote_list = []
        for i in range(len(query_embed)):

            query_embed_i = query_embed[i]

            distance_record = np.array([1000.] * per_frame_save_number)
            nearest_name_record = [0] * per_frame_save_number

            for embed_dir in embed_dir_list:

                for animal_voxel in os.listdir(embed_dir):
                    if animal_voxel[:-4] in val_split_list:
                        continue

                    voxel_data = np.load(os.path.join(embed_dir,animal_voxel))

                    sorted_distance = np.sort(np.linalg.norm(voxel_data - query_embed_i, axis=1))
                    distance = sorted_distance[:5].mean()

                    if distance<np.max(distance_record) and animal_voxel[:-4] not in nearest_name_record:

                        sub_index = np.argmax(distance_record)
                        distance_record[sub_index] = distance
                        nearest_name_record[sub_index] = animal_voxel[:-4]

            vote_list.extend(nearest_name_record)
        tmp_res = Counter(vote_list)

        choose_animal = sorted(tmp_res.items(), key=lambda x: x[1], reverse=True)[0][0]
        tmp_dict = {}
        for t in choose_animal:
            tmp_dict[t] = tmp_res[t]

        result_dict[query_animal] = tmp_dict

        print("For animal {}, Nearest animals are {}".format(query_animal, choose_animal))

        retrival_list.append(choose_animal)

    return val_split_list,retrival_list

if __name__ == '__main__':
    retrieve()

