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
    query_animal_list = ['cows','bear','camel']

    retrival_list = []

    query_dir = '/projects/perception/datasets/DAVIS'

    embed_dir_list = ['/home/yuefanw/yuefanw/CASA_code/clip_tmp/data_new']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    result_dict = {}

    for query_animal in query_animal_list:

        frame_list = []
        for frame in range(30):
            img_dir = os.path.join(query_dir,'JPEGImages','Full-Resolution',query_animal,'{:05d}.jpg'.format(frame+1))
            mask_dir = os.path.join(query_dir,'Annotations','Full-Resolution',query_animal,'{:05d}.png'.format(frame+1))

            raw_image = np.array(Image.open(img_dir))
            mask = np.array(Image.open(mask_dir))
            mask[mask>0] = 1
            image = np.expand_dims(mask, axis=-1) * raw_image

            frame_list.append(preprocess(Image.fromarray(image)).to(device))
        image_all = torch.stack(frame_list)
        image_features = model.encode_image(image_all)

        query_embed = image_features.detach().cpu().numpy()
        # pdb.set_trace()


        per_frame_save_number = 5

        vote_list = []

        for i in tqdm(range(len(query_embed))):

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
                        # print(distance)
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


