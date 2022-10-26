import os
import pdb

import torch
from PIL import Image
import numpy as np
import sys

import clip
import yaml
from easydict import EasyDict as edict

from tqdm import tqdm
from utils.split_utils import val_split_list

#
def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj

def retrieve(config,config_path):
    embedding_dir = './dataset/embeddings'

    retrival_list = []
    for real_animal in val_split_list:

        min_distance = 1000
        query_embed = np.load(os.path.join(embedding_dir, real_animal + '.npy'))

        for animal_voxel in tqdm(os.listdir(embedding_dir)):
            if animal_voxel[:-4] in val_split_list:
                continue

            voxel_data = np.load(os.path.join(embedding_dir, animal_voxel))
            for query_embed_i in query_embed:

                sorted_distance = np.sort(np.linalg.norm(voxel_data-query_embed_i,axis=1))
                distance = sorted_distance[:5].mean()

                if distance<min_distance:
                    min_distance = distance
                    animal_choose = animal_voxel[:-4]
                    
        print("For animal {}, Nearest animal is {}".format(real_animal.split('.')[0], animal_choose))

        retrival_list.append(animal_choose)

    yaml.dump(edict2dict(config), open(config_path, 'w'), default_flow_style=False)
    # return val_split_list,retrival_list


def retrieve_real(config):

    query_dir = config.data.input_dir
    embedding_dir = config.data.embedding_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    query_animal = config.data.test_animal

    min_distance = 1000
    real_img_root = os.path.join(query_dir, 'JPEGImages','Full-Resolution',query_animal)

    real_img_list = []
    for real_frame in sorted(os.listdir(real_img_root)):
        real_img_dir = os.path.join(real_img_root, real_frame)
        real_img = np.array(Image.open(real_img_dir))

        real_mask = np.array(Image.open(real_img_dir.replace('JPEGImages', 'Annotations').replace('jpg', 'png')))
        if len(real_mask.shape) == 3:
            real_mask = real_mask[:, :, 0]
        # pdb.set_trace()
        real_mask[real_mask != 0] = 1
        real_img = np.expand_dims(real_mask, axis=-1) * real_img

        real_image_encode = preprocess(Image.fromarray(real_img)).to(device)
        real_img_list.append(real_image_encode)
    image_all = torch.stack(real_img_list)
    image_features = model.encode_image(image_all)

    query_embed = image_features.detach().cpu().numpy()


    for animal_voxel in tqdm(os.listdir(embedding_dir)):
        if animal_voxel[:-4] in val_split_list:
            continue

        voxel_data = np.load(os.path.join(embedding_dir, animal_voxel))
        for query_embed_i in query_embed:
            sorted_distance = np.sort(np.linalg.norm(voxel_data - query_embed_i, axis=1))

            distance = sorted_distance[:5].mean()

            if distance < min_distance:
                min_distance = distance
                animal_choose = animal_voxel[:-4]

    print("For animal {}, Nearest animal is {}".format(query_animal.split('.')[0], animal_choose))

    config.data.retrieval_aniaml = animal_choose
    return config

def retrieve_synthetic(config):
    embedding_dir = config.data.embedding_dir

    test_animal = config.data.test_animal
    min_distance = 1000
    query_embed = np.load(os.path.join(embedding_dir, test_animal + '.npy'))

    for animal_voxel in tqdm(os.listdir(embedding_dir)):
        if animal_voxel[:-4] in val_split_list:
            continue

        voxel_data = np.load(os.path.join(embedding_dir, animal_voxel))
        for query_embed_i in query_embed:

            sorted_distance = np.sort(np.linalg.norm(voxel_data - query_embed_i, axis=1))
            distance = sorted_distance[:5].mean()

            if distance < min_distance:
                min_distance = distance
                animal_choose = animal_voxel[:-4]

    print("For animal {}, Nearest animal is {}".format(test_animal.split('.')[0], animal_choose))
    config.data.retrieval_animal = animal_choose
    return config

def retrieve_single(config, config_path):
    if config.type == 'synthetic':
        config = retrieve_synthetic(config)

    elif config.type == 'real':
        config = retrieve_real(config)

    yaml.dump(edict2dict(config), open(config_path, 'w'), default_flow_style=False)

    return config

if __name__ =='__main__':
    retrieve()
