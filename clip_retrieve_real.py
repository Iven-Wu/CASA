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

    embed_dir = './dataset/embeddings'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for query_animal in query_animal_list:

        min_distance = 1000
        real_img_root = os.path.join(query_dir, 'JPEGImages','Full-Resolution',query_animal)

        real_img_list = []
        for real_frame in sorted(os.listdir(real_img_root)):
            # img_dir = os.path.join(query_dir,'JPEGImages','Full-Resolution',query_animal,'{:05d}.jpg'.format(frame+1))
            # mask_dir = os.path.join(query_dir,'Annotations','Full-Resolution',query_animal,'{:05d}.png'.format(frame+1))

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
        # pdb.set_trace()


        for animal_voxel in tqdm(os.listdir(embed_dir)):
            if animal_voxel[:-4] in val_split_list:
                continue

            voxel_data = np.load(os.path.join(embed_dir, animal_voxel))
            for query_embed_i in query_embed:


                sorted_distance = np.sort(np.linalg.norm(voxel_data - query_embed_i, axis=1))


                distance = sorted_distance[:5].mean()

                # pdb.set_trace()
                if distance < min_distance:
                    min_distance = distance
                    animal_choose = animal_voxel[:-4]
            # pdb.set_trace()

        print("For animal {}, Nearest animal is {}".format(query_animal.split('.')[0], animal_choose))

        retrival_list.append(animal_choose)

    return val_split_list,retrival_list

if __name__ == '__main__':
    retrieve()


