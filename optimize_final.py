import os
import pdb
import warnings
warnings.filterwarnings("ignore")

import sys
import copy
import random
import numpy as np
from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.ops import masks_to_boxes

from lietorch import SO3, SE3, LieGroupParameter

# import wandb
import trimesh
from PIL import Image
import soft_renderer as sr
from ChamferDistance.chamfer3D.dist_chamfer_3D import chamfer_3DDist
# from chamferdist import ChamferDistance

from IOU3D import IIOU
from imageio import imread

# import wandb
from queue import Queue
import copy
import json

from utils.utils import per_frame_energy_plot, compute_color,flow_to_image, LBS, distance_matrix

from utils.loss_utils import ARAPLoss, LaplacianLoss,Preframe_ARAPLoss,Preframe_LaplacianLoss, OffsetNet

from utils.data_utils import read_obj,Optimization_data,readPFM



def forward_kinematic(vert_pos, skin_weight, bones_len_scale, R, T, ske, json_data, mesh_scale, ske_shift):
    all_bones = json_data['group_name']
    # all_bones = list(ske.keys())
    bones_dict = {}  # record index for each bone
    Rs_dict = {}  # record transformations of all parents for each bone
    for ind, bone in enumerate(all_bones):
        bones_dict[bone] = ind
        Rs_dict[bone] = []

    final_pos = torch.zeros_like(vert_pos[None].repeat(R.shape[0], 1, 1), requires_grad=True, device="cuda")  # final position
    root = ske['def_c_root_joint']
    q = Queue(maxsize=(len(bones_dict.keys()) + 1))
    q1 = Queue(maxsize=(len(bones_dict.keys()) + 1))
    root['new_head'] = root['head']  # new head/tail are the postion of head/tail after deformation
    root['new_tail'] = root['tail']
    root['tail_before_R'] = root['tail']
    for child in root['children']:
        if not child in all_bones:
            continue
        q.put(child)

    bone_sequence = []
    while not q.empty():  # record transformations of all parents for each bone in Rs_dict
        joint_name = q.get()
        bone_sequence.append(joint_name)
        bone_ind = bones_dict[joint_name]
        joint = ske[joint_name]
        for child in joint['children']:
            if not child in all_bones:
                continue
            q.put(child)
            Rs_dict[child].extend((Rs_dict[joint_name] + [bone_ind]))

    for child in root['children']:
        if not child in all_bones:
            continue
        q.put(child)

    # for bone in all_bones:
    #     if 'fur' in bone:
    #         continue
    #     joint = ske[bone]
    #     bone_ind = bones_dict[bone]
    #     bone_offset = w_offset[:, bone_ind].sum(0).detach().cpu().numpy()
    #     norm_scalar = skin_weight[:, bone_ind].sum().item()
    #     joint['head'] += bone_offset / norm_scalar
    #     joint['tail'] += bone_offset / norm_scalar

    while not q.empty():  # Apply transformation

        joint_name = q.get()
        joint = ske[joint_name]
        all_R_inds = Rs_dict[joint_name]
        bone_ind = bones_dict[joint_name]
        bone_scale = bones_len_scale[bone_ind]

        if "scale_head" not in joint.keys():
            joint['scale_head'] = joint['head'] * mesh_scale.clamp(0.15, 8).item()
            joint['scale_tail'] = joint['tail'] * mesh_scale.clamp(0.15, 8).item()
            joint['scale_head'] += ske_shift
            joint['scale_tail'] += ske_shift
        head, tail = torch.from_numpy(joint['scale_head']).float().cuda(), torch.from_numpy(joint['scale_tail']).float().cuda()
        origin_local_pos = vert_pos - head
        bone_vec = tail - head
        endpoint_weight = (torch.mul(vert_pos - head, bone_vec).sum(1) / (bone_vec ** 2).sum().item()).clamp(0., 1.)

        if "new_head" not in joint.keys():
            # joint['new_head'] = joint['head']
            # joint['new_tail'] = joint['tail']
            # joint['tail_before_R'] = joint['tail']
            joint['new_head'] = joint['scale_head'][:, None].repeat(R.shape[0], 1).transpose(1, 0)
            joint['new_tail'] = joint['scale_tail'][:, None].repeat(R.shape[0], 1).transpose(1, 0)
            joint['tail_before_R'] = joint['new_tail']

        new_head, new_tail = torch.from_numpy(joint['new_head']).float().cuda(), torch.from_numpy(
                joint['new_tail']).float().cuda()

        updated_vert_pos = vert_pos[None].repeat(R.shape[0], 1, 1).clone()
        if not len(all_R_inds) == 0:
            for R_ind in all_R_inds:  # apply transformations of parent skeletons
                R_bone_scale = bones_len_scale[R_ind]
                R_origin_head, R_origin_tail = torch.from_numpy(
                    ske[all_bones[R_ind]]['scale_head']).float().cuda(), torch.from_numpy(
                    ske[all_bones[R_ind]]['scale_tail']).float().cuda()
                R_head, R_tail = torch.from_numpy(ske[all_bones[R_ind]]['new_head']).float().cuda(), torch.from_numpy(
                    ske[all_bones[R_ind]]['tail_before_R']).float().cuda()  # take bone head as rotation center

                R_endpoint_weight = (torch.mul((vert_pos - R_origin_head), (R_origin_tail - R_origin_head)).sum(1) / (
                            (R_origin_tail - R_origin_head) ** 2).sum().item()).clamp(0., 1.)
                local_pos_for_update = updated_vert_pos - R_head[:, None]  # local position
                local_pos_for_update += (R_bone_scale - 1) * (R_endpoint_weight[:, None, None] * (R_tail - R_head)[None].repeat(vert_pos.shape[0], 1, 1)).permute(1, 0, 2)
                updated_local_vert_pos = R[:, [R_ind]].act(local_pos_for_update)  # transform based on lietorch
                updated_vert_pos = R_head[:, None] + updated_local_vert_pos  # global position

        local_pos = updated_vert_pos - new_head[:, None]
        local_pos += (bone_scale - 1) * (endpoint_weight[:, None, None] * (new_tail - new_head)[None].repeat(vert_pos.shape[0], 1, 1)).permute(1, 0, 2)
        # local_pos = updated_vert_pos - new_head[None]
        # local_pos += endpoint_weight[:, None] * (bone_scale - 1) * (new_tail - new_head)
        updated_local_pos = R[:, [bone_ind]].act(local_pos)
        final_pos = final_pos + (new_head[:, None] + updated_local_pos) * skin_weight[:, [bone_ind]]  # LBS

        if not len(joint['children']) == 0:  # update positions of new_head/new_tail for all child bones
            for child in joint['children']:
                if not child in all_bones:
                    continue
                q.put(child)
                q1.put(child)
            while not q1.empty():
                child_joint_name = q1.get()
                child_joint = ske[child_joint_name]
                if "scale_head" not in child_joint.keys():
                    child_joint['scale_head'] = child_joint['head'] * mesh_scale.clamp(0.15, 8).item()
                    child_joint['scale_tail'] = child_joint['tail'] * mesh_scale.clamp(0.15, 8).item()
                    child_joint['scale_head'] += ske_shift
                    child_joint['scale_tail'] += ske_shift
                if "new_head" not in child_joint.keys():
                    child_joint['new_head'] = child_joint['scale_head'][:, None].repeat(R.shape[0], 1).transpose(1, 0)
                    child_joint['new_tail'] = child_joint['scale_tail'][:, None].repeat(R.shape[0], 1).transpose(1, 0)
                    child_joint['tail_before_R'] = child_joint['new_tail']
                c_head_old, c_tail_old = torch.from_numpy(child_joint['scale_head']).float().cuda(), torch.from_numpy(
                    child_joint['scale_tail']).float().cuda()
                child_head, child_tail = torch.from_numpy(child_joint['new_head']).float().cuda(), torch.from_numpy(
                    child_joint['new_tail']).float().cuda()
                local_head, local_tail = child_head[None] - new_head[None], child_tail[None] - new_head[None]
                endweight_head = (torch.mul(c_head_old - head, bone_vec).sum() / (bone_vec ** 2).sum()).clamp(0.,
                                                                                                              1.).item()
                endweight_tail = (torch.mul(c_tail_old - head, bone_vec).sum() / (bone_vec ** 2).sum()).clamp(0.,
                                                                                                              1.).item()
                local_head += endweight_head * (bone_scale - 1) * (new_tail - new_head)[None]
                local_tail += endweight_tail * (bone_scale - 1) * (new_tail - new_head)[None]
                updated_local_head = R[:, [bone_ind]].act(local_head.permute(1, 0, 2))
                updated_local_tail = R[:, [bone_ind]].act(local_tail.permute(1, 0, 2))
                child_joint['new_head'] = (updated_local_head[:, 0] + new_head).detach().cpu().numpy()
                child_joint['new_tail'] = (updated_local_tail[:, 0] + new_head).detach().cpu().numpy()
                if not len(child_joint['children']) == 0:
                    for child in child_joint['children']:
                        if not child in all_bones:
                            continue
                        q1.put(child)
        joint['tail_before_R'] = joint['new_tail']
        new_tail = (new_head + R[:, [bone_ind]].act((bone_scale * (new_tail - new_head)[:, None]))[:, 0]).detach().cpu().numpy()
        joint['new_tail'] = new_tail  # update tail position for the currently selected bone

    new_final_pos = T.act(final_pos.clone())
    for child in root['children']:
        if not child in all_bones:
            continue
        q.put(child)
    while not q.empty():
        joint_name = q.get()
        # print(joint_name)
        joint = ske[joint_name]
        all_R_inds = Rs_dict[joint_name]
        bone_ind = bones_dict[joint_name]
        joint['newnew_head'] = T.act(torch.tensor(joint['new_head']).cuda().float()[:, None])[:, 0].detach().cpu().numpy()
        joint['newnew_tail'] = T.act(torch.tensor(joint['new_tail']).cuda().float()[:, None])[:, 0].detach().cpu().numpy()
        for child in joint['children']:
            if not child in all_bones:
                continue
            q.put(child)

    return new_final_pos, final_pos

def gen_mask_from_basic(basic_mesh1, faces1, intrin1, extrin1, soft_renderer):
    basic_mesh1 = basic_mesh1[:, :, [0, 2, 1, 3]]
    basic_mesh1[:, :, 1] *= -1
    verts1 = torch.matmul(intrin1[0], torch.matmul(extrin1[0], basic_mesh1.permute(0, 2, 1))).permute(0, 2, 1)
    depth_z1 = verts1[:, :, [2]]
    verts1 = verts1 / depth_z1
    verts1 = (verts1 - 512) / 512
    verts1[:, :, 1] *= -1
    mesh1 = sr.Mesh(verts1, faces1)
    rendering_mask1 = renderer_soft.render_mesh(mesh1)
    rendering_mask1 = rendering_mask1[:, -1]
    return rendering_mask1


def get_scale_init(basic_mesh1, face1, intrin1, extrin1, mask1, soft_renderer):
    rendering_mask = gen_mask_from_basic(basic_mesh1[None], face1, intrin1, extrin1, soft_renderer)
    bx_min_render, by_min_render, bx_max_render, by_max_render = masks_to_boxes(rendering_mask)[0]
    bx_min_gt, by_min_gt, bx_max_gt, by_max_gt = masks_to_boxes(mask1)[0]
    x_len = bx_max_gt - bx_min_gt
    y_len = by_max_gt - by_min_gt
    if x_len > y_len:
        x_len_render = bx_max_render - bx_min_render
        init_scale = x_len / x_len_render
    else:
        y_len_render = by_max_render - by_min_render
        init_scale = y_len / y_len_render
    basic_mesh_scaled = basic_mesh1 * init_scale
    rendering_mask_scaled = gen_mask_from_basic(basic_mesh_scaled[None], face1, intrin1, extrin1, soft_renderer)
    bx_min_render, by_min_render, bx_max_render, by_max_render = masks_to_boxes(rendering_mask_scaled)[0]
    bx_min_gt, by_min_gt, bx_max_gt, by_max_gt = masks_to_boxes(mask1)[0]
    mid_x_render, mid_y_render = (bx_min_render + bx_max_render) / 2, (by_min_render + by_max_render) / 2
    mid_x_gt, mid_y_gt = (bx_min_gt + bx_max_gt) / 2, (by_min_gt + by_max_gt) / 2
    trans = torch.tensor([mid_x_gt - mid_x_render, mid_y_gt - mid_y_render])
    return init_scale, trans




###############################################
################ Optimization #################
###############################################
test_animal_name = ['aardvark_female', 'aardvark_juvenile', 'aardvark_male', 'african_elephant_female', 'african_elephant_male',\
                    'african_elephant_juvenile','binturong_female','binturong_juvenile','binturong_male','grey_seal_female',\
                    'grey_seal_juvenile','grey_seal_male','bonobo_juvenile','bonobo_male','bonobo_female','polar_bear_female',\
                    'polar_bear_juvenile','polar_bear_male','gray_wolf_female','gray_wolf_juvenile','gray_wolf_male',\
                    'common_ostrich_female','common_ostrich_juvenile','common_ostrich_male']
# retrieval_animal_name = ['arctic_wolf_female', 'dhole_male', 'babirusa_juvenile', 'indian_elephant_male', 'indian_elephant_juvenile',\
#                     'indian_elephant_juvenile','cuviers_dwarf_caiman_juvenile','koala_juvenile','cuviers_dwarf_caiman_male','babirusa_juvenile',\
#                     'arctic_wolf_female','cuviers_dwarf_caiman_female','western_lowland_gorilla_female','western_lowland_gorilla_female','western_lowland_gorilla_female','formosan_black_bear_male',\
#                     'himalayan_brown_bear_male','grizzly_bear_male','arctic_wolf_female','dhole_male','african_wild_dog_female',\
#                     'pronghorn_antelope_juvenile','dhole_male','greater_flamingo_juvenile']
# retrieval_animal_name = ['arctic_wolf_juvenile', 'jaguar_male', 'common_warthog_female', 'indian_elephant_female', 'indian_elephant_female',\
#                     'indian_elephant_female','jaguar_male','cuviers_dwarf_caiman_female','jaguar_male','common_warthog_female',\
#                     'sun_bear_male','giant_otter_female','bornean_orangutan_male','western_chimpanzee_juvenile','western_chimpanzee_juvenile','babirusa_male',\
#                     'snow_leopard_female','babirusa_male','jaguar_male','jaguar_male','dhole_male',\
#                     'american_bison_juvenile','bengal_tiger_female','cassowary_female']
# retrieval_animal_name = ['koala_juvenile', 'jaguar_juvenile', 'arctic_wolf_juvenile', 'indian_elephant_juvenile', 'indian_elephant_male',\
#                     'sun_bear_female','jaguar_juvenile','diamondback_terrapin','indian_peafowl_juvenile','llama_female',\
#                     'king_penguin_female','american_bison_juvenile','western_chimpanzee_juvenile','western_chimpanzee_male','bactrian_camel_female','giant_panda_male',\
#                     'dall_sheep_juvenile','giant_panda_male','gemsbok_juvenile','arctic_wolf_female','jaguar_male',\
#                     'dingo_juvenile','cuviers_dwarf_caiman_female','indian_peafowl_juvenile']

retrieval_animal_name = ['koala_female', 'arctic_wolf_juvenile', 'babirusa_juvenile', 'indian_elephant_male', 'indian_elephant_juvenile',\
                    'indian_elephant_juvenile','giant_otter_male','cuviers_dwarf_caiman_female','cuviers_dwarf_caiman_female','babirusa_juvenile',\
                    'koala_juvenile','giant_otter_female','bornean_orangutan_juvenile','western_chimpanzee_juvenile','western_chimpanzee_juvenile','formosan_black_bear_male',\
                    'spotted_hyena_juvenile','grizzly_bear_male','spotted_hyena_female','arctic_wolf_juvenile','spotted_hyena_male',\
                    'american_bison_juvenile','babirusa_juvenile','greater_flamingo_juvenile']

if __name__ == '__main__':
    # for test_animal_ind in range(len(test_animal_name)):
    # for test_animal_ind in range(16, 20):
    # for test_animal_ind in range(20, len(test_animal_name)):
    for test_animal_ind in range(len(test_animal_name)):
    # for test_animal_ind in range(16, 20):
        test_animal = test_animal_name[test_animal_ind]
        retrieval_animal = retrieval_animal_name[test_animal_ind]
        # wandb.init(project="Test-optim_ske-37", entity="team-wu", name="{}".format(test_animal), reinit=True)
        fpath = '/projects/perception/datasets/animal_videos/version9/{}/'.format(test_animal)
        fpath1 = '/projects/perception/datasets/animal_videos/version9/{}/'.format(retrieval_animal)
        fpath2 = '/home/yuefanw/yuefanw/CASA_code/simplified_meshes/{}/'.format(retrieval_animal)
        out_path = '/home/yuefanw/scratch/test_optim_ske-camready/{}/'.format(test_animal)
        result_path = os.path.join(out_path, 'Epoch_201')
        # if os.path.exists(result_path):
        #     continue
        # print(test_animal)
        # w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 = 1e4, 1e4, 1e5, 1e3, 1e4, 5e5, 0e1, 1e6, 0e3, 0e2
        w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 = 1e4, 1e4, 1e6, 0e3, 0e4, 0e5, 0e1, 1e6, 1e4, 0e2

        #### loss = loss_mask * w2 + loss_flow * w3 + loss_color * w1
        #           + (loss_bone + loss_bone3) * w8 + loss_chamfer * w9


        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(os.path.join(out_path, 'temparray')):
            os.makedirs(os.path.join(out_path, 'temparray'))
        seed = 2000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        device_ids = [Id for Id in range(torch.cuda.device_count())]

        renderer_soft = sr.SoftRenderer(image_size=1024, sigma_val=1e-5,
                                        camera_mode='look_at', perspective=False, aggr_func_rgb='hard',
                                        light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)
        renderer_softtex = sr.SoftRenderer(image_size=1024, sigma_val=1e-4, gamma_val=1e-2,
                                           camera_mode='look_at', perspective=False, aggr_func_rgb='softmax',
                                           light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)

        ####
        start_idx, end_idx = 0, 30
        ####

        points_info, normals_info, face_info = read_obj(os.path.join(fpath2, 'remesh.obj'))
        # sphere = trimesh.creation.icosphere()
        # sphere.vertices += 1
        # points_info, face_info = sphere.vertices, sphere.faces + 1
        # points_info *= 2
        print("Mesh point number is ", points_info.shape[0])
        points_info = np.concatenate((points_info, np.ones((points_info.shape[0], 1))), axis=1)

        W = torch.tensor(np.load(os.path.join(fpath2, 'W1.npy')), requires_grad=True,
                         device="cuda")  # Initialization with skinning weights of retrieval animal
        N, Frames, B = points_info.shape[0], end_idx - start_idx, W.shape[1]

        ### initialize with Kmeans

        # W = torch.zeros((N, B), requires_grad=True, device="cuda")
        # cluster_ids_x, cluster_centers = kmeans(X=torch.tensor(points_info), num_clusters=B, distance='euclidean', device='cuda')
        # Dist_matrix = distance_matrix(cluster_centers)
        # with torch.no_grad():
        #     for n in range(N):
        #         W[n,cluster_ids_x[n]] += 1.

        ### initialize with gt info

        W = torch.tensor(np.load(os.path.join(fpath2, 'W1.npy')), requires_grad=True,
                         device="cuda")  # Initialization with skinning weights of retrieval animal

        ### transformation initialization ###
        basic_mesh = torch.tensor(points_info).float().cuda().detach()
        p1 = torch.randn((Frames, B, 4), requires_grad=True, device="cuda")
        p2 = torch.randn((Frames, 1, 7), requires_grad=True, device="cuda")
        p1_init = np.array([0., 0., 0., 1.])
        p2_init = np.array([0., 0., 0., 0., 0., 0., 1.])
        with torch.no_grad():
            for f in range(Frames):
                for b in range(B):
                    p1[f, b] = torch.tensor(p1_init)

            for f in range(Frames):
                p2[f, -1][:3] = torch.tensor([0, 0, 0.01 * f],
                                            dtype=torch.float64)  # Animal translation (0.01 y-axis for each frame)
                p2[f, -1][3:] = torch.tensor(p1_init)

        p3 = torch.randn((Frames, B, 4), requires_grad=True, device="cuda")
        p3_init = np.array([0., 0., 0., 1.])
        with torch.no_grad():
            for f in range(Frames):
                for b in range(B):
                    p3[f, b] = torch.tensor(p3_init)
        #####################################

        SO3_R = SO3.InitFromVec(p1)
        SE3_T = SE3.InitFromVec(p2)
        # Free_R = SO3.InitFromVec(p3)

        if points_info.shape[0] > 3500:
            B_size = 15
        else:
            B_size = 30
        train_loader = DataLoader(Optimization_data(test_animal, start_idx, end_idx, fpath), batch_size=B_size, drop_last=True,
                                shuffle=False, num_workers=0)
        print("Dataloader Length: ", len(train_loader))

        offset = torch.zeros((basic_mesh.shape[0], 3), requires_grad=True, device="cuda")
        offset_net = OffsetNet().cuda()
        zeros_tensor = torch.zeros_like(offset[:, [-1]], requires_grad=False, device="cuda")
        vert_color = torch.ones_like(offset, requires_grad=True, device="cuda")
        with torch.no_grad():
            vert_color *= 0.5
        bones_len_scale = torch.ones((B, 1), requires_grad=True, device="cuda")
        shifting = torch.zeros((1, 3), requires_grad=True, device="cuda")
        mesh_scale = torch.ones((1), requires_grad=True, device="cuda")
        # optimizer = optim.Adam([p1], lr=5e-3) # for optimization w/o basic shape offset
        optimizer = optim.Adam([mesh_scale], lr=1e-3)  # for optimization w/ basic shape offset
        optimizer.add_param_group({"params": shifting, 'lr': 5e-2})
        # optimizer.add_param_group({"params": mesh_scale, 'lr': 1e-2})
        arap_loss = ARAPLoss(points_info, face_info)
        lap_loss = LaplacianLoss(points_info, face_info)
        cham_loss = chamfer_3DDist()
        mesh_retrieval = trimesh.load(os.path.join(fpath2, 'remesh.obj'))
        epoch_num = 202
        flag = 0
        for epoch_id in range(epoch_num):
            if epoch_id <= 60 and epoch_id % 15 == 14:
                for params in optimizer.param_groups:
                    params['lr'] *= 0.5
            # if epoch_id == 30:
            #     optimizer.add_param_group({"params": mesh_scale, 'lr': 2e-2})
            if epoch_id > 60 and flag == 0:
                # optimizer = optim.Adam(offset_net.parameters(), lr=4e-3)
                optimizer = optim.Adam([p1, p2, vert_color], lr=4e-3)
                optimizer.add_param_group({"params": [mesh_scale], 'lr': 1e-4})
                optimizer.add_param_group({"params": [bones_len_scale], 'lr': 4e-3})
                # optimizer.add_param_group({"params": [offset], 'lr': 4e-3})
                optimizer.add_param_group({"params": offset_net.parameters(), 'lr': 1e-3})
                # mesh_scale.requires_grad = False
                flag = 1
            # if epoch_id > 60 and epoch_id % 40 == 19:
            if epoch_id in [100, 140, 175]:
                for params in optimizer.param_groups:
                    params['lr'] *= 0.85

            print("MESH SCALE: ", mesh_scale.clamp(0.15, 8).item())
            for iter_id, data in enumerate(train_loader):
                intrin, extrin, mask, flow, index, color = data
                intrin, extrin, mask, flow, color = intrin.cuda(), extrin.cuda(), mask.cuda(), flow.permute(0, 3, 1,
                                                                                                            2).cuda(), color.cuda()

                if epoch_id == 0 and iter_id == 0:
                    with torch.no_grad():
                        init_scale, shift = get_scale_init(basic_mesh, torch.tensor(face_info).cuda(), intrin, extrin,
                                                           mask, renderer_soft)
                        mesh_scale *= init_scale

                # flow[:, 1] = -flow[:, 1] epoch_id, iter_id, loss_mask.item() * w2
                SO3_R = SO3.InitFromVec(p1)
                SE3_T = SE3.InitFromVec(p2)
                # Free_R = SO3.InitFromVec(p3)
                color_mask = mask.unsqueeze(1)
                color_mask = color_mask.repeat(1, color.shape[1], 1, 1)
                color = color_mask * color
                optimizer.zero_grad()

                x = basic_mesh.clone()  # for optimization w/o basic shape offset
                offset = offset_net(x[:, :-1].cuda())
                homo_offset = torch.cat([offset, zeros_tensor], dim=1)  # for optimization w/ basic shape offset
                x = x + homo_offset  # for optimization w/ basic shape offset
                offset_x = x.clone()
                bb_center = (x.max(0)[0] + x.min(0)[0]) / 2
                x[:, :-1] *= mesh_scale.clamp(0.15, 8).item()
                bb_new_center = (x.max(0)[0] + x.min(0)[0]) / 2
                x += bb_center - bb_new_center
                x[:, :-1] += shifting
                ske_shift = (bb_center - bb_new_center)[:-1] + shifting[0] + offset.mean(0)
                # bp()

                W1 = F.softmax(W*10)
                W1 = (W1 / (W1.sum(1, keepdim=True).detach()))
                # w_offset = offset[:, None] * W1[:, :, None]
                # wbx = LBS(x, W1, T, R)
                ske = np.load(os.path.join(fpath1, 'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()[
                    'frame_000001']
                for key in ske.keys():
                    head, tail = ske[key]['head'], ske[key]['tail']
                    head[[1]] *= -1
                    tail[[1]] *= -1
                    head, tail = head[[0, 2, 1]], tail[[0, 2, 1]]
                    ske[key]['head'] = head
                    ske[key]['tail'] = tail
                with open(os.path.join(fpath1, 'weight', '{}.json'.format(retrieval_animal)), 'r', encoding='utf8') as fp:
                    json_data = json.load(fp)

                wbx, _ = forward_kinematic(x[:, :-1].clone(), W1.clone(), bones_len_scale, SO3_R[index[0]:index[-1] + 1], SE3_T[index[0]:index[-1] + 1], ske, json_data, mesh_scale, ske_shift.detach().cpu().numpy()
)

                ones_tensor = torch.ones_like(wbx[:, :, [0]])
                wbx = torch.cat([wbx, ones_tensor], dim=2)
                wbx0 = wbx[:, :, [0, 2, 1, 3]]
                wbx0[:, :, 1] *= -1

                verts = torch.matmul(intrin, torch.matmul(extrin, wbx0.permute(0, 2, 1))).permute(0, 2, 1)
                depth_z = verts[:, :, [2]]
                verts = verts / depth_z
                verts = (verts - 512) / 512
                verts[:, :, 1] *= -1

                faces = torch.tensor(face_info).cuda()
                mesh1 = sr.Mesh(verts, faces.repeat(index.shape[0], 1, 1))
                rendering_mask = renderer_soft.render_mesh(mesh1)
                rendering_mask = rendering_mask[:, -1]

                mesh_color = sr.Mesh(verts, faces.repeat(index.shape[0], 1, 1), textures=vert_color, texture_type="vertex")
                rendering_color = renderer_softtex.render_mesh(mesh_color)[:,:-1]

                assert color_mask.shape == rendering_color.shape

                rendering_color = color_mask * rendering_color

                mesh_flow = sr.Mesh(verts[:-1], faces.repeat(index.shape[0] -1, 1, 1), textures=verts[1:],
                                    texture_type="vertex")
                # bp()
                # pdb.set_trace()

                rendering_uv = renderer_softtex.render_mesh(mesh_flow)

                rendering_grid = torch.Tensor(
                    np.meshgrid(range(rendering_mask.shape[2]), range(rendering_mask.shape[1]))).cuda()
                rendering_grid[0] = rendering_grid[0] * 2 / (rendering_mask.shape[2]) - 1
                rendering_grid[1] = 1 - rendering_grid[1] * 2 / (rendering_mask.shape[2])
                # rendering_grid[1] = rendering_grid[1] * 210 / (rendering_mask.shape[2]) - 1
                rendering_grid = rendering_grid[None].repeat(rendering_uv.shape[0], 1, 1, 1)
                rendering_flow = rendering_uv[:, :2] - rendering_grid
                rendering_flow[:, 1] *= -1
                flow_mask = (rendering_mask[:-1, None].clone().detach().bool() | mask[:-1, None].bool())

                ##################################################################
                ######################### Loss Functions #########################
                ##################################################################
                loss_mask = F.mse_loss(mask, rendering_mask)
                if not epoch_id > 60:
                    loss = loss_mask * w2
                    # loss = loss_mask * 1e4
                    print("Loss for epoch {}, iter {} : mask: {:.2f}".format(epoch_id, iter_id, loss_mask.item() * w2))
                    sys.stdout.flush()
                    loss.backward()
                    optimizer.step()
                    continue
                if epoch_id % 5 == 0:
                    mask_temp_path = os.path.join(out_path, 'Mask', "Epoch_{}".format(epoch_id))
                    if not os.path.exists(mask_temp_path):
                        os.makedirs(mask_temp_path)
                    for mask_ind in range(mask.shape[0]):
                        Image.fromarray((rendering_mask[mask_ind].detach().cpu().numpy() * 255).astype(np.uint8)).save(
                            os.path.join(mask_temp_path, 'rendering_mask{}.jpg'.format(mask_ind)))
                        Image.fromarray((mask[mask_ind].detach().cpu().numpy() * 255).astype(np.uint8)).save(
                            os.path.join(mask_temp_path, 'gt_mask{}.jpg'.format(mask_ind)))
                # loss_offset = F.mse_loss(offset, torch.zeros_like(offset))  # for optimization w/ basic shape offset
                loss_color = F.mse_loss(rendering_color, color)
                if epoch_id % 10 == 9:
                    color_temp_path = os.path.join(out_path, 'Color', "Epoch_{}".format(epoch_id))
                    if not os.path.exists(color_temp_path):
                        os.makedirs(color_temp_path)
                    for color_ind in range(color.shape[0]):
                        # pdb.set_trace()
                        Image.fromarray(
                            (rendering_color[color_ind, :].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
                                np.uint8)).save(
                            os.path.join(color_temp_path, 'rendering_color{}.jpg'.format(color_ind)))
                        Image.fromarray((color[color_ind, :].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
                            np.uint8)).save(
                            os.path.join(color_temp_path, 'gt_color{}.jpg'.format(color_ind)))

                loss_flow = F.mse_loss(rendering_flow * flow_mask, flow[:-1] * flow_mask)
                flow_temp_path = os.path.join(out_path, 'Flow', "Epoch_{}".format(epoch_id))
                if not os.path.exists(flow_temp_path):
                    os.makedirs(flow_temp_path)
                for flow_ind in range(rendering_flow.shape[0]):
                    Image.fromarray((flow_to_image((rendering_flow[flow_ind,:2] * (mask[flow_ind][None] > 1e-5)).permute(1, 2, 0).detach().cpu().numpy() * 1024)).astype(np.uint8)).save(
                        os.path.join(flow_temp_path, 'rendering_flow{}.jpg'.format(flow_ind)))
                    Image.fromarray((flow_to_image((flow[flow_ind, :2] * (mask[flow_ind][None] > 1e-5)).permute(1, 2, 0).detach().cpu().numpy() * 1024)).astype(np.uint8)).save(
                        os.path.join(flow_temp_path, 'gt_flow{}.jpg'.format(flow_ind)))

                gt_bone = torch.zeros((SO3_R.shape[0] - 1, SO3_R.shape[1], 4)).cuda()
                gt_bone[:, :, -1] += 1
                gt_bone2 = torch.zeros((SE3_T.shape[0] - 1, SE3_T.shape[1], 4)).cuda()
                gt_bone2[:, :, -1] += 1
                # loss_bone = F.mse_loss(p1[1:, :, :], p1[:-1, :, :])
                loss_bone = F.mse_loss(SO3_R[:-1].inv().mul(SO3_R[1:]).vec(), gt_bone)
                # loss_bone2 = F.mse_loss(Free_R[:-1].inv().mul(Free_R[1:]).vec(), gt_bone)
                loss_bone3 = F.mse_loss(SE3_T[:-1].inv().mul(SE3_T[1:]).vec()[:, :, 3:], gt_bone2)
                # loss_bone2 = F.mse_loss((R[:-1, :-1, :-1] @ torch.inverse(R[1:, :-1, :-1])), torch.eye(3)[None].repeat(R.shape[0] - 1, 1, 1).cuda())
                #loss_smooth = lap_loss(x)
                # loss_smooth = lap_loss(wbx[:, :, :-1])

                # loss_offset = torch.mean((offset ** 2)) # for optimization w/ basic shape offset
                x_reverse = offset_x.clone().detach()
                x_reverse[:, 0] *= -1
                loss_chamfer = cham_loss(offset_x[None, :, :-1], x_reverse[None, :, :-1])[0].mean()


                loss = loss_mask * w2 + loss_flow * w3 + loss_color * w1 + (loss_bone + loss_bone3) * w8 + loss_chamfer * w9
                ##################################################################
                print(
                    "Loss for epoch {}, iter {} : mask: {:.2f}, flow: {:.2f}, color: {:.2f}, bone: {:.2f}, chamfer: {:.2f}".format(
                        epoch_id, iter_id, loss_mask.item() * w2,
                        loss_flow.item() * w3, loss_color * w1, (loss_bone + loss_bone3) * w8, loss_chamfer * w9))


                sys.stdout.flush()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    if epoch_id % 20 == 0:
                        per_frame_energy_plot(color, mask, flow, rendering_color, rendering_mask, rendering_flow, flow_mask, wbx,
                                              points_info, face_info, epoch_id, iter_id, B_size, out_path)

            with torch.no_grad():
                if epoch_id % 20 == 1:
                    SO3_R = SO3.InitFromVec(p1)
                    SE3_T = SE3.InitFromVec(p2)
                    # Free_R = SO3.InitFromVec(p3)
                    temp_out_path = os.path.join(out_path, "Epoch_{}".format(epoch_id))
                    os.makedirs(temp_out_path,exist_ok=True)

                    temp_out_path2 = os.path.join(out_path, "Epoch_{}_noroot".format(epoch_id))
                    os.makedirs(temp_out_path2,exist_ok=True)

                    temp_out_path3 = os.path.join(out_path, "Epoch_{}_notrans".format(epoch_id))
                    os.makedirs(temp_out_path3,exist_ok=True)

                    os.makedirs(os.path.join(temp_out_path, 'temparray'),exist_ok=True)

                    W1_numpy = W1.cpu().detach().numpy()
                    old_mesh_numpy = basic_mesh.cpu().detach().numpy()
                    p1_numpy = p1.cpu().detach().numpy()
                    p2_numpy = p2.cpu().detach().numpy()
                    x_numpy = x.cpu().detach().numpy()
                    bones_len_scale_numpy = bones_len_scale.cpu().detach().numpy()
                    mesh_scale_numpy = mesh_scale.cpu().detach().numpy()
                    vert_color_numpy = vert_color.cpu().detach().numpy()
                    shifting_numpy = shifting.cpu().detach().numpy()

                    np.save(os.path.join(temp_out_path, 'temparray', 'W1'), W1_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'old_mesh'), old_mesh_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'p1'), p1_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'p2'), p2_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'x'), x_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'bones_len_scale'), bones_len_scale_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'mesh_scale'), mesh_scale_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'vert_color'), vert_color_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'shifting'), shifting_numpy)

                    ske = np.load(os.path.join(fpath1, 'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()[
                    'frame_000001']
                    for key in ske.keys():
                        head, tail = ske[key]['head'], ske[key]['tail']
                        head[[1]] *= -1
                        tail[[1]] *= -1
                        head, tail = head[[0, 2, 1]], tail[[0, 2, 1]]
                        ske[key]['head'] = head
                        ske[key]['tail'] = tail
                    with open(os.path.join(fpath1, 'weight', '{}.json'.format(retrieval_animal)), 'r', encoding='utf8') as fp:
                        json_data = json.load(fp)

                    x = basic_mesh.clone()  # for optimization w/o basic shape offset
                    offset = offset_net(x[:, :-1].cuda())
                    homo_offset = torch.cat([offset, zeros_tensor], dim=1)  # for optimization w/ basic shape offset
                    x = x + homo_offset  # for optimization w/ basic shape offset
                    offset_x = x.clone()
                    offset_numpy = offset.clone().detach().cpu().numpy()
                    offset_x_numpy = x.clone().detach().cpu().numpy()
                    np.save(os.path.join(temp_out_path, 'temparray', 'offset_x'), offset_x_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'offset'), offset_numpy)
                    bb_center = (x.max(0)[0] + x.min(0)[0]) / 2
                    x[:, :-1] *= mesh_scale
                    bb_new_center = (x.max(0)[0] + x.min(0)[0]) / 2
                    x += bb_center - bb_new_center
                    x[:, :-1] += shifting
                    x_beforetran_numpy = x.cpu().detach().numpy()
                    np.save(os.path.join(temp_out_path, 'temparray', 'x_beforetran'), x_beforetran_numpy)
                    ske_shift = (bb_center - bb_new_center)[:-1] + shifting[0] + offset.mean(0)

                    # w_offset = offset[:, None] * W1[:, :, None]

                    wbx_results, wbx_no_root = forward_kinematic(x[:, :-1].clone(), W1.clone(), bones_len_scale, SO3_R, SE3_T, ske, json_data, mesh_scale, ske_shift.detach().cpu().numpy()
)
                    wbx_numpy = wbx_results.cpu().detach().numpy()
                    wbx_no_root_numpy = wbx_no_root.cpu().detach().numpy()
                    np.save(os.path.join(temp_out_path, 'temparray', 'wbx'), wbx_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'wbx_noroot'), wbx_no_root_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'ske'), ske)
                    faces_final = faces[None].clone()
                    ########## Evaluate IOU Metric ##########
                    all_iou_value = 0
                    iou_sample_num = 0
                    for ind, vertices in enumerate(wbx_results):
                        sr.Mesh(vertices, faces_final).save_obj(os.path.join(temp_out_path, 'Frame{}.obj'.format(ind + 1)))
                        sr.Mesh(wbx_no_root_numpy[ind], faces_final).save_obj(os.path.join(temp_out_path2, 'Frame{}.obj'.format(ind + 1)))
                        sr.Mesh(x_beforetran_numpy[:, :-1], faces_final).save_obj(os.path.join(temp_out_path3, 'Frame{}.obj'.format(ind + 1)))