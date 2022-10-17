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

from lietorch import SO3, SE3, LieGroupParameter

import trimesh
from PIL import Image
import soft_renderer as sr
from ChamferDistance.chamfer3D.dist_chamfer_3D import chamfer_3DDist

from IOU3D import IIOU
from imageio import imread

import copy
import json


from utils.utils import LBS
from utils.loss_utils import  OffsetNet
from utils.data_utils import read_obj,Optimization_data,readPFM
from utils.train_utils import get_scale_init, forward_kinematic

seed = 2000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

###############################################
################ Optimization #################
###############################################
test_animal_name = ['aardvark_female', 'aardvark_juvenile', 'aardvark_male', 'african_elephant_female', 'african_elephant_male',\
                    'african_elephant_juvenile','binturong_female','binturong_juvenile','binturong_male','grey_seal_female',\
                    'grey_seal_juvenile','grey_seal_male','bonobo_juvenile','bonobo_male','bonobo_female','polar_bear_female',\
                    'polar_bear_juvenile','polar_bear_male','gray_wolf_female','gray_wolf_juvenile','gray_wolf_male',\
                    'common_ostrich_female','common_ostrich_juvenile','common_ostrich_male']


retrieval_animal_name = ['koala_female', 'arctic_wolf_juvenile', 'babirusa_juvenile', 'indian_elephant_male', 'indian_elephant_juvenile',\
                    'indian_elephant_juvenile','giant_otter_male','cuviers_dwarf_caiman_female','cuviers_dwarf_caiman_female','babirusa_juvenile',\
                    'koala_juvenile','giant_otter_female','bornean_orangutan_juvenile','western_chimpanzee_juvenile','western_chimpanzee_juvenile','formosan_black_bear_male',\
                    'spotted_hyena_juvenile','grizzly_bear_male','spotted_hyena_female','arctic_wolf_juvenile','spotted_hyena_male',\
                    'american_bison_juvenile','babirusa_juvenile','greater_flamingo_juvenile']


def optimize(test_animal,retrieval_animal,raw_info_dir,retrieve_info_dir,retrieve_skeleton_dir,out_path):

        w_mask, w_flow, w_smooth,w_symm = 1e4,1e6,1e6,1e4

        renderer_soft = sr.SoftRenderer(image_size=1024, sigma_val=1e-5,
                                        camera_mode='look_at', perspective=False, aggr_func_rgb='hard',
                                        light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)
        renderer_softtex = sr.SoftRenderer(image_size=1024, sigma_val=1e-4, gamma_val=1e-2,
                                           camera_mode='look_at', perspective=False, aggr_func_rgb='softmax',
                                           light_mode='vertex', light_intensity_ambient=1., light_intensity_directionals=0.)

        ####
        start_idx, end_idx = 0, 30
        ####

        points_info, normals_info, face_info = read_obj(os.path.join(retrieve_skeleton_dir, 'remesh.obj'))

        faces = torch.tensor(face_info).cuda()

        print("Mesh point number is ", points_info.shape[0])
        points_info = np.concatenate((points_info, np.ones((points_info.shape[0], 1))), axis=1)

        ### initialize with gt info
        W = torch.tensor(np.load(os.path.join(retrieve_skeleton_dir, 'W1.npy')), requires_grad=True,
                         device="cuda")  # Initialization with skinning weights of retrieval animal
        N, Frames, B = points_info.shape[0], end_idx - start_idx, W.shape[1]


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

        #####################################

        ## deal with memory issue
        if points_info.shape[0] > 3500:
            B_size = 15
        else:
            B_size = 30

        train_loader = DataLoader(Optimization_data(test_animal, start_idx, end_idx, raw_info_dir), batch_size=B_size, drop_last=True,
                                shuffle=False, num_workers=0)
        print("Dataloader Length: ", len(train_loader))

        offset = torch.zeros((basic_mesh.shape[0], 3), requires_grad=True, device="cuda")
        offset_net = OffsetNet().cuda()
        zeros_tensor = torch.zeros_like(offset[:, [-1]], requires_grad=False, device="cuda")

        bones_len_scale = torch.ones((B, 1), requires_grad=True, device="cuda")
        shifting = torch.zeros((1, 3), requires_grad=True, device="cuda")
        mesh_scale = torch.ones((1), requires_grad=True, device="cuda")

        optimizer = optim.Adam([mesh_scale], lr=1e-3)  # for optimization w/ basic shape offset
        optimizer.add_param_group({"params": shifting, 'lr': 5e-2})
        cham_loss = chamfer_3DDist()

        epoch_num = 202
        flag = 0
        for epoch_id in range(epoch_num):
            if epoch_id <= 60 and epoch_id % 15 == 14:
                for params in optimizer.param_groups:
                    params['lr'] *= 0.5

            if epoch_id > 60 and flag == 0:
                optimizer = optim.Adam([p1, p2], lr=4e-3)
                optimizer.add_param_group({"params": [mesh_scale], 'lr': 1e-4})
                optimizer.add_param_group({"params": [bones_len_scale], 'lr': 4e-3})
                optimizer.add_param_group({"params": offset_net.parameters(), 'lr': 1e-3})
                flag = 1

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


                SO3_R = SO3.InitFromVec(p1)
                SE3_T = SE3.InitFromVec(p2)

                optimizer.zero_grad()

                ###　basic transformation
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


                W1 = F.softmax(W*10)
                W1 = (W1 / (W1.sum(1, keepdim=True).detach()))

                ### load skeleton and skinning
                ske = np.load(os.path.join(retrieve_info_dir, 'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()[
                    'frame_000001']
                for key in ske.keys():
                    head, tail = ske[key]['head'], ske[key]['tail']
                    head[[1]] *= -1
                    tail[[1]] *= -1
                    head, tail = head[[0, 2, 1]], tail[[0, 2, 1]]
                    ske[key]['head'] = head
                    ske[key]['tail'] = tail
                with open(os.path.join(retrieve_info_dir, 'weight', '{}.json'.format(retrieval_animal)), 'r', encoding='utf8') as fp:
                    json_data = json.load(fp)


                wbx, _ = forward_kinematic(x[:, :-1].clone(), W1.clone(), bones_len_scale, SO3_R[index[0]:index[-1] + 1], SE3_T[index[0]:index[-1] + 1], ske, json_data, mesh_scale, ske_shift.detach().cpu().numpy()
)

                ### diff rendering

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
                ##################################################################
                ######################### Loss Functions #########################
                ##################################################################

                ### mask loss
                rendering_mask = renderer_soft.render_mesh(mesh1)
                rendering_mask = rendering_mask[:, -1]
                loss_mask = F.mse_loss(mask, rendering_mask)
                if not epoch_id > 60:
                    loss = loss_mask * w_mask
                    print("Loss for epoch {}, iter {} : mask: {:.2f}".format(epoch_id, iter_id, loss_mask.item() * w_mask))
                    sys.stdout.flush()
                    loss.backward()
                    optimizer.step()
                    continue

                ### flow loss
                mesh_flow = sr.Mesh(verts[:-1], faces.repeat(index.shape[0] -1, 1, 1), textures=verts[1:],
                                    texture_type="vertex")

                rendering_uv = renderer_softtex.render_mesh(mesh_flow)

                rendering_grid = torch.Tensor(
                    np.meshgrid(range(rendering_mask.shape[2]), range(rendering_mask.shape[1]))).cuda()
                rendering_grid[0] = rendering_grid[0] * 2 / (rendering_mask.shape[2]) - 1
                rendering_grid[1] = 1 - rendering_grid[1] * 2 / (rendering_mask.shape[2])
                rendering_grid = rendering_grid[None].repeat(rendering_uv.shape[0], 1, 1, 1)
                rendering_flow = rendering_uv[:, :2] - rendering_grid
                rendering_flow[:, 1] *= -1
                flow_mask = (rendering_mask[:-1, None].clone().detach().bool() | mask[:-1, None].bool())

                loss_flow = F.mse_loss(rendering_flow * flow_mask, flow[:-1] * flow_mask)


                ### smoothing loss
                gt_bone = torch.zeros((SO3_R.shape[0] - 1, SO3_R.shape[1], 4)).cuda()
                gt_bone[:, :, -1] += 1
                gt_bone2 = torch.zeros((SE3_T.shape[0] - 1, SE3_T.shape[1], 4)).cuda()
                gt_bone2[:, :, -1] += 1

                loss_bone = F.mse_loss(SO3_R[:-1].inv().mul(SO3_R[1:]).vec(), gt_bone)

                loss_bone3 = F.mse_loss(SE3_T[:-1].inv().mul(SE3_T[1:]).vec()[:, :, 3:], gt_bone2)

                ### symmetry loss
                x_reverse = offset_x.clone().detach()
                x_reverse[:, 0] *= -1
                loss_chamfer = cham_loss(offset_x[None, :, :-1], x_reverse[None, :, :-1])[0].mean()


                loss = loss_mask * w_mask + loss_flow * w_flow + (loss_bone + loss_bone3) * w_smooth + loss_chamfer * w_symm
                ##################################################################
                print(
                    "Loss for epoch {}, iter {} : mask: {:.2f}, flow: {:.2f},  bone: {:.2f}, chamfer: {:.2f}".format(
                        epoch_id, iter_id, loss_mask.item() * w_mask,
                        loss_flow.item() * w_flow,  (loss_bone + loss_bone3) * w_smooth, loss_chamfer * w_symm))


                sys.stdout.flush()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                if epoch_id % 20 == 1:
                    SO3_R = SO3.InitFromVec(p1)
                    SE3_T = SE3.InitFromVec(p2)
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
                    shifting_numpy = shifting.cpu().detach().numpy()

                    np.save(os.path.join(temp_out_path, 'temparray', 'W1'), W1_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'old_mesh'), old_mesh_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'p1'), p1_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'p2'), p2_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'x'), x_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'bones_len_scale'), bones_len_scale_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'mesh_scale'), mesh_scale_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'shifting'), shifting_numpy)

                    ske = np.load(os.path.join(retrieve_info_dir, 'skeleton', 'skeleton_all_frames.npy'), allow_pickle=True).item()[
                    'frame_000001']
                    for key in ske.keys():
                        head, tail = ske[key]['head'], ske[key]['tail']
                        head[[1]] *= -1
                        tail[[1]] *= -1
                        head, tail = head[[0, 2, 1]], tail[[0, 2, 1]]
                        ske[key]['head'] = head
                        ske[key]['tail'] = tail
                    with open(os.path.join(retrieve_info_dir, 'weight', '{}.json'.format(retrieval_animal)), 'r', encoding='utf8') as fp:
                        json_data = json.load(fp)

                    x = basic_mesh.clone()  # for optimization w/o basic shape offset
                    offset = offset_net(x[:, :-1].cuda())
                    homo_offset = torch.cat([offset, zeros_tensor], dim=1)  # for optimization w/ basic shape offset
                    x = x + homo_offset  # for optimization w/ basic shape offset
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


                    wbx_results, wbx_no_root = forward_kinematic(x[:, :-1].clone(), W1.clone(), bones_len_scale, SO3_R, SE3_T, ske, json_data, mesh_scale, ske_shift.detach().cpu().numpy()
)
                    wbx_numpy = wbx_results.cpu().detach().numpy()
                    wbx_no_root_numpy = wbx_no_root.cpu().detach().numpy()
                    np.save(os.path.join(temp_out_path, 'temparray', 'wbx'), wbx_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'wbx_noroot'), wbx_no_root_numpy)
                    np.save(os.path.join(temp_out_path, 'temparray', 'ske'), ske)
                    faces_final = faces[None].clone()
                    ########## Evaluate IOU Metric ##########

                    for ind, vertices in enumerate(wbx_results):
                        sr.Mesh(vertices, faces_final).save_obj(os.path.join(temp_out_path, 'Frame{}.obj'.format(ind + 1)))
                        sr.Mesh(wbx_no_root_numpy[ind], faces_final).save_obj(os.path.join(temp_out_path2, 'Frame{}.obj'.format(ind + 1)))
                        sr.Mesh(x_beforetran_numpy[:, :-1], faces_final).save_obj(os.path.join(temp_out_path3, 'Frame{}.obj'.format(ind + 1)))


if __name__ == '__main__':


    for test_animal_ind in range(len(test_animal_name)):
        test_animal = test_animal_name[test_animal_ind]
        retrieval_animal = retrieval_animal_name[test_animal_ind]

        raw_info_dir = '/projects/perception/datasets/animal_videos/version9/{}/'.format(test_animal)
        retrieve_info_dir = '/projects/perception/datasets/animal_videos/version9/{}/'.format(retrieval_animal)
        retrieve_skeleton_dir = '/home/yuefanw/yuefanw/CASA_code/simplified_meshes/{}/'.format(retrieval_animal)
        out_path = '/home/yuefanw/scratch/test_optim_ske-camready_final/{}/'.format(test_animal)

        os.makedirs(out_path,exist_ok=True)
        os.makedirs(os.path.join(out_path, 'temparray'),exist_ok=True)

        optimize(test_animal,retrieval_animal,raw_info_dir,retrieve_info_dir,retrieve_skeleton_dir,out_path)