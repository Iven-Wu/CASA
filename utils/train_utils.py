import torch
from torchvision.ops import masks_to_boxes
import soft_renderer as sr
from queue import Queue
import os
import numpy as np
import random

seed = 2000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

### init


def gen_mask_from_basic(basic_mesh1, faces1, intrin1, extrin1, soft_renderer):
    basic_mesh1 = basic_mesh1[:, :, [0, 2, 1, 3]]
    basic_mesh1[:, :, 1] *= -1
    verts1 = torch.matmul(intrin1[0], torch.matmul(extrin1[0], basic_mesh1.permute(0, 2, 1))).permute(0, 2, 1)
    depth_z1 = verts1[:, :, [2]]
    verts1 = verts1 / depth_z1
    verts1 = (verts1 - 512) / 512
    verts1[:, :, 1] *= -1
    mesh1 = sr.Mesh(verts1, faces1)
    rendering_mask1 = soft_renderer.render_mesh(mesh1)
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




def forward_kinematic(vert_pos, skin_weight, bones_len_scale, R, T, ske, json_data, mesh_scale, ske_shift):
    # ske = ske_raw.copy()
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
        bone_vec = tail - head
        endpoint_weight = (torch.mul(vert_pos - head, bone_vec).sum(1) / (bone_vec ** 2).sum().item()).clamp(0., 1.)

        if "new_head" not in joint.keys():
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
        joint = ske[joint_name]
        joint['newnew_head'] = T.act(torch.tensor(joint['new_head']).cuda().float()[:, None])[:, 0].detach().cpu().numpy()
        joint['newnew_tail'] = T.act(torch.tensor(joint['new_tail']).cuda().float()[:, None])[:, 0].detach().cpu().numpy()
        for child in joint['children']:
            if not child in all_bones:
                continue
            q.put(child)

    return new_final_pos, final_pos


