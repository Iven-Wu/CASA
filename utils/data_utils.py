import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import os
import numpy as np
import cv2
import random

seed = 2000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class Optimization_data(data.Dataset):
    def __init__(self, config):
        super().__init__()
        animal = config.data.test_animal
        start = config.data.start_idx
        end = config.data.end_idx
        info_dir = os.path.join(config.data.info_dir,animal)

        self.all_info_list = [np.load(os.path.join(info_dir, 'info', '%04d.npz' % (i + 1))) for i in range(start, end)]
        self.color_imgs = [cv2.imread(os.path.join(info_dir, 'info', '%04d.png' % (i + 1))).transpose((2, 0, 1)) for i in
                           range(start, end)]

    def __getitem__(self, index):
        intrin = torch.tensor(self.all_info_list[index]['intrinsic_mat']).float()
        extrin = torch.tensor(self.all_info_list[index]['extrinsic_mat']).float()
        mask = torch.tensor(self.all_info_list[index]['segmentation_masks'] / 255).float()
        flow = torch.tensor(self.all_info_list[index]['optical_flow'] / 1024).float()
        color = torch.tensor(self.color_imgs[index]).float() / 255.0
        ret_ind = torch.tensor([index])

        return intrin, extrin, mask, flow, ret_ind, color

    def __len__(self):
        return len(self.all_info_list)

class Optimization_data_LASR(data.Dataset):
    def __init__(self, animal, start=1, end=49, fpath2='/home/yuefanw/scratch/planetzoo_rendering_cpu/aardvark_female/'):
        super().__init__()

        self.all_info_list = [np.load(os.path.join(fpath2, 'info', '%04d.npz' % (i + 1))) for i in range(start, end)]
        self.color_imgs = [imread(os.path.join(fpath2, 'info', '%04d.png' % (i + 1))).transpose((2, 0, 1)) for i in
                           range(start, end)]
        self.flow_path = '/home/yuefanw/scratch/lasr/database/DAVIS/FlowFW/Full-Resolution/{}/'.format(animal)
        self.flow_list = sorted(os.listdir(self.flow_path))

    def __getitem__(self, index):
        intrin = torch.tensor(self.all_info_list[index]['intrinsic_mat']).float()
        extrin = torch.tensor(self.all_info_list[index]['extrinsic_mat']).float()
        mask = torch.tensor(self.all_info_list[index]['segmentation_masks'] / 255).float()
        # flow = torch.tensor(self.all_info_list[index]['optical_flow'] / 1024).float()
        flow = torch.tensor(readPFM(os.path.join(self.flow_path, self.flow_list[index]))[0][::-1][:, :, :-1] / 1024).float()
        color = torch.tensor(self.color_imgs[index]).float() / 255.0
        ret_ind = torch.tensor([index])

        return intrin, extrin, mask, flow, ret_ind, color

    def __len__(self):
        return len(self.all_info_list)


class Optimization_data_real(data.Dataset):
    def __init__(self,start=1,end=49,root_dir='/home/yuefanw/scratch/lasr/database_davis/DAVIS/',real_animal='bear',
                 cam_root_dir='/scratch/users/yuefanw/Optimization/maxp_info_davis/gt_bear_re_grizzly_bear_juvenile_iou_0.73_0/'):
        super(Optimization_data_real, self).__init__()

        self.image_dir = os.path.join(root_dir,'JPEGImages','Full-Resolution',real_animal)
        self.mask_dir = os.path.join(root_dir,'Annotations','Full-Resolution',real_animal)

        self.flow_dir = os.path.join(root_dir,'FlowFW','Full-Resolution',real_animal)
        self.cam_loc_list = np.load(os.path.join(cam_root_dir,'loc.npy'))[start:end]

        self.image_list = [cv2.imread(os.path.join(self.image_dir,i)).transpose((2,0,1)) for i in sorted(os.listdir(self.image_dir))[start:end]]
        self.mask_list = [cv2.imread(os.path.join(self.mask_dir,i))[:,:,0] for i in sorted(os.listdir(self.mask_dir))[start:end]]

        self.flow_name_list = sorted([os.path.join(self.flow_dir,i) for i in os.listdir(self.flow_dir) if 'flo' in i])[start:end]
        self.flow_list = []
        for i in sorted(self.flow_name_list):
            flow = readPFM(os.path.join(root_dir, 'FlowFW', 'Full-Resolution', real_animal, i))[0][::-1][:, :, :-1]
            self.flow_list.append(flow)

        self.cam_intrin = torch.tensor(np.array([[2.4178e+03, 0.0000e+00, 5.1150e+02],
                                [0.0000e+00, 2.4178e+03, 5.1150e+02],
                                [0.0000e+00, 0.0000e+00, 1.0000e+00]]))


    def __getitem__(self, item):
        image = torch.tensor(self.image_list[item] / 255.0).float()
        mask = torch.tensor(self.mask_list[item] / 255.0).float()
        intrin = self.cam_intrin.float()

        cam_loc = torch.tensor(self.cam_loc_list[0]).float()
        flow = torch.tensor(self.flow_list[item] / 1024.).float()

        return intrin, cam_loc, image, mask, flow, item

    def __len__(self):
        return len(self.image_list)

def normalize(vec):
    normalized_vec = vec / torch.norm(vec, dim=1, keepdim=True)
    return normalized_vec

def rotm_from_lookat(lookat, up_axis):
    if len(up_axis.shape) == 1:
        up_axis = up_axis.unsqueeze(0)
    assert up_axis.shape == (1, 3)

    if len(lookat.shape) == 3:
        lookat = lookat.squeeze(1)



    up_axis = up_axis.repeat(lookat.shape[0], 1)

    z_axis = normalize(lookat)
    x_axis = normalize(torch.cross(up_axis, z_axis))
    y_axis = normalize(torch.cross(z_axis, x_axis))

    R = torch.stack([x_axis, y_axis, z_axis], dim=1).permute(0, 2, 1)  # cv2world
    return R

def world_to_cam(rot_mat, trans):

    rot_mat = rot_mat.view(-1,3,3)
    trans = trans.view(-1,3)
    R_bcam2cv = torch.tensor([[[1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]]]).repeat(rot_mat.shape[0], 1, 1).float().cuda()
    # location = np.array([extrin_mat.decompose()[0]]).T
    location = trans.T


    # R_world2bcam = np.array(extrin_mat.decompose()[1].to_matrix().transposed())
    R_world2bcam = rot_mat.T
    # T_world2bcam = torch.matmul(-1 * R_world2bcam, location)
    T_world2bcam = torch.bmm(-1 * R_world2bcam.reshape(-1, 3, 3), location.reshape(-1, 3, 1))
    R_world2cv = torch.bmm(R_bcam2cv, R_world2bcam.permute(2, 0, 1))
    T_world2cv = torch.matmul(R_bcam2cv, T_world2bcam)

    extr = torch.cat((R_world2cv, T_world2cv), dim=-1)

    return extr


### custom read obj

def read_obj(obj_path, for_open_mesh=False):
    with open(obj_path) as file:
        flag = 0
        points = []
        normals = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == 'o' and flag == 0:
                flag = 1
                continue
            elif strs[0] == 'o' and flag == 1:
                break
            if strs[0] == 'v':
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))

            if strs[0] == 'vn':
                normals.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == 'f':
                single_line_face = strs[1:]

                f_co = []
                for sf in single_line_face:
                    face_tmp = sf.split('/')[0]
                    f_co.append(face_tmp)
                if for_open_mesh == False:
                    if len(f_co) == 3:
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[2])))
                    elif len(f_co) == 4:
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[2])))
                        faces.append((int(f_co[0]), int(f_co[1]), int(f_co[3])))
                        faces.append((int(f_co[1]), int(f_co[2]), int(f_co[3])))
                        faces.append((int(f_co[0]), int(f_co[3]), int(f_co[2])))
                else:
                    faces.append([int(ver) for ver in f_co])

    points = np.array(points)

    normals = np.array(normals)
    #### here minus 1
    faces = np.array(faces) -1
    return points, normals, faces


### read flow from LASR generation

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if (sys.version[0]) == '3':
        header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    if (sys.version[0]) == '3':
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    else:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    if (sys.version[0]) == '3':
        scale = float(file.readline().rstrip().decode('utf-8'))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


