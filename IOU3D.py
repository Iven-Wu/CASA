import numpy as np
import trimesh
from pdb import set_trace as bp

import os
import torch
from torch.utils.data import DataLoader


class IIOU():

    def __init__(self,):

        self.resolution = 256
        self.b_min = np.array([-1, -1, -1])
        self.b_max = np.array([1.5, 1.5, 1.5])
        self.transform = None
        self.num_samples = 2097152
        self.max_score = -100


    def create_grid(self, resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):

        coords = np.mgrid[:resX, :resY, :resZ]
        coords = coords.reshape(3, -1)
        coords_matrix = np.eye(4)
        length = b_max - b_min
        coords_matrix[0, 0] = length[0] / resX
        coords_matrix[1, 1] = length[1] / resY
        coords_matrix[2, 2] = length[2] / resZ
        coords_matrix[0:3, 3] = b_min
        coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
        if transform is not None:
            coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
            coords_matrix = np.matmul(transform, coords_matrix)
        coords = coords.reshape(3, resX, resY, resZ)
        return coords, coords_matrix

    def normalize(self, mesh):
        scale = 4 / np.max(mesh.extents)
        matrix = np.eye(4)
        matrix[:3, :3] *= scale
        mesh.apply_transform(matrix)

        return mesh


    def center(self,mesh):
        bb = mesh.bounds
        b_center = np.mean(bb,axis=0)
        mesh.vertices -= b_center
        return mesh

    def compute_metric_obj(self,mesh_gt,mesh_tar):
        if type(mesh_gt)==str:
            mesh_gt = trimesh.load(mesh_gt)
        if type(mesh_tar)==str:
            mesh_tar = trimesh.load(mesh_tar)

        mesh_gt = self.center(mesh_gt)
        mesh_tar = self.center(mesh_tar)

        coords, mat = self.create_grid(self.resolution, self.resolution, self.resolution,
                            self.b_min, self.b_max)
        points = coords.reshape([3, -1]).T

        pred_gt = mesh_gt.contains(points)
        pred_tar = mesh_tar.contains(points)
        intersection = np.logical_and(pred_gt,pred_tar).astype(np.int32).sum()
        union = np.logical_or(pred_gt,pred_tar).astype(np.int32).sum()

        iou = intersection/union

        del pred_tar, pred_gt, mesh_tar, mesh_gt, points

        return iou

# if __name__ =='__main__':
#     oo = IIOU()
#     gt_mesh = '/scratch/users/yuefanw/version8/aardvark_male/frame_000001.obj'
#     tar_mesh = '/scratch/users/yuefanw/ppp/results_recon_cpu123/aardvark_male/0001.obj'
#     gt_mesh = tar_mesh
#     iou = oo.compute_metric_obj(gt_mesh,tar_mesh)
#     print(iou)




