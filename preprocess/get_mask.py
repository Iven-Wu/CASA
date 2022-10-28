import cv2
import glob
import numpy as np
import pdb
import os

import argparse

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

import sys
from detectron2.projects import point_rend

def gen_mask(img_dir):
    # coco_metadata = MetadataCatalog.get("coco_2017_val")

    # img_dir = '/scratch/users/yuefanw/dataset/3DPW/imageFiles/courtyard_basketball_01/'

    img_outdir = 'dataset/JPEGImages/Full-Resolution/custom'
    mask_outdir = 'dataset/Annotations/Full-Resolution/custom'
    os.makedirs(img_outdir,exist_ok=True)
    os.makedirs(mask_outdir,exist_ok=True)

    detbase = './detectron2'

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.3
    cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'

    predictor = DefaultPredictor(cfg)

    expected_frames = 30

    counter=0
    for i,path in enumerate(sorted(glob.glob('%s/*'%img_dir))[:expected_frames]):
        print(path)
        img = cv2.imread(path)
        shape = img.shape[:2]
        mask = np.zeros(shape)

        imgt = img
        segs = predictor(imgt)['instances'].to('cpu')

        for it,ins_cls in enumerate(segs.pred_classes):
            print(ins_cls)
            #if ins_cls ==15: # cat
            # mask += np.asarray(segs.pred_masks[it])
            if ins_cls==0 or (ins_cls >= 14 and ins_cls <= 23):
                mask += np.asarray(segs.pred_masks[it])

        if (mask.sum())<1000: continue

        mask = mask.astype(bool).astype(int)*128
        mask = np.concatenate([mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]],-1)
        mask[:,:,:2] = 0

        cv2.imwrite('%s/%05d.jpg'%(img_outdir,counter), img)
        cv2.imwrite('%s/%05d.png'%(mask_outdir,counter), mask)


        counter+=1

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--img_dir', type=str, default='/scratch/users/yuefanw/dataset/3DPW/imageFiles/courtyard_basketball_01/', help='path to image sequence')
    args = parser.parse_args()