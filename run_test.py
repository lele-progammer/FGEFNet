import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for FGEFNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='output',
                        help='path where to save')
    parser.add_argument('--weight_path', default='./ckpt/ACF-03-31_18-23_SHHA_vgg16_bn_0.0001_best_mae.pth',
                        help='path where the trained weights saved')

    
#ckpt/ACF-03-31_18-23_SHHA_vgg16_bn_0.0001_best_mae.pth
#/ACFM(2)--04-01_10-47_SHHA_vgg16_bn_0.0001_best_mae.pth
#ckpt/04-01_19-27_SHHA_vgg16_bn_0.0001_best_mae.pth
    
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser


def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    print(args)
    device = torch.device('cuda')
    # get the FGEFNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    # img_path = "./vis/custom5.jpg"
    img_name = "img_0076"
    img_path = "./vis/" + img_name + ".jpg"
    print(img_path)
    gt_path = "./vis/" + img_name + "_ann.txt"
    # load the ground truth
    gt = []
    with open(gt_path) as f:
        for line in f.readlines():
            xy = []
            for i in line.split():
                xy.append(float(i))
            gt.append(xy)

    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    img_org = img_raw
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = 0.5
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    img_blank = img_to_draw
    blank_to_draw = np.zeros((new_height, new_width, 3))
    blank_to_draw[:] = [255, 0, 0]
    heat_to_draw = np.zeros((new_height, new_width))
    for p in points:
        # img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        blank_to_draw = cv2.circle(blank_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    img_name = img_path.split('/')[-1].split('.')[-2]
    model_name = args.weight_path.split('/')[-1].split('.')[-2]
    att_map = []
    for fea_map in outputs['att_map']:
        fea_map = torch.mean(fea_map, dim=1).permute(1, 2, 0).cpu().detach().numpy()
        print(fea_map.shape)

        fea_map = (fea_map - fea_map.min()) / (fea_map.max() - fea_map.min()) * 255
        fea_map = fea_map.astype(np.uint8)

        fea_map = cv2.resize(fea_map, (img_blank.shape[1], img_blank.shape[0]), interpolation=cv2.INTER_LINEAR)
        fea_map = cv2.applyColorMap(fea_map.astype(np.uint8), cv2.COLORMAP_JET)
        fea_map = cv2.addWeighted(fea_map.astype(np.uint8), 0.5, img_blank.astype(np.uint8), 0.5, 0)
        att_map.append(fea_map)
    att_map_final = cv2.vconcat(att_map)
    cv2.imwrite(
        os.path.join(args.output_dir, '{}_model_{}_pred{}_att_map.jpg'.format(img_name, model_name, predict_cnt)),
        att_map_final)

    # 未经滤波的预测点图
    cv2.imwrite(os.path.join(args.output_dir, '{}_model_{}_pred{}_blank.jpg'.format(img_name, model_name, predict_cnt)), blank_to_draw)
    heat_to_draw = blank_to_draw[:, :, 2]
    # heat_to_draw = cv2.GaussianBlur(heat_to_draw, (3, 3), 1)
    heat_to_draw = heat_to_draw.astype(np.uint8)
    heat_to_draw = cv2.applyColorMap(heat_to_draw, cv2.COLORMAP_JET)  # cv2.COLORMAP_JET

    # 滤波后的预测点图
    cv2.imwrite(os.path.join(args.output_dir, '{}_model_{}_pred{}_heat.jpg'.format(img_name, model_name, predict_cnt)),
                heat_to_draw)

    # 预测点图与原图结合
    img_final = cv2.addWeighted(heat_to_draw.astype(np.uint8), 0.5, img_blank.astype(np.uint8), 0.5, 0)
    img_final = cv2.vconcat([att_map_final, img_final])
    cv2.imwrite(os.path.join(args.output_dir, '{}_model_{}_pred{}_final.jpg'.format(img_name, model_name, predict_cnt)),
                img_final)

    print('saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FGEFNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)