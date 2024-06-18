# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable
from scipy import spatial as ss
from scipy.optimize import linear_sum_assignment
from numpy.core.fromnumeric import transpose

import torch

import util.misc as utils
from util.misc import *
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms
import cv2


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis_one(sample, target, pred, vis_dir, exp_name, epoch, writer, des=None):
    '''
    samples -> tensor: [3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''

    # gts = [t['point'].tolist() for t in targets]
    gts = target['point'] 
    pil_to_tensor = standard_transforms.ToTensor()

    x = []

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one 
    sample = restore_transform(sample)  # tensor to img
    
    sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255 # numpy (3, 128, 128)
    
    # to ndarray
    sample_origin = sample.transpose([1, 2, 0]).astype(np.uint8).copy()
    sample_gt = sample.transpose([1, 2, 0]).astype(np.uint8).copy()
    sample_pred = sample.transpose([1, 2, 0]).astype(np.uint8).copy()


    # max_len = np.max(sample_gt.shape)

    size = 2
    # draw gt
    for t in gts:
        sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0))
        sample_gt = cv2.putText(sample_gt, str(len(gts)), (0, 0), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 5)
    # draw predictions
    for p in pred:
        sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255))
        sample_pred = cv2.putText(sample_pred, str(len(pred)), (0, 0), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 5)
    
    name = target['image_id']
    sample_origin = cv2.putText(sample_origin, str(name), (0, 0), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 5)
    x.extend([pil_to_tensor(sample_origin), pil_to_tensor(sample_gt), pil_to_tensor(sample_pred)])
    # save the visualized images
    sample_gt = cv2.cvtColor(sample_gt, cv2.COLOR_RGB2BGR)
    sample_pred = cv2.cvtColor(sample_pred, cv2.COLOR_RGB2BGR)
    sample_origin = cv2.cvtColor(sample_origin, cv2.COLOR_RGB2BGR)
    if des is not None:
        cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                      des, len(gts), len(pred))),sample_gt)
        cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                      des, len(gts), len(pred))), sample_pred)
    else:
        cv2.imwrite(
            os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts), len(pred))),sample_gt)
        cv2.imwrite(
            os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts), len(pred))),sample_pred)


    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy()).astype(np.uint8)
    writer.add_image(exp_name + '_epoch_' + str(epoch + 1), x)

    # writer.add_image(exp_name + '_epoch_' + str(epoch + 1), sample_gt)

def vis(samples, targets, pred, vis_dir, exp_name, epoch, writer, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''

    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    x = []

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
      
      if idx > 1:  # show only one group
          break
      sample = restore_transform(samples[idx])
      sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
      sample_origin = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
      sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
      sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
      max_len = np.max(sample_gt.shape)
      size = 2
      # draw gt
      for t in gts[idx]:
          sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
          sample_gt = cv2.putText(sample_gt, str(len(gts[idx])), (0, 0), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 5)
      # draw predictions
      for p in pred[idx]:
          sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
          sample_pred = cv2.putText(sample_pred, str(len(pred[idx])), (0, 0), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 5)
      name = targets[idx]['image_id']
      sample_origin = cv2.putText(sample_origin, str(name), (0, 0), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 5)
      x.extend([pil_to_tensor(sample_origin), pil_to_tensor(sample_gt), pil_to_tensor(sample_pred)])
      # save the visualized images
      
      if des is not None:
          cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name),
                                                                                des, len(gts[idx]), len(pred[idx]))),
                      sample_gt)
          cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name),
                                                                                  des, len(gts[idx]),
                                                                                  len(pred[idx]))), sample_pred)
      else:
          cv2.imwrite(
              os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
              sample_gt)
          cv2.imwrite(
              os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
              sample_pred)


    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy()).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch + 1), x)


# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, exp_name,
                    vis_dir, writer,
                    max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for samples, targets in data_loader:  # each is a mini_batch
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #t为dict 将target中的每个样本的元素进行to(device) 同时将target变为list
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)  # *************

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # # ---------------------------------vis-----------------------
        # # if specified, save the visualized images
        # threshold = 0.5
        # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        # # outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][:2]
        # outputs_points = outputs['pred_points'][0] 
        # # points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        # points = outputs_points[outputs_scores > threshold]
        # if vis_dir is not None:
        #     vis_one(samples[0], targets[0], points, vis_dir, exp_name, epoch, writer)
        
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, exp_name, epoch, writer, vis_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    for samples, targets in data_loader:
        samples = samples.to(device)

        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]    # [num_archor,1]

        outputs_points = outputs['pred_points'][0]  # [num_archor, 2]

        gt_cnt = targets[0]['point'].shape[0]  # 'point' 长度即真实人数
                # 0.5 is used by default
        threshold = 0.5

                # calcu F1 measure Pre Rec

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()  # [num_pred,2]
        predict_cnt = int((outputs_scores > threshold).sum())
                # if specified, save the visualized images
        # if vis_dir is not None:
        #     vis_one(samples[0], targets[0], points, vis_dir, exp_name, epoch, writer)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse
            


# the inference routine
@torch.no_grad()
def evaluate_crowd(model,  data_loader, device, args, exp_name, epoch, writer, vis_dir=None):

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE MSE F1-Measure Pre Roc
    maes = []
    mses = []
    metrics = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'pred_cnt': AverageMeter()}

    for samples, targets in data_loader:
        samples = samples.to(device)
        #eval target是list of tuple  一个tuple包含一个点的信息
        outputs = model(samples)

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]    # [num_archor,1]
        outputs_points = outputs['pred_points'][0]  # [num_archor, 2]

        gt_cnt = targets[0]['point'].shape[0]  # 'point' 长度即真实人数
        pred_cnt = outputs_points.shape[0]
        # 0.5 is used by default
        threshold = 0.5

        # points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()  # [num_pred,2]
        predict_cnt = int((outputs_scores > threshold).sum())

        # ==========================calc TP FP FN
        bs, num_queries = outputs["pred_logits"].shape[:2]  # num_queries : 锚点数量

        # We flatten to compute the cost matrices in a batch 将前面两维展平，方便计算
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # 预测点为各个类别的概率 [num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # 预测点的坐标 [num_queries, 2]

        # Also concat the target labels and points
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_ids = targets[0]["labels"]  # 真实点的标签 [真实点数, 1]
        tgt_points = targets[0]["point"].cuda()  # 真实点的坐标 [真实点数,2]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]  # 分类损失矩阵 [batch_size * num_queries, 一个batch总的真实点数]

        # Compute the L2 cost between point
        cost_point = torch.cdist(out_points, tgt_points, p=2)  # 距离损失矩阵 [batch_size * num_queries, 一个batch总的真实点数]

        # Final cost matrix  总损失矩阵
        C = args.set_cost_point * cost_point + args.set_cost_class * cost_class
        C = C.cpu()  # C [ num_queries, 一个batch总的真实点数]

        # sizes = [len(v["point"]) for v in targets]  # 每个样本的真实点的数目 [B, num]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = linear_sum_assignment(C)

        # indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        indices = [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]

        # 真实点之间的距离
        match_martrix =  np.zeros((1, gt_cnt),dtype=bool)
        dist_gt = torch.cdist(tgt_points, tgt_points, p=2)  # [一个batch总的真实点数,一个batch总的真实点数]
        dist_gt_sort, dist_gt_index = torch.sort(dist_gt, descending=False, dim=-1)
        dist_mean = torch.mean(dist_gt_sort[:, :3], dim=-1)  # 只取前距离最近的三个的真值点的距离做平均
        delta = 0.5
        tp_cnt = 0
        for i,j in enumerate(indices[1]):
            print(cost_point[indices[0][i]][j])
            print(dist_mean[i] * delta)
            if cost_point[indices[0][i]][j]< dist_mean[i]*delta and cost_class[indices[0][i]][j]*(-1)>0.5: tp_cnt+=1
        metrics['tp'].update(tp_cnt)
        metrics['fp'].update(gt_cnt-tp_cnt)
        metrics['fn'].update(gt_cnt-tp_cnt)
        metrics['pred_cnt'].update(pred_cnt)

        # ==========================calc TP FP FN

        # if specified, save the visualized images
        # if vis_dir is not None:
        #     vis_one(samples[0], targets[0], points, vis_dir, exp_name, epoch, writer)

        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))
    # calc Pre Roc F1_Measure
    pre = metrics['tp'].sum / (metrics['tp'].sum + metrics['fp'].sum + 1e-20)
    roc = metrics['tp'].sum / (metrics['tp'].sum + metrics['fn'].sum + 1e-20)
    f1m = 2 * pre * roc / (pre + roc + 1e-20)
    nap = (metrics['pred_cnt'].sum - metrics['fp'].sum - metrics['fn'].sum) / metrics['pred_cnt'].sum

    return mae, mse, pre, roc, f1m, nap

# def loc_metrix(output, target, matcher):


