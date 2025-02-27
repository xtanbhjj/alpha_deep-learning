
import os
import unittest
from unittest.mock import patch
from io import StringIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision
import torchinfo
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.plot import ImageUtils
from utils.accumulator import Accumulator
from utils.timer import Timer
import utils.dlf as dlf

# The transfer of Bounding box
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return torch.stack((cx, cy, w, h), dim=-1)

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack((x1, y1, x2, y2), dim=-1)

def bbox_to_rect(bbox, color):
    # bbox: The abbreviation for bounding box
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)

# 1. 生成锚框
def multibox_prior(data, sizes, ratios):
    #sizes -> 缩放比 ratio -> 宽高比
    #width = 
    # Generate a list of anchor boxes
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_num = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offser_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # Generate all center pts
    # 生成归一化后的每个中心点序列
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offser_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    
    # w' = w * s * sqrt(r) * (h / w)
    # h' = h * s / sqrt(r)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] / torch.sqrt(ratio_tensor[1:])))
    #拿到所有中心点 * boxes_num的偏移量
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2
    #拿到所有中心点 * boxes_num的中心点
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_num, dim=0)

    output = out_grid + anchor_manipulations
    #所有框的左上与右下坐标（归一化后的）
    return output.unsqueeze(0)

# 2. 利用交互比（IoU）给锚框打标签, 并且计算offset
def box_iou(box1, box2):
    box_area = lambda box: ((box[:, 2] - box[:, 0] * (box[:, 3] - box[:, 1])))
    
    area1 = box_area(box1) #(n_box1, )
    area2 = box_area(box2) #(n_box2, )
    
    # None就是在所处的位置添加一个纬度
    inter_upper = torch.max(box1[:, None, :2], box2[:, :2]) #(n_box1, n_box2, 2)
    inter_lower = torch.min(box1[:, None, 2:], box2[:, 2:])
    inter = (inter_upper - inter_lower).clamp(min=0)

    inter_areas = inter[:, :, 0] * inter[:, :, 1]
    union_areas = area1[:, None] + area2 - inter_areas

    return inter_areas / union_areas

def assign_anchor_to_boxes(ground_truth, anchors, device, iou_threshold=0.5):
    # Assign closest ground-truth bounding boxes to anchor boxes.
    num_box, num_real = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)

    # 先给每个都分配对于自己最有可能的real box
    box_real_map = torch.full((num_box, ), -1, dtype=torch.long, device=device)
    max_ious, idx = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = idx[max_ious >= iou_threshold]
    box_real_map[anc_i] = box_j
    col_discard = torch.full((num_box, ), -1)
    row_discard = torch.full((num_real, ), -1)

    # 根据算法按顺序做，相当于是刚好反了个顺序
    for _ in range(num_real):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_real).long()
        anc_idx = (max_idx / num_box).long()
        box_real_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    
    return box_real_map

def boxes_offset(anchors, assigned_bb, eps=1e-6):
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:]) / c_anc[:, 2:]
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    
    return offset

def multibox_target(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        label = labels[i, :, :]
        box_real_map = assign_anchor_to_boxes(label[:, 1:], anchors, device)
        box_mask = ((box_real_map >= 0).float().unsqueeze(-1).repeat(1, 4))
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        indice_true = torch.nonzero(box_real_map >= 0)
        bb_idx = box_real_map[indice_true]
        class_labels[indice_true] = label[bb_idx, 0].long() + 1 #锚框对应的真实框序号
        assigned_bb[indice_true] = label[bb_idx, 1:] #锚框对应的true框
        offset = boxes_offset(anchors, assigned_bb) * box_mask

        batch_offset.append(offset.reshape(-1))
        batch_mask.append(box_mask.reshape(-1))
        batch_class_labels.append(class_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)

    return bbox_offset, bbox_mask, class_labels