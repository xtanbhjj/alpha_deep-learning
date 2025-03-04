
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
from utils.accumulator import Accumulator
from utils.plot import ImageUtils
from utils.timer import Timer
import utils.dlf as dlf

dlf.DATA_HUB['banana-detection'] = (
    dlf.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

def read_data_bananas(is_train=True):
    data_dir = '../data/banana-detection'
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')

    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val',
                             'images', f'{img_name}')))
        targets.append(list(target))
    
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    # Define a custom bananas dataset. We should override __getitem__ and __len__ methods.
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read', str(len(self.features)),
              (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)

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
def box_iou(boxes1, boxes2):
    print("box1 shape:", boxes1.shape)  
    print("box2 shape:", boxes2.shape)
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    # The shapes of the boxes1, boxes2, areas1 and areas2.
    # (the number of boxes1, 4)
    # (the number of boxes2, 4)
    # (the number of boxes1, )
    # (the number of boxes2, )
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # The shapes of the inter_upperlefts, inter_lowerrights, inters
    # (the number of boxes1, the number of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # The shape of the inter_areasandunion_areas is (the number of boxes1, the number of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
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

def offset_inverse(authors, offset_preds):
    anc = box_corner_to_center(authors)
    pred_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_box = torch.cat((pred_xy, pred_wh), dim=1)
    predicted_box = box_center_to_corner(pred_box)

    return predicted_box

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

def nms(boxes, scores, iou_threshold):
    b = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while b.numel() > 0:
        i = b[0]
        keep.append(i)

        if b.numel() == 1:
            break

        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[b[1:], :].reshape(-1, 4)).reshape(-1)
        indices = torch.nonzero(iou <= iou_threshold).reshape(-1)
        b = b[indices + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
        
    return torch.stack(out)

class ClassPredictor(nn.Module):
    def __init__(self, num_inputs, num_anchors, num_classes):
        super(ClassPredictor, self).__init__()
        self.net = nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)
    
class BBoxPredictor(nn.Module):
    def __init__(self, num_inputs, num_anchors):
        super(BBoxPredictor, self).__init__()
        self.net = nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)
    
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        blk = []
        for _ in range(2):
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            blk.append(nn.BatchNorm2d(out_channels))
            blk.append(nn.ReLU())
            in_channels = out_channels
        blk.append(nn.MaxPool2d(2))

        self.net = nn.Sequential(*blk)
    
    def forward(self, x):
        return self.net(x)
    
class BaseNetworkBlock(nn.Module):
    def __init__(self):
        super(BaseNetworkBlock, self).__init__()
        blk = []
        num_filters = [3, 16, 32, 64]
        for i in range(len(num_filters) - 1):
            blk.append(DownSamplingBlock(num_filters[i], num_filters[i + 1]))
        
        self.net = nn.Sequential(*blk)
    
    def forward(self, x):
        return self.net(x)
    
class TinySSD(nn.Module):
    def __init__(self, num_classes, sizes, ratios, num_anchors, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        # num_anchors指的是对于每个像素点生成的锚框的个数

        self.num_classes = num_classes
        self.sizes = sizes
        self.ratios = ratios
        idx_to_in_channel = [64, 128, 128, 128, 128]

        for i in range(5):
            setattr(self, f'blk_{i}', self.get_blk(i))
            setattr(self, f'cls_{i}', ClassPredictor(idx_to_in_channel[i], num_anchors, num_classes))
            setattr(self, f'offset_{i}', BBoxPredictor(idx_to_in_channel[i], num_anchors))

    def forward(self, x):
        default_classes = 5
        empty_tensor = torch.tensor([])
        anchors = [empty_tensor] * default_classes
        cls_preds = [empty_tensor] * default_classes
        bbox_preds = [empty_tensor] * default_classes

        for i in range(5):
            x, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                x, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'offset_{i}'))
            
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
    
    @staticmethod
    def get_blk(i):
        blks = [
            BaseNetworkBlock(),
            DownSamplingBlock(64, 128),
            DownSamplingBlock(128, 128),
            DownSamplingBlock(128, 128),
            nn.AdaptiveMaxPool2d((1, 1)),
        ]
        return blks[i]
    
    @staticmethod
    def blk_forward(x, blk, size, ratio, cls_predictor, bbox_predictor):
        y = blk(x)
        anchors = multibox_prior(y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(y)
        bbox_preds = bbox_predictor(y)
        return y, anchors, cls_preds, bbox_preds
    
class ObjectDetectionLossCalc:
    def __init__(self):
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.bbox_loss = nn.L1Loss(reduction='none')
    
    def __call__(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
        cls = self.cls_loss(cls_preds.reshape(-1, num_classes),
                            cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = self.bbox_loss(bbox_preds * bbox_masks,
                              bbox_labels * bbox_masks).mean(dim=1)
        return cls + bbox
    
def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

def train(net, optimizer, loss, device, train_iter):
    net.train()
    metric = Accumulator(4)
    for feature, target in train_iter:
        optimizer.zero_grad()
        x, y = feature.to(device), target.to(device)
        anchors, cls_preds, bbox_preds = net(x)
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, y)
        l = loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        optimizer.step()

        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
    
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    return cls_err, bbox_mae

def display(img, output, threshold):
    fig = ImageUtils.imshow(img)

    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        ImageUtils.show_boxes(fig.axes, bbox, ['%.2f' % score], ['w'])
def inference(model, device, val_iter):
    model.eval()

    x = torchvision.io.read_image(os.path.join(str(Path(__file__).resolve().parent),
                                                   'banana.jpg')).unsqueeze(0).float()
    img = x.squeeze(0).permute(1, 2, 0).long()
    anchors, cls_preds, bbox_preds = model(x.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_preds, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    output = output[0, idx]

    display(img, output.cpu(), threshold=0.9)
    plt.show()

def main():
     # hyperparameters
    batch_size, learning_rate, num_epochs = 32, 0.2, 20
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619],
            [0.71, 0.79], [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1

    # dataloader
    train_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=True),
        batch_size=batch_size, shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=False),
        batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = TinySSD(num_classes=1, sizes=sizes, ratios=ratios, num_anchors=num_anchors)
    device = dlf.devices('cpu')[0]
    print(device)
    model = model.to(device)
    '''
    devices = [0, 1, 2, 3]
    model = nn.DataParallel(model, device_ids=devices)
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    loss = ObjectDetectionLossCalc()

    for epoch in range(num_epochs):
        cl, bl = train(model, optimizer, loss, device, train_iter)
        print(f'iter: {epoch+1}, ', f'class error: {cl:.2e}, ', f'bbox mae: {bl:.2e}')
    
    inference(model, device, val_iter)
    

if __name__ == '__main__':
    main()