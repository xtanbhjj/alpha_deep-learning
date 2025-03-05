#! coding: utf-8


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


# Bounding box
def box_corner_to_center(boxes):
    # (left_upper_x, left_upper_y, right_lower_x, right_lower_y) -> (middle_x, middle_y, width, height)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def box_center_to_corner(boxes):
    # (center_x, center_y, width, height) -> (left_upper_x, left_lower_y, right_lower_x, right_lower_y)
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


dlf.DATA_HUB['banana-detection'] = (
    dlf.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')


def read_data_bananas(is_train=True):
    # Read images and labels from banana dataset
    data_dir = dlf.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')

    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val',
                             'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
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


def load_data_bananas(batch_size):
    # 对于 物体检测 来说，batch_size 与实际检测得到的物体数不同，有可能一张图片中
    # 会检测出多个物体，一般来说设置一个上限，即最多检测多少个物体，如果超出该上限，
    # 则多出来的物体就直接扔掉， 如果少了，则填一些0在里面，这样就可以让一个batch
    # 中的数据构成一个规整的tensor.
    # 这样的话 [batch_size, num_objects, feature, x1, y1, x2, y2]
    # Load banana dataset
    train_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=True),
        batch_size=batch_size, shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=False),
        batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, val_iter


def multibox_prior(data: torch.Tensor, sizes: list[float], ratios: list[float]):
    """
    The multibox_prior function generates prior (anchor) boxes through the following steps:
        1. Determine the normalized center coordinates for each pixel based on
           the dimensions of the input feature map.
        2. Compute the widths and heights of the prior boxes using the provided
           scales and aspect ratios.
        3. Construct the offsets for multiple prior boxes at each pixel and add
           them to the pixel centers to obtain the complete prior box coordinates.
        4. Finally, add a batch dimension before returning the result.
    :param data: (batch_size, channels, width, height).
    :param sizes: list of the scale (width / height).
    :param ratios: list of the aspect ratio.
    :return: anchor_boxes.shape = (batch_size, num_anchors, 4)
    """
    # Generate a list of anchor boxes for each pixel.
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # Generate all center pts
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def box_iou(boxes1, boxes2):
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


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    # Assign closest ground-truth bounding boxes to anchor boxes.
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j.
    jaccard = box_iou(anchors, ground_truth)

    # initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor.
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    # Assign ground-truth bounding boxes according to the threshold.
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)

    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    # Transform for anchor box offsets.
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset


# If an anchor box is not assigned a ground-truth bounding box, we just label the
# class of the anchor box as "background". Anchor boxes whose classes are
# background are often referred to as negative anchor boxes, and the rest are called
# positive anchor boxes.
# The multibox_target function to label classes and offsets for anchor boxes (the
# anchors' argument) using ground-truth bounding boxes (the labels' argument). This
# function sets the background class to zero and increments the integer index of
# a new class by one.
def multibox_target(anchors, labels):
    # Label anchor boxes using ground-truth bounding boxes.
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1).repeat(1, 4))
        # Initialize class labels and assigned bounding box coordinates with zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        # Label classes of anchor boxes using their assigned ground-truth bounding boxes.
        # If an anchor box is not assigned any, we label its class as background (the
        # value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return bbox_offset, bbox_mask, class_labels


# During prediction, we generate multiple anchor boxes for the image and predict classes
# and offsets for each of them. A predicted bounding box is thus obtained according
# to an anchor box with its predicted offset. Below we implement the offset_inverse
# function that takes in anchors and offset predictions as inputs and applies inverse
# offset transformations to return the predicted bounding box coordinates.
def offset_inverse(anchors, offset_preds):
    # Predict bounding boxes based on anchor boxes with predicted offsets.
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


# When there are many anchor boxes, many similar (with significant overlap) predicted
# bounding boxes can be potentially output for surrounding the same object. To simplify
# the output, we can merge similar predicted bounding boxes that belong to the same
# by using non-maximum suppression (NMS).
# Here is how non-maximum suppression works. For a predicted bounding box B, the
# object detection model calculates the predicted likelihood for each class. Denoting
# by p the largest predicted likelihood, the class corresponding to this probability
# is the predicted class for B. Specifically, we refere to p as the confidence (score)
# of the predicted bounding box B. On the same image, all the predicted non-background
# bounding boxes are sorted by confidence in descending order to generate a list L.
# Then we manipulate the sorted list L in the following steps:
#  1. Select the predicted bounding box B1 with the highest confidence from L as a basis
#     and remove all non-basis predicted bounding boxes whose IoU with B1 exceeds a
#     predefined threshold e from L. At this point, L keeps the predicted bounding box
#     with the highest confidence but drops others that are too similar to it. In a
#     nutshell, those with non-maximum confidence scores aare suppressed.
#  2. Select the predicted bounding box B2 with the second highest confidence from L
#     as another basis and remove all non-basis predicted bounding boxes whose IoU with
#     B2 exceeds e from L.
#  3. Repeat the above proces until all the predicted bounding boxes in L have been used
#     as a basis. At this time, the IoU of any pair of predicted bounding boxes in L is
#     below the threshold e, thus no pair is too similar with each other.
#  4. Output all the predicted bounding boxes in the list L.
# The following nms function sorts confidence scores in descedning order and returns
# their indices.
def nms(boxes, scores, iou_threshold):
    # Sort confidence scores of predicted bounding boxes.
    b = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept.
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
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []

    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # Find all non-keep indices, and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]

        # Here `pos_threshold` is a  threshold for positive (non-background)
        # predictions.
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
        self.net = nn.Sequential(
            nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1))

    def forward(self, x):
        return self.net(x)


# The design of the bounding box prediction layer is similar to that of the class
# prediction layer. the only difference lies in the number of outputs for each anchor
# box: here we need to predict four offsets rather than (q + 1) classes.
class BBoxPredictor(nn.Module):
    def __init__(self, num_inputs, num_anchors):
        super(BBoxPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1))

    def forward(self, x):
        return self.net(x)


# Note that the channel dimension holds the predictions for anchor boxes with the
# same center. We first move this dimension to the innermost. Since the batch size
# remains the same for different scales, we can transform the prediction output into
# a two-dimensional tensor with shape (batch_size, height, width, num_channels). Then
# we can concatenate such outputs at different scales along dimension 1.
# In this way, even though different preds have different sizes in channels, heights,
# and widths, we can still concatenate these two prediction outputs at two different
# scales for the same minibatch.
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


# The base network block is used to extract features from input images. For simplicity,
# we construct a small base network consisting of three down-sampling blocks that double
# the number of channels at each block.
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


# Single shot multibox detection is a multiscale object detection model. Via its
# base network and several multiscale feature map blocks, single-shot multibox
# detection generates a varying number of anchor boxes with different sizes,
# and detects varying-size objects by predicting classes and offsets of these
# anchors boxes (thus the bounding boxes).
# When training the single-shot multibox detection model, the loss function is
# calculated based on the predicted and labeled values of the anchor box classes
# and offsets.
class TinySSD(nn.Module):
    def __init__(self, num_classes, sizes, ratios, num_anchors, **kwargs):
        super(TinySSD, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.sizes = sizes
        self.ratios = ratios
        idx_to_in_channels = [64, 128, 128, 128, 128]

        for i in range(5):
            # Equivalent to the assignment statement 'self.blk_i = get_blk(i)
            setattr(self, f'blk_{i}', self.get_blk(i))
            setattr(self, f'cls_{i}', ClassPredictor(idx_to_in_channels[i],
                                                     num_anchors, num_classes))
            setattr(self, f'bbox_{i}', BBoxPredictor(idx_to_in_channels[i], num_anchors))

    def forward(self, x):
        default_classes = 5
        # For depress the PyCharm warning
        empty_tensor = torch.tensor([])
        anchors = [empty_tensor] * default_classes
        cls_preds = [empty_tensor] * default_classes
        bbox_preds = [empty_tensor] * default_classes

        for i in range(5):
            # Here getattr(self, 'blk_%d' % i) accesses self.blk_i
            x, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                x, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))

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
    return float((cls_preds.argmax(dim=-1).to(dtype=cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float(torch.abs((bbox_labels - bbox_preds) * bbox_masks).sum())


def train(net: nn.Module, train_iter, optimizer, calc_loss, num_epochs, device):

    net = net.to(device=device)

    timer = Timer()

    for epoch in range(num_epochs):
        # Sum of training accuracy, no. of examples in sum of training accuracy,
        # Sum of absolute error, no. of examples in sum of absolute error
        metric = Accumulator(4)
        net.train()

        for features, target in train_iter:
            timer.start()
            optimizer.zero_grad()
            x, y = features.to(device), target.to(device)
            # Generate multiscale anchor boxes and predict their classes and offsets
            anchors, cls_preds, bbox_preds = net(x)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, y)
            # Calculate the loss function using the predicted and labeled values
            # of the classes and offsets
            loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            loss.mean().backward()
            optimizer.step()

            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        print(f'iter: {epoch+1}, ', f'class error: {cls_err:.2e}, ', f'bbox mae: {bbox_mae:.2e}')

    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on ',
          f'{str(device)}')


def display(img, output, threshold):
    fig = ImageUtils.imshow(img)

    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        ImageUtils.show_boxes(fig.axes, bbox, ['%.2f' % score], ['w'])


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_path = os.path.join(str(Path(__file__).resolve().parent), 'catdog.png')
        self.model_path = os.path.join(str(Path(__file__).resolve().parent), 'tiny_ssd.pth')
        pass

    def test_simple(self):
        img = ImageUtils.open(self.img_path)
        fig = ImageUtils.imshow(img)

        dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

        boxes = torch.tensor((dog_bbox, cat_bbox))

        self.assertTrue(torch.all(box_center_to_corner(box_corner_to_center(boxes)) == boxes).item())

        fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
        fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))

        # Execute the plt.show() manually after the function.
        plt.show()

    def test_dataset_shapes(self):
        batch_size, edge_size = 32, 256
        train_iter, val_iter = load_data_bananas(batch_size)
        batch = next(iter(train_iter))
        print(batch[0].shape, batch[1].shape)

        self.assertEqual(torch.Size([batch_size, 3, edge_size, edge_size]), batch[0].shape)
        self.assertEqual(torch.Size([batch_size, 1, 5]), batch[1].shape)

        imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
        axes = ImageUtils.show_images(imgs, 2, 5, scale=2)
        for ax, label in zip(axes, batch[1][0:10]):
            ImageUtils.show_boxes(ax, [label[0][1:5] * edge_size], colors=['w'])

        # Add plt.show() at the end of the function, because I don't know how
        # to execute plt.show() before this function exiting automatically.
        plt.show()

    def test_anchor_boxes(self):
        img = ImageUtils.imread(self.img_path)
        h, w = img.shape[:2]
        print(h, w)
        self.assertEqual((728, 561), (w, h))

        x = torch.rand(size=(1, 3, h, w))
        y = multibox_prior(x, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
        print('y.shape: ', y.shape)
        self.assertEqual(y.shape, torch.Size([1, 2042040, 4]))

        boxes = y.reshape(h, w, 5, 4)
        print(boxes[250, 250, 0, :])

        bbox_scale = torch.tensor((w, h, w, h))
        fig = ImageUtils.imshow(img)
        ImageUtils.show_boxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
                              ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1',
                               's=0.75, r=2', 's=0.75, r=0.5'])

        # plt.show() manually
        plt.show()

    def test_annotate_classes_and_offsets(self):
        ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                     [1, 0.55, 0.2, 0.9, 0.88]])
        anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                                [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                                [0.57, 0.3, 0.92, 0.9]])
        img = ImageUtils.imread(self.img_path)
        h, w = img.shape[:2]
        bbox_scale = torch.tensor((w, h, w, h))

        fig = ImageUtils.imshow(img)
        ImageUtils.show_boxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], ['k'])
        ImageUtils.show_boxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

        # plt.show() manually
        plt.show()

        labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))

        print(labels[2])
        self.assertTrue(torch.equal(torch.tensor([[0, 1, 2, 0, 2]]), labels[2]))

    def test_predictions(self):
        anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                                [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
        offset_preds = torch.tensor([0] * anchors.numel())
        cls_probs = torch.tensor([[0] * 4,  # Predicted background likelihood
                                  [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood
                                  [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood

        img = ImageUtils.imread(self.img_path)
        fig = ImageUtils.imshow(img)

        h, w = img.shape[:2]
        bbox_scale = torch.tensor((w, h, w, h))
        ImageUtils.show_boxes(fig.axes, anchors * bbox_scale,
                              ['dog=0.9', 'dog=0.8', 'cat=0.7', 'cat=0.9'])

        # plt.show() manually
        plt.show()

        output = multibox_detection(cls_probs.unsqueeze(dim=0),
                                    offset_preds.unsqueeze(dim=0),
                                    anchors.unsqueeze(dim=0),
                                    nms_threshold=0.5)
        print(output)

        fig = ImageUtils.imshow(img)
        for i in output[0].detach().numpy():
            if i[0] == -1:
                continue
            label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
            ImageUtils.show_boxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)

        # plt.show() manually
        plt.show()

    def test_display_anchors(self):
        img = ImageUtils.imread(self.img_path)
        h, w = img.shape[:2]
        print(f'(H, W) = ({h}, {w})')
        self.assertEqual(561, h)
        self.assertEqual(728, w)

        def display_anchors(fmap_w, fmap_h, s):
            fmap = torch.zeros((1, 10, fmap_h, fmap_w))
            anchors = multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
            bbox_scale = torch.tensor((w, h, w, h))
            ImageUtils.show_boxes(ImageUtils.imshow(img).axes, anchors[0] * bbox_scale)

        display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
        plt.show()

        display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
        plt.show()

        display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
        plt.show()

    def test_multiscale_prediction(self):
        def forward(x, block):
            return block(x)

        # Let's assume that:
        #   * generate 5 anchor boxes
        #   * The number of object classes is 10.
        y1 = forward(torch.zeros((2, 8, 20, 20)), ClassPredictor(8, 5, 10))
        print(y1.shape)
        # Then the numbers of channels in the class prediction output are
        # 5 * (10 + 1) = 55
        self.assertEqual(torch.Size([2, 55, 20, 20]), y1.shape)

        # Let's assume that:
        #   * generate 3 anchor boxes
        #   * The number of object classes is 10.
        y2 = forward(torch.zeros((2, 16, 10, 10)), ClassPredictor(16, 3, 10))
        print(y2.shape)
        # Then the numbers of channels in the class prediction output are
        # 3 * (10 + 1) = 33
        self.assertEqual(torch.Size([2, 33, 10, 10]), y2.shape)
        
        res = concat_preds([y1, y2])
        print(res.shape)
        self.assertEqual(torch.Size([2, 25300]), res.shape)

        y = forward(torch.zeros((2, 3, 20, 20)), DownSamplingBlock(3, 10))
        print(y.shape)
        self.assertEqual(torch.Size([2, 10, 10, 10]), y.shape)

        y = forward(torch.zeros((2, 3, 256, 256)), BaseNetworkBlock())
        print(y.shape)
        self.assertEqual(torch.Size([2, 64, 32, 32]), y.shape)

    def test_tinyssd_basics(self):
        sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        ratios = [[1, 2, 0.5]] * 5
        num_anchors = len(sizes[0]) + len(ratios[0]) - 1
        net = TinySSD(num_classes=1, sizes=sizes, ratios=ratios, num_anchors=num_anchors)
        x = torch.zeros((32, 3, 256, 256))
        anchors, cls_preds, bbox_preds = net(x)

        print('output anchors: ', anchors.shape)
        print('output classes preds: ', cls_preds.shape)
        print('output bbox preds: ', bbox_preds.shape)
        self.assertEqual(torch.Size([1, 5444, 4]), anchors.shape)
        self.assertEqual(torch.Size([32, 5444, 2]), cls_preds.shape)
        self.assertEqual(torch.Size([32, 21776]), bbox_preds.shape)

        with patch('sys.stdout', new_callable=StringIO) as log:
            torchinfo.summary(model=net, input_size=(32, 3, 256, 256))
            act_output = log.getvalue().strip()

        expected_output = """
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
TinySSD                                       [1, 5444, 4]              --
├─BaseNetworkBlock: 1-1                       [32, 64, 32, 32]          --
│    └─Sequential: 2-1                        [32, 64, 32, 32]          --
│    │    └─DownSamplingBlock: 3-1            [32, 16, 128, 128]        2,832
│    │    └─DownSamplingBlock: 3-2            [32, 32, 64, 64]          14,016
│    │    └─DownSamplingBlock: 3-3            [32, 64, 32, 32]          55,680
├─ClassPredictor: 1-2                         [32, 8, 32, 32]           --
│    └─Sequential: 2-2                        [32, 8, 32, 32]           --
│    │    └─Conv2d: 3-4                       [32, 8, 32, 32]           4,616
├─BBoxPredictor: 1-3                          [32, 16, 32, 32]          --
│    └─Sequential: 2-3                        [32, 16, 32, 32]          --
│    │    └─Conv2d: 3-5                       [32, 16, 32, 32]          9,232
├─DownSamplingBlock: 1-4                      [32, 128, 16, 16]         --
│    └─Sequential: 2-4                        [32, 128, 16, 16]         --
│    │    └─Conv2d: 3-6                       [32, 128, 32, 32]         73,856
│    │    └─BatchNorm2d: 3-7                  [32, 128, 32, 32]         256
│    │    └─ReLU: 3-8                         [32, 128, 32, 32]         --
│    │    └─Conv2d: 3-9                       [32, 128, 32, 32]         147,584
│    │    └─BatchNorm2d: 3-10                 [32, 128, 32, 32]         256
│    │    └─ReLU: 3-11                        [32, 128, 32, 32]         --
│    │    └─MaxPool2d: 3-12                   [32, 128, 16, 16]         --
├─ClassPredictor: 1-5                         [32, 8, 16, 16]           --
│    └─Sequential: 2-5                        [32, 8, 16, 16]           --
│    │    └─Conv2d: 3-13                      [32, 8, 16, 16]           9,224
├─BBoxPredictor: 1-6                          [32, 16, 16, 16]          --
│    └─Sequential: 2-6                        [32, 16, 16, 16]          --
│    │    └─Conv2d: 3-14                      [32, 16, 16, 16]          18,448
├─DownSamplingBlock: 1-7                      [32, 128, 8, 8]           --
│    └─Sequential: 2-7                        [32, 128, 8, 8]           --
│    │    └─Conv2d: 3-15                      [32, 128, 16, 16]         147,584
│    │    └─BatchNorm2d: 3-16                 [32, 128, 16, 16]         256
│    │    └─ReLU: 3-17                        [32, 128, 16, 16]         --
│    │    └─Conv2d: 3-18                      [32, 128, 16, 16]         147,584
│    │    └─BatchNorm2d: 3-19                 [32, 128, 16, 16]         256
│    │    └─ReLU: 3-20                        [32, 128, 16, 16]         --
│    │    └─MaxPool2d: 3-21                   [32, 128, 8, 8]           --
├─ClassPredictor: 1-8                         [32, 8, 8, 8]             --
│    └─Sequential: 2-8                        [32, 8, 8, 8]             --
│    │    └─Conv2d: 3-22                      [32, 8, 8, 8]             9,224
├─BBoxPredictor: 1-9                          [32, 16, 8, 8]            --
│    └─Sequential: 2-9                        [32, 16, 8, 8]            --
│    │    └─Conv2d: 3-23                      [32, 16, 8, 8]            18,448
├─DownSamplingBlock: 1-10                     [32, 128, 4, 4]           --
│    └─Sequential: 2-10                       [32, 128, 4, 4]           --
│    │    └─Conv2d: 3-24                      [32, 128, 8, 8]           147,584
│    │    └─BatchNorm2d: 3-25                 [32, 128, 8, 8]           256
│    │    └─ReLU: 3-26                        [32, 128, 8, 8]           --
│    │    └─Conv2d: 3-27                      [32, 128, 8, 8]           147,584
│    │    └─BatchNorm2d: 3-28                 [32, 128, 8, 8]           256
│    │    └─ReLU: 3-29                        [32, 128, 8, 8]           --
│    │    └─MaxPool2d: 3-30                   [32, 128, 4, 4]           --
├─ClassPredictor: 1-11                        [32, 8, 4, 4]             --
│    └─Sequential: 2-11                       [32, 8, 4, 4]             --
│    │    └─Conv2d: 3-31                      [32, 8, 4, 4]             9,224
├─BBoxPredictor: 1-12                         [32, 16, 4, 4]            --
│    └─Sequential: 2-12                       [32, 16, 4, 4]            --
│    │    └─Conv2d: 3-32                      [32, 16, 4, 4]            18,448
├─AdaptiveMaxPool2d: 1-13                     [32, 128, 1, 1]           --
├─ClassPredictor: 1-14                        [32, 8, 1, 1]             --
│    └─Sequential: 2-13                       [32, 8, 1, 1]             --
│    │    └─Conv2d: 3-33                      [32, 8, 1, 1]             9,224
├─BBoxPredictor: 1-15                         [32, 16, 1, 1]            --
│    └─Sequential: 2-14                       [32, 16, 1, 1]            --
│    │    └─Conv2d: 3-34                      [32, 16, 1, 1]            18,448
===============================================================================================
Total params: 1,010,376
Trainable params: 1,010,376
Non-trainable params: 0
Total mult-adds (G): 31.38
===============================================================================================
Input size (MB): 25.17
Forward/backward pass size (MB): 2063.57
Params size (MB): 4.04
Estimated Total Size (MB): 2092.78
===============================================================================================
        """
        self.assertEqual(expected_output.strip(), act_output)

    def test_ssd_training(self):
        # hyperparameters
        batch_size, learning_rate, num_epochs = 32, 0.2, 20
        sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619],
                 [0.71, 0.79], [0.88, 0.961]]
        ratios = [[1, 2, 0.5]] * 5
        num_anchors = len(sizes[0]) + len(ratios[0]) - 1
        train_iter, _ = load_data_bananas(batch_size)

        device = dlf.devices()[0]
        net = TinySSD(num_classes=1, sizes=sizes, ratios=ratios, num_anchors=num_anchors)
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4)
        calc_loss = ObjectDetectionLossCalc()

        train(net, train_iter, optimizer, calc_loss, num_epochs, device)

        torch.save(net, self.model_path)

    def test_ssd_inference(self):
        device = dlf.devices()[0]
        if not os.path.exists(self.model_path):
            self.test_ssd_training()

        net = torch.load(self.model_path, weights_only=False).to(device=device)

        def predict(x):
            net.eval()
            anchors, cls_preds, bbox_preds = net(x.to(device))
            cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
            output_ = multibox_detection(cls_probs, bbox_preds, anchors)
            idx = [i for i, row in enumerate(output_[0]) if row[0] != -1]
            return output_[0, idx]

        x = torchvision.io.read_image(os.path.join(str(Path(__file__).resolve().parent),
                                                   'banana.jpg')).unsqueeze(0).float()
        img = x.squeeze(0).permute(1, 2, 0).long()
        output = predict(x)

        display(img, output.cpu(), threshold=0.9)

        plt.show()

        self.assertTrue(True)

    def test_smooth_loss_function(self):
        def smooth_l1(data, scalar):
            out = []
            for i in data:
                if abs(i) < 1 / (scalar ** 2):
                    out.append(((scalar * i) ** 2) / 2)
                else:
                    out.append(abs(i) - 0.5 / (scalar ** 2))
            return torch.tensor(out)

        sigmas = [10, 1, 0.5]
        lines = ['-', '--', '-.']
        x = torch.arange(-2, 2, 0.1)

        for l, s in zip(lines, sigmas):
            y = smooth_l1(x, scalar=s)
            plt.plot(x, y, l, label='sigma=%.1f' % s)
        plt.legend()
        plt.show()

        self.assertTrue(True)

    def test_focal_loss(self):
        # In the experiment, we use cross-entropy loss for class prediction: denoting
        # by p_j the predicted probability for the ground-truth class j, the cross-entropy
        # loss is (-log(p_j)). We can also use the focal loss:
        #            -alpha * (1 - p_j)^gamma * log(p_j)
        # As we can see, increasing gamma can effectively reduce the relative loss
        # for well-classified examples (e.g., p_j > 0.5) so the training can focus
        # more on those difficult examples that are mis-classified.

        def focal_loss(gamma, x):
            return -(1 - x) ** gamma * torch.log(x)

        lines = ['-', '--', '-.']
        x = torch.arange(0.01, 1, 0.01)

        for l, gamma in zip(lines, [0, 1, 5]):
            plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
        plt.legend()

        plt.show()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=True)
