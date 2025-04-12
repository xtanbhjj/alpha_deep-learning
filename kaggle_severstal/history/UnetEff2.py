import os
import sys
sys.path.append("../")
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import torchvision
from torchvision import models
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import csv

def rle_decode(mask_rle: str = '', shape: tuple = (1600, 256)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  

def rle_encode(mask: np.ndarray):
    if mask is None or mask.size == 0:
        return ''

    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])  # 在两端添加 0，保证边界变化检测
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 找到值变化的位置

    if len(runs) % 2 != 0:  # 确保 runs 长度为偶数
        runs = runs[:-1]  # 移除最后一个元素（理论上不应该发生，但保险起见）

    runs[1::2] -= runs[::2]  # 计算 RLE 长度
    return ' '.join(map(str, runs))

def visualize_image_mask(image_id, df, image_dir='../data/severstal-steel-defect-detection/train_images'):
    image_path = os.path.join(image_dir, image_id)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(img)
    axes[0].set_title(f'Image: {image_id}')
    '''
    masks = [rle_decode(mask_rle) for mask_rle in df[df['ImageId'] == image_id]['EncodedPixels']]
    class_ids = df[df['ImageId'] == image_id]['ClassId'].values
    '''
    masks = []
    encoded_pixels_list = df[df['ImageId'] == image_id]['EncodedPixels'].tolist()
    class_ids = df[df['ImageId'] == image_id]['ClassId'].tolist()

    for i, rle in enumerate(encoded_pixels_list):
        if rle:
            decoded_mask = rle_decode(rle)
            masks.append(decoded_mask)

    for i in range(4):
        mask = masks[i] if i < len(masks) else np.zeros((256, 1600), dtype=np.uint8)
        axes[i+1].imshow(mask, cmap='gray')
        axes[i+1].set_title(f'Class {i+1}')
    plt.tight_layout()
    plt.show()

class SteelDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, train=True):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = self.df['ImageId'].unique()
        self.train = train

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if self.train:
            encoded_pixels_list = self.df[self.df['ImageId'] == image_id]['EncodedPixels'].tolist()
            class_ids = self.df[self.df['ImageId'] == image_id]['ClassId'].tolist()

            mask = np.zeros((256, 1600, 4), dtype=np.uint8)
            for i, rle in enumerate(encoded_pixels_list):
                if rle:
                    decoded_mask = rle_decode(rle)
                    mask[:, :, class_ids[i]-1] = decoded_mask 
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  
            mask = augmented['mask']   
            mask = mask.permute(2, 0, 1).float() 
            return image, mask
        else:
            augmented = self.transform(image=image)
            image = augmented['image']
            return image_id, image

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomCrop(height=256, width=256, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # ImageNet mean
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_valid_transforms():
    return A.Compose([  
        A.RandomCrop(height=256, width=256, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # ImageNet mean
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def dice_coeff(outputs, targets, smooth=1e-6, label=[0.7, 0.7, 0.6, 0.6]):
    dice = 0.0
    num_classes = outputs.size(1)
    if label:
        label = torch.tensor(label, dtype=torch.float).to(outputs.device)
        label = label.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    prediction = outputs > label
    
    
    for c in range(num_classes):
        mask = prediction[:, c, :, :]
        out = targets[:, c, :, :]
        output_flat = mask.contiguous().view(-1)  
        target_flat = out.contiguous().view(-1)
        
        intersection = (output_flat * target_flat).sum()
        union = output_flat.sum() + target_flat.sum()
        
        # 累加每个类别的Dice系数
        dice += (2.0 * intersection + smooth) / (union + smooth)
    
    return dice / num_classes

def train(model, optimizer, device, loss, train_loader, epoch, num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0
    generate = 8
    now = 0
    loss2 = torch.nn.BCEWithLogitsLoss()
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        #resized_outputs = F.interpolate(outputs, size=(256, 1600), mode='bilinear', align_corners=False)
        l = (loss(outputs, masks) + loss2(outputs, masks)) / generate
        l.backward()
        now += 1
        if now % generate == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += l.item() * images.size(0)
        train_dice += dice_coeff(torch.sigmoid(outputs), masks).item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    train_dice /= len(train_loader.dataset)
    return train_loss, train_dice

def validate(model, device, loss, val_loader, epoch, num_epochs):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_dice = 0.0
    loss2 = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            #resized_outputs = F.interpolate(outputs, size=(256, 1600), mode='bilinear', align_corners=False)
            l = loss(outputs, masks) + loss2(outputs, masks)
            val_loss += l.item() * images.size(0)
            val_dice += dice_coeff(torch.sigmoid(outputs), masks).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
    return val_loss, val_dice
def dice_loss(pred, target, smooth=1.):
    loss = 0
    for c in range(pred.shape[1]):
        intersection = (pred[:, c] * target[:, c]).sum()
        denominator = pred[:, c].sum() + target[:, c].sum()
        loss += 1 - (2. * intersection + smooth) / (denominator + smooth)
    return loss / pred.shape[1]

def focal_loss(inputs, targets, alpha, gamma, reduction='mean'):
    B, C, H, W = inputs.shape
    inputs = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = targets * inputs + (1 - targets) * (1 - inputs)
    focal_weight = (1 - pt) ** gamma

    if alpha is not None:
        alpha = torch.tensor(alpha, dtype=torch.float).to(inputs.device)
        alpha = alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # 形状 (1, C, 1, 1) 用于广播
        focal_weight = focal_weight * alpha

    loss = focal_weight * ce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def focal_dice_loss(pred, target, alpha=[0.5, 0.5, 0.25, 0.5], gamma=2):
    total_pixels = 497010 + 101587 + 1.7476646e+07 + 2370098
    freq_class_1 = 497010 / total_pixels
    freq_class_2 = 101587 / total_pixels
    freq_class_3 = 1.7476646e+07 / total_pixels
    freq_class_4 = 2370098 / total_pixels
    alpha = [
        (1 - freq_class_1) ** 0.5,
        (1 - freq_class_2) ** 0.5,
        (1 - freq_class_3) ** 0.5,
        (1 - freq_class_4) ** 0.5,
    ]
    weights = [
        1. / (freq_class_1 + 1e-6),
        1. / (freq_class_2 + 1e-6),
        1. / (freq_class_3 + 1e-6),
        1. / (freq_class_4 + 1e-6),
    ]
    sum_weights = sum(weights)
    alpha = [w / sum_weights for w in weights]

    focal = focal_loss(pred, target, alpha, gamma)
    dice = dice_loss(torch.softmax(pred, dim=1), target)
    return focal

def main():
    #hyperparams
    batch_size, num_epochs, lr = 4, 20, 5e-4
    ENCODER = "efficientnet-b3"
    ENCODER_WEIGHTS = 'imagenet'
    #dataloader
    df = pd.read_csv('../data/severstal-steel-defect-detection/trainer.csv')
    df['ClassId'] = df['ClassId'].apply(lambda x: int(x)) 
    df['EncodedPixels'].fillna('', inplace=True)
    fp = pd.read_csv('../data/severstal-steel-defect-detection/sample_submission.csv')

    train_df, val_df = train_test_split(df.drop_duplicates(subset=['ImageId']), test_size=0.2, random_state=42)

    train_dataset = SteelDataset(train_df, '../data/severstal-steel-defect-detection/train_images', transform=get_train_transforms())
    val_dataset = SteelDataset(val_df, '../data/severstal-steel-defect-detection/train_images', transform=get_valid_transforms())
    test_dataset = SteelDataset(fp, '../data/severstal-steel-defect-detection/test_images', transform=get_valid_transforms(), train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=4,
        activation=None
    )
    model.to(device)
    print(device)
    loss = focal_dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    best_valid_loss = 1e5

    for epoch in range(num_epochs):
        train_loss, train_dice = train(model, optimizer, device, loss, train_loader, epoch, num_epochs)
        val_loss, val_dice = validate(model, device, loss, val_loader, epoch, num_epochs)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f},  Train Dice Coeff: {train_dice:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f},  Validation Dice Coeff: {val_dice:.4f}")
        if best_valid_loss > val_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), './save/unetpp_effi.pth')

        scheduler.step(val_loss) 

if __name__ == "__main__":
    main()