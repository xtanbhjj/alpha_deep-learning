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

def voc_rand_crop(feature, label, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

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
    def __init__(self, df, image_dir, image_ids, transform=None, train=True):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = image_ids
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
        A.RandomCrop(height=256, width=512, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # ImageNet mean
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_valid_transforms():
    return A.Compose([  
        A.RandomCrop(height=256, width=512, p=1.0),
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

def pixel_accuracy(outputs, masks): 
    with torch.no_grad():
        batch_size, num_classes, height, width = outputs.shape
        prediction = outputs > 0.55
        correct = (prediction == masks).float()
        accuracy = correct.sum() / (batch_size * num_classes * height * width) 
    return accuracy

def train(model, optimizer, device, loss, train_loader, epoch, num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0
    generate = 6
    now = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        #resized_outputs = F.interpolate(outputs, size=(256, 1600), mode='bilinear', align_corners=False)
        resized_outputs = outputs
        l = loss(resized_outputs, masks) / generate
        l.backward()
        now += 1
        if now % generate == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += l.item() * images.size(0)
        train_dice += dice_coeff(torch.sigmoid(resized_outputs), masks).item() * images.size(0)
        train_acc += pixel_accuracy(torch.sigmoid(resized_outputs), masks).item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    train_dice /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_dice, train_acc

def validate(model, device, loss, val_loader, epoch, num_epochs):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            #resized_outputs = F.interpolate(outputs, size=(256, 1600), mode='bilinear', align_corners=False)
            resized_outputs = outputs
            l = loss(resized_outputs, masks) 
            val_loss += l.item() * images.size(0)
            val_dice += dice_coeff(torch.sigmoid(resized_outputs), masks).item() * images.size(0)
            val_acc += pixel_accuracy(torch.sigmoid(resized_outputs), masks).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
    return val_loss, val_dice, val_acc

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.75, dice_weight=0.25, bce_pos_weight=None, device='cpu'):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.device = device # Store device

        # Use pos_weight if provided
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight.to(self.device) if bce_pos_weight is not None else None)

        # Ensure DiceLoss is also on the correct device implicitly by operating on tensors
        self.dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)

    def forward(self, outputs, targets):
        bce_l = self.bce_loss(outputs, targets)
        dice_l = self.dice_loss(outputs, targets)
        # Combine losses
        combined_loss = (self.bce_weight * bce_l) + (self.dice_weight * dice_l)
        return combined_loss

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

    image_ids = df['ImageId'].unique()
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    train_dataset = SteelDataset(df, '../data/severstal-steel-defect-detection/train_images', train_ids, transform=get_train_transforms(), train=True)
    val_dataset = SteelDataset(df, '../data/severstal-steel-defect-detection/train_images', val_ids, transform=get_valid_transforms(), train=True)
    #test_dataset = SteelDataset(fp, '../data/severstal-steel-defect-detection/test_images', transform=get_valid_transforms(), train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=4,
        activation=None
    )
    path_vgg = '/A432/zyz/kaggle/kaggle_severstal/save/fpn2.pth'
    #fpn2 只有bce
    #fpn3 0.75 * bce + 0.25 * dice
    checkpoint = torch.load(path_vgg, map_location=device)
    state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    print(device)
    pos_weights = torch.tensor([2.0, 2.0, 1.0, 1.5]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    loss = CombinedLoss(bce_weight=0.75, dice_weight=0.25, bce_pos_weight=pos_weights, device=device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    best_valid_loss = 1e5

    for epoch in range(num_epochs):
        train_loss, train_dice, train_acc = train(model, optimizer, device, loss, train_loader, epoch, num_epochs)
        val_loss, val_dice, val_acc = validate(model, device, loss, val_loader, epoch, num_epochs)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f},  Train Dice Coeff: {train_dice:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f},  Validation Dice Coeff: {val_dice:.4f}, Validation Acc: {val_acc:.4f}")
        if best_valid_loss > val_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), './save/trans.pth')

        scheduler.step(val_loss) 

if __name__ == "__main__":
    main()