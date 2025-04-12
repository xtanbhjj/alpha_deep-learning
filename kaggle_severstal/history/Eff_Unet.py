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
        if self.transform:
            augmented = self.transform(image=image) # 只对图像进行变换
            image = augmented['image']

        if self.train:
            masks = []
            encoded_pixels_list = self.df[self.df['ImageId'] == image_id]['EncodedPixels'].tolist()
            class_ids = self.df[self.df['ImageId'] == image_id]['ClassId'].tolist()

            mask = np.zeros((256, 1600, 4), dtype=np.uint8)
            for i, rle in enumerate(encoded_pixels_list):
                if rle:
                    decoded_mask = rle_decode(rle)
                    mask[:, :, class_ids[i]-1] = decoded_mask

            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
            image, mask = voc_rand_crop(image, mask, 256, 256) 
            return image, mask
        else:
            return image_id, image

def get_train_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'image'}) 

def get_valid_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'image'}) 

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

class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smooth_dice=1.0, dice_weight=0.25, focal_weight=0.75):
        super().__init__()
        
        self.smooth_dice = smooth_dice
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        pos_weight = torch.tensor([2.0, 2.0, 1.0, 1.5], dtype=torch.float32)
        pos_weight = pos_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = smp.losses.DiceLoss(
            mode='multilabel', # Output has shape (B, C, H, W)
            from_logits=True,  # Expect raw logits asdevicedevice
            smooth=self.smooth_dice
        )

    def forward(self, pred_logits, target_masks):
        bce = self.bce_loss(pred_logits, target_masks) # Alternative
        dice = self.dice_loss(pred_logits, target_masks)
        combined = self.dice_weight * dice + self.focal_weight * bce

        return combined

def train(model, optimizer, device, loss, train_loader, epoch, num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_dice = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        #resized_outputs = F.interpolate(outputs, size=(256, 1600), mode='bilinear', align_corners=False)
        resized_outputs = outputs
        l = loss(resized_outputs, masks) # 使用 Focal Loss + Dice Loss
        l.backward()
        optimizer.step()
        train_loss += l.item() * images.size(0)
        train_dice += dice_coeff(torch.sigmoid(resized_outputs), masks).item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    train_dice /= len(train_loader.dataset)
    return train_loss, train_dice

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
            l = loss(resized_outputs, masks) # 使用 Focal Loss + Dice Loss
            val_loss += l.item() * images.size(0)
            val_dice += dice_coeff(torch.sigmoid(resized_outputs), masks).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
    return val_loss, val_dice

def test(device, test_iter, model=None):
    if not model:
        model = None
        path_vgg = './save/unet_vgg_steel_defect1.pth'
        checkpoint = torch.load(path_vgg, map_location=device)
        state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)

    model.eval()
    output_csv_path = '../data/severstal-steel-defect-detection/output.csv'
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageId', 'EncodedPixels', 'ClassId'])

        with torch.no_grad():
            for ids, images in test_iter:
                images = images.to(device)
                original_tensor = images
                original_width = original_tensor.shape[3]
                target_width = 256
                num_full_crops = original_width // target_width
                remainder_width = original_width % target_width

                for batch_idx, image_id in enumerate(ids):
                    single_image_tensor = original_tensor[batch_idx].unsqueeze(0) # 处理单张图片

                    predicted_masks = []
                    # 处理完整的 256 宽度片段
                    for i in range(num_full_crops):
                        start_width = i * target_width
                        end_width = (i + 1) * target_width
                        crop = single_image_tensor[:, :, :, start_width:end_width]
                        prediction = model(crop)
                        prediction = torch.softmax(prediction, dim=1)
                        prediction = torch.argmax(prediction, dim=1)
                        predicted_mask_crop = prediction.squeeze(0).cpu().numpy() # (5, 256, 256)
                        predicted_masks.append(predicted_mask_crop)

                    # 处理剩余部分
                    if remainder_width > 0:
                        # 获取前一个片段的后 192 个像素
                        prev_start_width = (num_full_crops - 1) * target_width + 64
                        combined_crop = single_image_tensor[:, :, :, prev_start_width:]
                        prediction = model(combined_crop)
                        prediction = torch.softmax(prediction, dim=1)
                        prediction = torch.argmax(prediction, dim=1)
                        predicted_mask_combined = prediction.squeeze(0).cpu().numpy()

                        # 只取最后 64 个像素的预测结果
                        predicted_masks.append(predicted_mask_combined[:, (target_width - remainder_width):])

                    # 拼接所有预测的 mask
                    final_predicted_mask = np.concatenate(predicted_masks, axis=1) # 注意这里是 axis=2，因为我们处理的是单张图片

                    for class_id in range(1, 5):
                        mask_to_encode = (final_predicted_mask == class_id)
                        encoded_mask = rle_encode(mask_to_encode)
                        writer.writerow([image_id, encoded_mask, class_id])

    print(f"CSV 文件已保存到: {output_csv_path}")

def main():
    #hyperparams
    batch_size, num_epochs, lr, wd = 12, 10, 0.001, 1e-3
    ENCODER = 'efficientnet-b3' # Choose EfficientNet variant
    ENCODER_WEIGHTS = 'imagenet'
    #dataloader
    df = pd.read_csv('../data/severstal-steel-defect-detection/train.csv')
    df['ClassId'] = df['ClassId'].apply(lambda x: int(x)) 
    df['EncodedPixels'].fillna('', inplace=True)
    fp = pd.read_csv('../data/severstal-steel-defect-detection/sample_submission.csv')

    train_df, val_df = train_test_split(df.drop_duplicates(subset=['ImageId']), test_size=0.15, random_state=42)

    train_dataset = SteelDataset(train_df, '../data/severstal-steel-defect-detection/train_images', transform=get_train_transforms())
    val_dataset = SteelDataset(val_df, '../data/severstal-steel-defect-detection/train_images', transform=get_valid_transforms())
    test_dataset = SteelDataset(fp, '../data/severstal-steel-defect-detection/test_images', transform=get_valid_transforms(), train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=4
    )
    '''
    model = smp.UNuet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=4
    )
    model = UNetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=4
    )
    '''
    model.to(device)
    print(device)
    loss = CombinedLoss().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=wd) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(num_epochs):
        train_loss, train_dice = train(model, optimizer, device, loss, train_loader, epoch, num_epochs)
        val_loss, val_dice = validate(model, device, loss, val_loader, epoch, num_epochs)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f},  Train Dice Coeff: {train_dice:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f},  Validation Dice Coeff: {val_dice:.4f}")
        scheduler.step(val_loss) 

    #count(train_loader)
    # test(device, test_loader, model)
    torch.save(model.state_dict(), './save/unet_vgg_steel_defect.pth')


if __name__ == "__main__":
    main()