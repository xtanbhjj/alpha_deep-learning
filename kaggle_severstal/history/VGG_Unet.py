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

            mask = np.zeros((256, 1600, 5), dtype=np.uint8)
            for i, rle in enumerate(encoded_pixels_list):
                if rle:
                    decoded_mask = rle_decode(rle)
                    mask[:, :, class_ids[i]] = decoded_mask
            background_mask = np.logical_not(np.any(mask[:, :, 1:], axis=-1))
            mask[:, :, 0] = background_mask.astype(np.uint8)

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

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act2(x)

class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)

class UNetWithVGGEncoder(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        # 加载预训练的VGG16模型，包含特征提取器
        vgg16 = models.vgg16(pretrained=True).features

        # 提取VGG16的卷积层作为编码器
        self.enc1 = nn.Sequential(*vgg16[:4]) # Output: (64, H/1, W/1)
        self.enc2 = nn.Sequential(*vgg16[4:9]) # Output: (128, H/2, W/2)
        self.enc3 = nn.Sequential(*vgg16[9:16]) # Output: (256, H/4, W/4)
        self.enc4 = nn.Sequential(*vgg16[16:23]) # Output: (512, H/8, W/8)
        self.enc5 = nn.Sequential(*vgg16[23:30]) # Output: (512, H/16, W/16)

        # 冻结编码器部分的权重
        for param in self.enc1.parameters():
            param.requires_grad = False
        for param in self.enc2.parameters():
            param.requires_grad = False
        '''
        for param in self.enc3.parameters():
            param.requires_grad = False
        for param in self.enc4.parameters():
            param.requires_grad = False
        for param in self.enc5.parameters():
            param.requires_grad = False
        '''

        # 中间层
        self.middle_conv = DoubleConvolution(512, 1024)

        # 解码器部分
        self.up_sample1 = UpSample(1024, 512)
        self.up_conv1 = DoubleConvolution(512 + 512, 512)

        self.up_sample2 = UpSample(512, 256)
        self.up_conv2 = DoubleConvolution(256 + 256, 256)  # 修改这里

        self.up_sample3 = UpSample(256, 128)
        self.up_conv3 = DoubleConvolution(128 + 128, 128)  # 修改这里

        self.up_sample4 = UpSample(128, 64)
        self.up_conv4 = DoubleConvolution(64 + 64, 64)    # 修改这里

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # 中间层
        middle = self.middle_conv(enc5)

        # 解码器
        dec1 = self.up_sample1(middle)
        dec1 = self.up_conv1(torch.cat([dec1, enc4], dim=1))

        dec2 = self.up_sample2(dec1)
        dec2 = self.up_conv2(torch.cat([dec2, enc3], dim=1))

        dec3 = self.up_sample3(dec2)
        dec3 = self.up_conv3(torch.cat([dec3, enc2], dim=1))

        dec4 = self.up_sample4(dec3)
        dec4 = self.up_conv4(torch.cat([dec4, enc1], dim=1))

        # 最终卷积
        output = self.final_conv(dec4)
        return output
def dice_coeff(outputs, targets, smooth=1e-6):
    
    prediction = torch.argmax(outputs, dim=1)
    dice = 0.0
    num_classes = outputs.size(1)
    
    for c in range(1, num_classes):
        mask = torch.zeros_like(prediction, dtype=torch.float32)
        out = targets[:, c].unsqueeze(1)
        mask[prediction == c] = 1.0
        output_flat = mask.contiguous().view(-1)  
        target_flat = out.contiguous().view(-1)
        
        intersection = (output_flat * target_flat).sum()
        union = output_flat.sum() + target_flat.sum()
        
        # 累加每个类别的Dice系数
        dice += (2.0 * intersection + smooth) / (union + smooth)
    
    # 返回平均Dice系数
    return dice / num_classes

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

def focal_dice_loss(pred, target, alpha=[0, 0.5, 0.5, 0.25, 0.5], gamma=2):
    total_pixels = 3.2899e+08 + 497010 + 101587 + 1.7476646e+07 + 2370098
    freq_class_0 = 3.2899e+08 / total_pixels
    freq_class_1 = 497010 / total_pixels
    freq_class_2 = 101587 / total_pixels
    freq_class_3 = 1.7476646e+07 / total_pixels
    freq_class_4 = 2370098 / total_pixels
    alpha = [
        (1 - freq_class_0) ** 0.5,  
        (1 - freq_class_1) ** 0.5,
        (1 - freq_class_2) ** 0.5,
        (1 - freq_class_3) ** 0.5,
        (1 - freq_class_4) ** 0.5,
    ]
    weights = [
        1. / (freq_class_0 + 1e-6),  
        1. / (freq_class_1 + 1e-6),
        1. / (freq_class_2 + 1e-6),
        1. / (freq_class_3 + 1e-6),
        1. / (freq_class_4 + 1e-6),
    ]
    sum_weights = sum(weights)
    alpha = [w / sum_weights for w in weights]

    focal = focal_loss(pred, target, alpha, gamma)
    dice = dice_loss(torch.softmax(pred, dim=1), target)
    return dice + focal

def pixel_accuracy(outputs, masks): 
    with torch.no_grad():
        batch_size, num_classes, height, width = outputs.shape
        outputs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1)
        true_masks = torch.argmax(masks, dim=1)

        correct = (predicted == true_masks).float()
        accuracy = correct.sum() / (batch_size * height * width)
    return accuracy

def train(model, optimizer, device, train_loader, epoch, num_epochs):
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
        loss = focal_dice_loss(resized_outputs, masks) # 使用 Focal Loss + Dice Loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        train_acc += pixel_accuracy(resized_outputs, masks) * images.size(0)
        train_dice += dice_coeff(torch.softmax(resized_outputs, dim=1), masks).item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    train_dice /= len(train_loader.dataset)
    return train_loss, train_acc, train_dice

def validate(model, device, val_loader, epoch, num_epochs):
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
            loss = focal_dice_loss(resized_outputs, masks) # 使用 Focal Loss + Dice Loss
            val_loss += loss.item() * images.size(0)
            val_acc += pixel_accuracy(resized_outputs, masks) * images.size(0)
            val_dice += dice_coeff(torch.softmax(resized_outputs, dim=1), masks).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
    return val_loss, val_acc, val_dice

def test(device, test_iter, model=None):
    if not model:
        model = UNetWithVGGEncoder(5)
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

def count(train_loader):
    num = [0] * 5
    for i, j in train_loader:
        for k in range(5):
            num[k] += j[:,k,:,:].sum()
    
    for i in range(5):
        print(num[i])
            

def main():
    #hyperparams
    batch_size, num_epochs, lr, wd = 8, 10, 0.002, 1e-3
    #dataloader
    df = pd.read_csv('../data/severstal-steel-defect-detection/train.csv')
    df['ClassId'] = df['ClassId'].apply(lambda x: int(x)) 
    df['EncodedPixels'].fillna('', inplace=True)
    fp = pd.read_csv('../data/severstal-steel-defect-detection/sample_submission.csv')

    '''
    sample_image_id = df['ImageId'].unique()[5]
    visualize_image_mask(sample_image_id, df)
    '''

    train_df, val_df = train_test_split(df.drop_duplicates(subset=['ImageId']), test_size=0.2, random_state=42)

    train_dataset = SteelDataset(train_df, '../data/severstal-steel-defect-detection/train_images', transform=get_train_transforms())
    val_dataset = SteelDataset(val_df, '../data/severstal-steel-defect-detection/train_images', transform=get_valid_transforms())
    test_dataset = SteelDataset(fp, '../data/severstal-steel-defect-detection/test_images', transform=get_valid_transforms(), train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetWithVGGEncoder(5)
    path_vgg = './save/unet_vgg_steel_defect1.pth'
    checkpoint = torch.load(path_vgg, map_location=device)
    state_dict = checkpoint
    model.load_state_dict(state_dict) 
    model.to(device)
    print(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True) 

    for epoch in range(num_epochs):
        train_loss, train_acc, train_dice = train(model, optimizer, device, train_loader, epoch, num_epochs)
        val_loss, val_acc, val_dice = validate(model, device, val_loader, epoch, num_epochs)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Pixel Accuracy: {train_acc:.4f}, Train Dice Coeff: {train_dice:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Pixel Accuracy: {val_acc:.4f}, Validation Dice Coeff: {val_dice:.4f}")
        scheduler.step(val_loss) 

    #count(train_loader)
    # test(device, test_loader, model)
    torch.save(model.state_dict(), './save/unet_vgg_steel_defect.pth')


if __name__ == "__main__":
    main()