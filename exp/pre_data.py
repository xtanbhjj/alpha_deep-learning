import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import os
import torchvision
import torch.nn as nn
import torch.utils.data as Data 
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor()  
])

class EyeDataset(Data.Dataset):
    def __init__(self, left_eye_paths, right_eye_paths, labels, transform=None, val_transform=None, is_train = True):
        self.left_eye_paths = [os.path.join('../data/exp_data/Training_Dataset', image_path) for image_path in left_eye_paths]
        self.right_eye_paths = [os.path.join('../data/exp_data/Training_Dataset', image_path) for image_path in right_eye_paths]
        self.labels = labels  # 图像标签（多标签二进制向量）
        self.transform = transform  # 预处理操作
        self.val_transform = val_transform
        self.is_train = is_train
    def __len__(self):
        return len(self.labels)  # 返回数据集中的样本数

    def __getitem__(self, index):
        # 根据索引加载左右眼图像并应用预处理
        left_img = Image.open(self.left_eye_paths[index]).convert('RGB')  # 加载左眼图像
        right_img = Image.open(self.right_eye_paths[index]).convert('RGB')  # 加载右眼图像
        label = torch.tensor(self.labels[index], dtype=torch.float32)  # 将标签转换为张量
        # 判断是否为训练集，应用不同的预处理操作
        if self.is_train:
            if self.transform:
                left_img = self.transform(left_img)  # 应用预处理到左眼图像
                right_img = self.transform(right_img)  # 应用预处理到右眼图像
        else:
            if self.val_transform:
                left_img = self.val_transform(left_img)
                right_img = self.val_transform(right_img)
        return left_img, right_img, label  # 返回处理后的图像和标签

class DiagnosticDataset(Data.Dataset):
    def __init__(self, feature_vectors, labels):
        self.features = torch.tensor(np.vstack(feature_vectors), dtype=torch.float32)  # 转换为张量
        self.labels = torch.tensor(labels, dtype=torch.long)  # 目标标签

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
class pictokeysDataset(Data.Dataset):
    def __init__(self, eye_paths, keys):
        self.eye_paths = [os.path.join('../data/exp_data/Training_Dataset', image_path) for image_path in eye_paths]
        self.keys = keys
        self.transform = transform
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        img = Image.open(self.eye_paths[index]).convert('RGB')
        img = self.transform(img)
        return img, self.keys[index] 

def Get_data_conv():
    df = pd.read_csv('../data/exp_data/Traning_Dataset.csv')

    df["Left-Diagnostic Keywords"] = df["Left-Diagnostic Keywords"].astype(str).apply(lambda x: x.split("，") if x.strip() else [])
    df["Right-Diagnostic Keywords"] = df["Right-Diagnostic Keywords"].astype(str).apply(lambda x: x.split("，") if x.strip() else [])

    all_keywords = set([kw for sublist in df["Left-Diagnostic Keywords"] for kw in sublist]) | \
                set([kw for sublist in df["Right-Diagnostic Keywords"] for kw in sublist])
    n = len(all_keywords)
    keyword_to_index = {keyword: idx for idx, keyword in enumerate(all_keywords)}
    def encode_binary_vector(keyword_list):
        vector = np.zeros(n, dtype=int)  # 创建长度为 n 的零向量
        for kw in keyword_list:
            if kw in keyword_to_index:
                vector[keyword_to_index[kw]] = 1  # 关键词对应位置设为 1
        return vector

    df["Left-Diagnostic Encoded"] = df["Left-Diagnostic Keywords"].apply(encode_binary_vector)
    df["Right-Diagnostic Encoded"] = df["Right-Diagnostic Keywords"].apply(encode_binary_vector)
    # 创建第一个 DataFrame (左眼信息)
    df_left = df[['Left-Fundus', 'Left-Diagnostic Encoded']].copy()
    df_left.rename(columns={'Left-Fundus': 'Fundus', 'Left-Diagnostic Encoded': 'Diagnostic Keywords'}, inplace=True)

    # 创建第二个 DataFrame (右眼信息)
    df_right = df[['Right-Fundus', 'Right-Diagnostic Encoded']].copy()
    df_right.rename(columns={'Right-Fundus': 'Fundus', 'Right-Diagnostic Encoded': 'Diagnostic Keywords'}, inplace=True)

    df_combined = pd.concat([df_left, df_right], ignore_index=True)
    return pictokeysDataset(df_combined['Fundus'], df_combined['Diagnostic Keywords'])
 

def Get_data_total():
    df = pd.read_csv('../data/exp_data/Traning_Dataset.csv')
    left_eye_paths = df["Left-Fundus"].tolist()
    right_eye_paths = df["Right-Fundus"].tolist()
    binary_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    df['label'] = df[binary_columns].apply(lambda row: int(''.join(row.astype(int).astype(str)), 2),axis=1)
    return EyeDataset(left_eye_paths, right_eye_paths, df['label'].tolist(), transform=transform, val_transform=val_transform) 

def Get_data_linear():
    df = pd.read_csv('../data/exp_data/Traning_Dataset.csv')

    df["Left-Diagnostic Keywords"] = df["Left-Diagnostic Keywords"].astype(str).apply(lambda x: x.split("，") if x.strip() else [])
    df["Right-Diagnostic Keywords"] = df["Right-Diagnostic Keywords"].astype(str).apply(lambda x: x.split("，") if x.strip() else [])

    all_keywords = set([kw for sublist in df["Left-Diagnostic Keywords"] for kw in sublist]) | \
                set([kw for sublist in df["Right-Diagnostic Keywords"] for kw in sublist])
    n = len(all_keywords)
    keyword_to_index = {keyword: idx for idx, keyword in enumerate(all_keywords)}
    def encode_binary_vector(keyword_list):
        vector = np.zeros(n, dtype=int)  # 创建长度为 n 的零向量
        for kw in keyword_list:
            if kw in keyword_to_index:
                vector[keyword_to_index[kw]] = 1  # 关键词对应位置设为 1
        return vector

    df["Left-Diagnostic Encoded"] = df["Left-Diagnostic Keywords"].apply(encode_binary_vector)
    df["Right-Diagnostic Encoded"] = df["Right-Diagnostic Keywords"].apply(encode_binary_vector)
    df["Feature Vector"] = df.apply(lambda row: np.concatenate((row["Left-Diagnostic Encoded"], row["Right-Diagnostic Encoded"])), axis=1) 
    df['label'] = df[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.tolist()
    return DiagnosticDataset(df['Feature Vector'], df['label'])

def main():
    dataset = Get_data_linear()
    for X, y in dataset:
        print(X.shape, y.shape)

if __name__ == "__main__":
    main()
