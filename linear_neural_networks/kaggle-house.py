#! coding: utf-8
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import utils.dlf as dlf


dlf.DATA_HUB['kaggle_house_train'] = (
    dlf.DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

dlf.DATA_HUB['kaggle_house_test'] = (
    dlf.DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

def log_rmse(model, loss, features, labels):
    model.eval()
    clipped_preds = torch.clamp(model(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def infer(test_data, test_features, model):
    model.eval()
    preds = model(test_features).detach().numpy()

    test_data['SalePrice'] = pd.Series(preds.flatten())
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

def train(model, optimizer, loss, train_features, train_lables, test_features, test_lables, 
          num_epochs, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_lables), batch_size)

    model.train()
    for epoch in range(num_epochs):
        for x, y in train_iter:
            optimizer.zero_grad()
            l = loss(model(x), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(model, loss, train_features, train_lables))
        if test_lables is not None:
            test_ls.append(log_rmse(model, loss, test_features, test_lables))
    
    return train_ls, test_ls

def get_k_fold_data(k, i, x, y):
    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train, x_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[idx, :], y[idx]

        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat([x_train, x_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    
    return x_train, y_train, x_valid, y_valid

def k_fold(k, model, optimizer, loss, x_train, y_train, num_epochs, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    
    for i in range(k):
        k_data = get_k_fold_data(k, i, x_train, y_train)
        train_ls, valid_ls = train(model, optimizer, loss, *k_data, num_epochs, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print(f'The {i + 1}-th traing loss rmse{float(train_ls[-1]):f}, ',
              f'validation log rmse{float(valid_ls[-1]):f}')
        
    return train_l_sum / k, valid_l_sum / k

def main():

    # Get dataloader
    train_data = pd.read_csv(dlf.download('kaggle_house_train'))
    test_data = pd.read_csv(dlf.download('kaggle_house_test'))

    #Standardize the numeric_features
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    #One-hot encoding for the other data
    all_features = pd.get_dummies(all_features, dummy_na=True)

    n_train = train_data.shape[0]
    train_features = torch.from_numpy(all_features[:n_train].values.astype(float)).to(dtype=torch.float32)
    test_features = torch.from_numpy(all_features[n_train:].values.astype(float)).to(dtype=torch.float32)
    train_labels = torch.from_numpy(train_data.SalePrice.values.reshape(-1, 1)).to(dtype=torch.float32)

    loss = nn.MSELoss()
    in_features = train_features.shape[1]
    model = nn.Sequential(nn.Linear(in_features, 1))

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #k交叉
    train_l, valid_l = k_fold(k, model, optimizer, loss, train_features, train_labels, num_epochs, batch_size)
    print(f'{k}-fold validation: average traing log rmse: {float(train_l):f}, ',
              f'average testing log rmse: {float(valid_l):f}')
    
    train(model, optimizer, loss, train_features, train_labels, None, None, num_epochs, batch_size)
    infer(test_data, test_features, model)

if __name__ == "__main__":
    main()