
import torch
import torch.nn as nn
import sys
import math
sys.path.append('../')
from recurrent_neural_network.pre_data import *
#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_length=None):
        '''
        queries: (batchsize * n * query_size)
        keys: (batchsize * m * key_size)
        values: (batchsize * m * value_size)
        '''
        queries, keys = self.W_q(queries), self.W_k(keys)
        # (batchsize * n * num_hidden) (batchsize * m * num_hidden)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # (batchsize * n * m * num_hidden)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        # (batchsize * n * m)
        #print(scores)
        self.attention_weights = masked_softmax(scores, valid_length)
        #print(self.attention_weights)

        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_length):
        '''
        queries: (batchsize * n * query_size)
        keys: (batchsize * m * key_size)
        values: (batchsize * m * value_size)
        query_size == key_size == value_size
        '''
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_length)
        return torch.bmm(self.dropout(self.attention_weights), values)

class positional_encoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(positional_encoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    #print(X.shape)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class multihead_attention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(multihead_attention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_len):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # 因为attention是作用在每个batch上的，所以我们只需要吧num_heads维度放在batch上就好了
        if valid_len is not None:
            valid_len = torch.repeat_interleave(valid_len, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_len)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class positionwiseFFN(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(positionwiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

def main():
    queries, keys = torch.ones((2, 1, 20)), torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_length = torch.tensor([1, 1])
    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_length))
    #print(masked_softmax(torch.rand(2,2,4), torch.tensor([2, 3])))
    print(attention.attention_weights)


if __name__ == '__main__':
    main()