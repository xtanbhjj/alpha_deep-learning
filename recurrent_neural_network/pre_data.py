import collections
import re
import os
import sys
import random
import torch
from torch.utils import data
sys.path.append('../')
from utils import dlf

dlf.DATA_HUB['time_machine'] = (dlf.DATA_URL + 'timemachine.txt',
                     '090b5e7e70c295757f55df93cb0a180b9691891a')

dlf.DATA_HUB['frg-eng'] = (dlf.DATA_URL + 'fra-eng.zip', 
                        '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_time_machine():
    """Load the time machine dataset into a list of text lines."""
    with open(dlf.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def read_data_nmt():
    """Load the dataset of the translation from Franch to English."""
    data_dir = dlf.download_extract('frg-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def tokenize(lines, token='word'):
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError('unknown token type: ' + token)

def preprocess_nmt(text):
    def no_space(char, pre_char):
        return char in set(',.!?') and pre_char != ' '
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text)]
    return ''.join(out)

def truncate_pad(line, num_steps, padding='<pad>'):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding] * (num_steps - len(line))

def tokenize_nmt(text, num_examples=None):
    sourse, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            sourse.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    
    return sourse, target
        

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        counts = collections.Counter(tokens)
        self._token_freqs = sorted(counts.items(), key=lambda x: x[1],
                                  reverse=True)

        #list
        self.idx_to_token = ['<unk>'] + reserved_tokens
        #dict
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """Count token frequencies."""
    # Flatten a nested list representing a corpus into a list of tokens
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_data(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, token='char')
    tokens = [token for line in tokens for token in line]
    vocab = Vocab(tokens)

    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def bulid_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])

    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def seq_data_iter_random(corpus, batch_size, num_steps):
    '''先考虑num_steps, 有很多个序列之后, 再随机取, 每次取barch_size'''
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]
    
    num_batches = num_subseqs // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    '''先考虑batch_size, 然后每次在各个维度取num_steps, 看能取多少次'''
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps

    #print(Xs.shape, num_batches, num_steps)
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_data(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

def load_data_nmt(batch_size, num_steps, num_examples=600):
    text = preprocess_nmt(read_data_nmt())
    sourse, target = tokenize_nmt(text, num_examples)
    sourse_data = [word for line in sourse for word in line]
    target_data = [word for line in target for word in line]
    src_vocab = Vocab(sourse_data, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target_data, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = bulid_array_nmt(sourse, src_vocab, num_steps)
    tgt_array, tgt_valid_len = bulid_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
    
def main():
    data_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8) 
    for X, X_valid_len, Y, Y_valid_len in data_iter:
        print('X:', X.type(torch.int32), '\n', X_valid_len, '\n',
              'Y:', Y.type(torch.int32), '\n', Y_valid_len, '\n')
        break

if __name__ == '__main__':
    main()


