'''
one-hot -> embedding
encoder decoder
'''
import torch
from torch import nn
import sys
from pre_data import *
from utils.accumulator import Accumulator
from utils import dlf
sys.path.append('../')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        embedded = self.embedding(X)
        embedded = embedded.permute(1, 0, 2)
        output, state = self.rnn(embedded)
        return output, state

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        embedded = self.embedding(X).permute(1, 0, 2)
        # embedded = time batch embedding
        # state[-1] = batch embedding
        # print(embedded.shape)
        context = state[-1].repeat(embedded.shape[0], 1, 1)
        X = torch.cat((embedded, context), 2)
        print(X.shape)
        output, state = self.rnn(X, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout, **kwargs):
        super(Seq2Seq, self).__init__(**kwargs)
        self.encoder = Encoder(src_vocab_size, embed_size, num_hiddens, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout)
    
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        dec_output, dec_state = self.decoder(dec_X, dec_state)
        return dec_output, dec_state

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def train(model, optimizer, loss, device, train_iter, tgt_vocab):
    model.train()
    metric = Accumulator(2)
    for batch in train_iter:
        X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
        bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape((-1, 1))
        dec_input = torch.cat([bos, Y[:, :-1]], 1)
        Y_hat, _ = model(X, dec_input, X_valid_len)
        l = loss(Y_hat, Y, Y_valid_len)
        optimizer.zero_grad()
        l.sum().backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        with torch.no_grad():
            metric.add(l.sum(), l.numel())

    return metric[0] / metric[1]
def inference(model, device, src_vocab, tgt_vocab, num_steps, src_sentence, save_attention_weights=False):
    model.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    env_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, len(src_tokens), src_vocab['<pad>'])
    #infer
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long), dim=0).to(device)
    enc_outputs = model.encoder(enc_X)
    dec_state = model.decoder.init_state(enc_outputs, env_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    #print(enc_X.shape, dec_state.shape)

    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k): 
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def main():
    #hyperparameters
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 300
    #dataloader
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=batch_size, num_steps=num_steps)
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    device = dlf.devices()[0]
    model.to(device)

    for epoch in range(num_epochs):
        l = train(model, optimizer, loss, device, train_iter, tgt_vocab)
        print(f'epoch: {epoch+1}, loss: {l:.2e}')
    src_sentence = "i lost ."
    ans = inference(model, device, src_vocab, tgt_vocab, num_steps, src_sentence)
    print(ans)
    print(type(ans))

if __name__ == '__main__':
    main()
