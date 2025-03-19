import torch
import torch.nn as nn
import sys
import math
sys.path.append('../')
from attention.attention_utils import *
from recurrent_neural_network.pre_data import *
from utils.accumulator import Accumulator
from utils import dlf

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = multihead_attention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = positionwiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_length):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_length))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = positional_encoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, dropout, use_bias))
    
    def forward(self, X, valid_length):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_length)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = multihead_attention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = multihead_attention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = positionwiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, state):
        enc_outputs, enc_valid_length = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_length = torch.arange(1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_length = None
        
        X2 = self.attention1(X, key_values, key_values, dec_valid_length)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_length)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = positional_encoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), DecoderBlock(key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, dropout, i))
        
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_length, *args):
        return [enc_outputs, enc_valid_length, [None]*self.num_layers]
    
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    
    @property
    def attention_weights(self):
        return self._attention_weights
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_layers, dropout, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.encoder = TransformerEncoder(src_vocab_size, key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_layers, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_layers, dropout)
    
    def forward(self, enc_X, dec_X, encoder_valid_len):
        enc_outputs = self.encoder(enc_X, encoder_valid_len)
        dec_state = self.decoder.init_state(enc_outputs, encoder_valid_len)
        dec_output, dec_state = self.decoder(dec_X, dec_state)
        return dec_output, dec_state

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
    enc_outputs = model.encoder(enc_X, env_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, env_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    #print(enc_X.shape, dec_state.shape)

    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(model.decoder.attention_weights)
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
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs = 0.005, 200
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    src_sentence = "nice to meet you"
    #dataloader
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=batch_size, num_steps=num_steps)
    model = Transformer(len(src_vocab), len(tgt_vocab), key_size, query_size, value_size, num_hiddens, num_heads, norm_shape, ffn_num_input, ffn_num_hiddens, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    device = dlf.devices()[0]
    model.to(device)
    print(device)

    for epoch in range(num_epochs):
        l = train(model, optimizer, loss, device, train_iter, tgt_vocab)
        print(f'epoch: {epoch+1}, loss: {l:.2e}')
    ans = inference(model, device, src_vocab, tgt_vocab, num_steps, src_sentence, True)
    print(ans[0])

if __name__ == '__main__':
    main()