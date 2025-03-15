import sys
sys.path.append('../')

from recurrent_neural_network.pre_data import *
from utils import dlf
from utils.accumulator import Accumulator
import math
import torch
from torch import nn
from torch.nn import functional as F

class RNNModel(nn.Module):
    def __init__(self, layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
    
    def forward(self, inputs, state):
        x = F.one_hot(inputs.T.long(), self.vocab_size)
        x = x.to(torch.float32)
        Y, state = self.rnn(x, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size, device):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            return (torch.zeros((
                self.num_directions *self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))

def train(model, optimizer, loss, device, train_iter, use_random_iter=False):
    model.train()
    state = None
    metric = Accumulator(2)

    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = model.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(state, tuple):
                state = (state[0].detach(), state[1].detach())
            else:
                state.detach_()
        
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = model(X, state)
        l = loss(y_hat, y.long()).mean()
        optimizer.zero_grad()
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1])

def inference(model, device, vocab, str):
    model.eval()
    state = model.begin_state(batch_size=1, device=device)
    output = [vocab[str[0]]]
    get_input = lambda: torch.tensor([output[-1]], device=device).reshape((1, 1))
    for i in str[1:]:
        _, state = model(get_input(), state)
        output.append(vocab[i])
    
    for i in range(100):
        y, state = model(get_input(), state)
        y = int(y.argmax(dim=1).reshape(1))
        output.append(y)
    
    return ''.join([vocab.to_tokens(i) for i in output])

def main():
    # hyperparameters
    batch_size, num_steps = 32, 35
    num_hiddens, num_layers = 256, 2
    num_epochs, lr = 500, 1

    #dataloader
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    #layer
    rnn_layer = nn.RNN(input_size=len(vocab), hidden_size=num_hiddens)
    gru_layer = nn.GRU(input_size=len(vocab), hidden_size=num_hiddens)
    lstm_layer = nn.LSTM(input_size=len(vocab), hidden_size=num_hiddens) # bidirectional=True

    layer = lstm_layer

    #model
    model = RNNModel(layer, len(vocab))
    device = dlf.devices()[0]
    model.to(device)
    print(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    #print(inference(model, device, vocab, "time traveller ")) 
    #train
    for epoch in range(num_epochs):
        ls = train(model, optimizer, loss, device, train_iter)
        print(f'epoch: {epoch+1}, loss: {ls:.2e}')
    print(inference(model, device, vocab, "time traveller "))

if __name__ == '__main__':
    main()