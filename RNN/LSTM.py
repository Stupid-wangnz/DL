from __future__ import unicode_literals, print_function, division
from io import open
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

import time

from utils import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
device = torch.device('cpu')


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

for filename in findFiles('/codes/DeepLearning/DL/RNN/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.i_fgate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_fgate = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_igate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_igate = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_cell = nn.Linear(input_size, hidden_size, bias=True)
        self.h_cell = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_ogate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_ogate = nn.Linear(hidden_size, hidden_size, bias=True)
        
        #init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x, hidden):
        ht_prev, ct_prev = hidden

        fgate = self.sigmoid(self.i_fgate(x) + self.h_fgate(ht_prev))
        igate = self.sigmoid(self.i_igate(x) + self.h_igate(ht_prev))
        cellt_hat = self.tanh(self.i_cell(x) + self.h_cell(ht_prev))
        ogate = self.sigmoid(self.i_ogate(x) + self.h_ogate(ht_prev))

        ct_next = torch.mul(ct_prev, fgate) + torch.mul(igate, cellt_hat)
        ht_next = torch.mul(self.tanh(ct_next), ogate)

        return (ht_next, ct_next)
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        hidden = self.lstm_cell(x, hidden)
        h, _ = hidden
        out = self.fc(h)
        out = self.softmax(out)

        return out, hidden

    def initHidden(self):
        h0 = torch.zeros(1, self.hidden_size).to(device)
        c0 = torch.zeros(1, self.hidden_size).to(device)
        return h0, c0
    
n_hidden = 128
lstm = LSTM(n_letters, n_hidden, n_categories)
model = lstm.to(device)



def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

criterion = nn.NLLLoss().to(device)
learning_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(category_tensor, line_tensor):
    hidden = model.initHidden()
    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i].to(device), hidden)

    loss = criterion(output, category_tensor.to(device))
    loss.backward()
    optimizer.step()

    return output, loss.item()



if __name__ == "__main__":
    print(device)

    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = 'r' if guess == category else 'x (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
        