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

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.i_rgate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_rgate = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_zgate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_zgate = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_cell = nn.Linear(input_size, hidden_size, bias=True)
        self.h_cell = nn.Linear(hidden_size, hidden_size, bias=True)

        self.i_hgate = nn.Linear(input_size, hidden_size, bias=True)
        self.h_hgate = nn.Linear(hidden_size, hidden_size, bias=True)
        
        #init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x, hidden):

        rgate = self.sigmoid(self.i_rgate(x) + self.h_rgate(hidden))
        zgate = self.sigmoid(self.i_zgate(x) + self.h_zgate(hidden))
        hgate = self.tanh(self.i_hgate(x) + self.h_hgate(torch.mul(hidden, rgate)))

        hidden_next = torch.mul(zgate, hidden) + torch.mul((1 - zgate), hgate)

        return hidden_next
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = GRUCell(input_size=input_size, hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        hidden = self.gru_cell(x, hidden)
        out = self.fc(hidden)
        out = self.softmax(out)

        return out, hidden

    def initHidden(self):
        h0 = torch.zeros(1, self.hidden_size).to(device)
        return h0
    
n_hidden = 128
gru = GRU(n_letters, n_hidden, n_categories)
model = gru.to(device)



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
        