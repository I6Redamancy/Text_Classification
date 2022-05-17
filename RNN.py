# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(args.word_dim, args.hidden_size, num_layers=args.num_layers, bidirectional=True, dropout=args.dropout)
        self.fc = nn.Linear(args.hidden_size * 2, args.class_num)

    def forward(self, x):	 # (batch_size, word_num, word_dim)
        x = x.permute(1, 0, 2)   # (word_num, batch_size, word_dim)
        output, (h, c) = self.rnn(x)	# (2, batch_size, hidden_size)
        h = torch.cat((h[0], h[1]), dim=1)	# (batch_size, 2 * hidden_size)
        out = self.fc(h)
        return out