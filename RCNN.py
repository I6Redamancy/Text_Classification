# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class RCNN(nn.Module):
	def __init__(self, args):
		super(RCNN, self).__init__()
		self.batch_size = args.batch_size
		self.output_size = args.class_num
		self.hidden_size = args.hidden_size
		self.dropout = args.dropout
		self.word_dim = args.word_dim
		self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
		self.init_strategy = args.init_strategy
		self.model_type = args.model_type
		self.num_layers = args.num_layers

		if args.model_type == "LSTM":
			self.model = nn.LSTM(self.word_dim, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
		else:
			self.model = nn.GRU(self.word_dim, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
		self.W2 = nn.Linear(2 * self.hidden_size + self.word_dim, self.hidden_size)
		self.label = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, x):   # (batch_size, word_num, word_dim)
		x = x.permute(1, 0, 2)      # (word_num, batch_size, word_dim)
		if self.init_strategy == "zero":
			h_0 = Variable(torch.zeros(2 * self.num_layers, x.shape[1], self.hidden_size).to(self.device)) 
			c_0 = Variable(torch.zeros(2 * self.num_layers, x.shape[1], self.hidden_size).to(self.device)) # 仅对LSTM有用
		else:
			h_0 = Variable(torch.rand(2 * self.num_layers, x.shape[1], self.hidden_size).to(self.device)) 
			c_0 = Variable(torch.rand(2 * self.num_layers, x.shape[1], self.hidden_size).to(self.device))
		if self.model_type == "LSTM":
			output, _ = self.model(x, (h_0, c_0))   
		else:
			output, _ = self.model(x, h_0)   
		# output:  (word_num, batch_size, 2 * hidden_size)
		
		final_encoding = torch.cat((output, x), 2)
		final_encoding = final_encoding.permute(1, 0, 2) 
		# final_encoding: (word_num, batch_size, 2 * hidden_size + word_dim)
		y = self.W2(final_encoding)         # (batch_size, word_num, hidden_size)
		y = y.permute(0, 2, 1)              # (batch_size, hidden_size, word_num)
		y = F.max_pool1d(y, y.size()[2])    # (batch_size, hidden_size, 1)
		y = y.squeeze(2)
		logits = self.label(y)
		
		return logits
