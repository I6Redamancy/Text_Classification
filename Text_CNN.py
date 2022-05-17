import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        H = args.word_num         # 一句话词向量的个数
        W = args.word_dim         # 每个词向量的维度
        C = args.class_num        # 输出层神经元个数
        Ci = 1
        Co = args.kernel_num    # 100
        Ks = args.kernel_sizes  # [3, 4, 5]

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, W)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):  # (N, H, W)
        x = x.unsqueeze(1)  # (N, Ci, H, W)
        # conv(x) (N, Co, H-K+1, 1)    F.relu(conv(x)).squeeze(3) (N, Co, H-K+1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...] * len(Ks)
        # F.max_pool1d(i, i.size(2)) (N, Co, 1)     F.max_pool1d(i, i.size(2)).squeeze(2) (N, Co)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...] * len(Ks)
        x = torch.cat(x, 1)  # (N, len(Ks) * Co)
        x = self.dropout(x)  # (N, len(Ks) * Co)
        logit = self.fc1(x)  # (N, C)
        return logit
