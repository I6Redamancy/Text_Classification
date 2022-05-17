import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(args.word_dim * args.word_num, args.hidden_1), nn.BatchNorm1d(args.hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(args.hidden_1, args.hidden_2), nn.BatchNorm1d(args.hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(args.hidden_2, args.class_num))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x