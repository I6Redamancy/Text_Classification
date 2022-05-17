import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

'''Attention Is All You Need'''

class ClassifierLayer(nn.Module):
    def __init__(self, word_dim, hidden, class_num, dropout=0.0):
        super(ClassifierLayer, self).__init__()
        self.fc1 = nn.Linear(word_dim, hidden)
        self.fc2 = nn.Linear(hidden, class_num)
        self.act_func = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, word_dim]
        out = self.fc1(x) # [batch_size, hidden]
        out = self.act_func(out)
        out = self.dropout(out)
        out = self.fc2(out) # [batch_size, class_num]
        out = F.softmax(out, dim=-1)
        return out

class WeightSum(nn.Module):
    def __init__(self, word_num, word_dim):
        super(WeightSum, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(np.random.randn(word_dim)), requires_grad=True)

    def forward(self, x):
        # x: [batch_size, word_num, word_dim]
        attention_context = torch.matmul(self.weight, x.permute(0, 2, 1)) # [batch_size, word_num]
        attention_context = attention_context.masked_fill_(attention_context==0.0, -1e10) # mask
        attention_w = F.softmax(attention_context, dim=-1) # [batch_size, word_num]
        attention_w = attention_w.unsqueeze(dim=1) # [batch_size, 1, word_num]
        out = torch.bmm(attention_w, x)  #[batch_size, 1, word_dim] 
        out = out.squeeze(dim=1)  #[batch, word_dim]
        return out

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        # head_num * head_dim = word_dim
        self.word_num = args.word_num
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.choose_weight = args.choose_weight

        self.postion_embedding = Positional_Encoding(args.word_dim, args.word_num, args.dropout, self.device)
        self.encoder = Encoder(args.word_dim, args.head_dim, args.head_num, args.hidden, args.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)     # 使得每个encoder参数不同
            for _ in range(args.encoder_num)])
        if self.choose_weight == 'weight':
            self.weight_sum = WeightSum(args.word_num, args.word_dim)
            self.classifer = ClassifierLayer(args.word_dim, args.hidden, args.class_num, args.dropout)
        elif self.choose_weight == 'none':
            self.fc1 = nn.Linear(args.word_num * args.word_dim, args.class_num)

    def forward(self, x):   # (batch_size, word_num, word_dim)
        out = self.postion_embedding(x)     # (batch_size, word_num, word_dim)
        for encoder in self.encoders:
            out = encoder(out)           # out: (batch_size, word_num, word_dim)
        if self.choose_weight == 'none':
            out = out.view(out.size(0), -1)     #   (batch_size, word_num * word_dim)
            out = self.fc1(out)
            # out = F.softmax(out, dim=-1)
        elif self.choose_weight == 'weight':
            out = self.weight_sum(out)
            out = self.classifer(out)
        return out


class Encoder(nn.Module):
    def __init__(self, word_dim, head_dim, head_num, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(word_dim, head_dim, head_num, dropout)
        self.feed_forward = Position_wise_Feed_Forward(word_dim, hidden, dropout)

    def forward(self, x):
        out = self.attention(x) # (batch_size, word_num, word_dim)
        out = self.feed_forward(out) # (batch_size, word_num, word_dim)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, word_dim, word_num, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / word_dim)) for i in range(word_dim)] for pos in range(word_num)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, word_dim, head_dim, head_num, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim

        self.fc_Q = nn.Linear(word_dim, head_num * head_dim)
        self.fc_K = nn.Linear(word_dim, head_num * head_dim)
        self.fc_V = nn.Linear(word_dim, head_num * head_dim)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(head_num * head_dim, word_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(word_dim)

    def forward(self, x):   # (batch_size, word_num, word_dim)
        batch_size = x.size(0) 
        scale = x.size(2) ** -0.5
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)    # (batch_size, word_num, head_num * head_dim)
        Q = Q.view(batch_size * self.head_num, -1, self.head_dim)
        K = K.view(batch_size * self.head_num, -1, self.head_dim)
        V = V.view(batch_size * self.head_num, -1, self.head_dim)   # (batch_size * head_num, word_num, head_dim)
        # scale = K.size(-1) ** -0.5  # 缩放因子:避免点乘导致结果过大，进入 softmax 函数的饱和域，导致梯度消失。
        context = self.attention(Q, K, V, scale)    # (batch_size * head_num, word_num, head_dim)

        context = context.view(batch_size, -1, self.head_dim * self.head_num)    # (batch_size, word_num, head_num * head_dim)
        out = self.fc(context)      # (batch_size, word_num, head_num * head_dim)
        out = self.dropout(out)
        out = out + x  # 残差连接 (batch_size, word_num, head_num * head_dim)
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, word_dim, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(word_dim, hidden)
        self.fc2 = nn.Linear(hidden, word_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(word_dim)

    def forward(self, x):      # (batch_size, word_num, word_dim)
        out = self.fc1(x)      # out: (batch_size, word_num, hidden)
        out = F.relu(out)      
        out = self.fc2(out)    # (batch_size, word_num, word_dim)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
