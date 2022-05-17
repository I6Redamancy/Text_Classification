import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import os
from gensim.models import keyedvectors
from Text_CNN import CNN_Text
from RCNN import RCNN
from RNN import RNN
from Transformer import Transformer
from MLP import MLP
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50) # Transformer: 70
parser.add_argument('--lr', type=float, default=0.1)    # Transformer: 0.02
parser.add_argument('--batch_size', type=int, default=128)    # Transformer: 64
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--algo', type=str, default='CNN')  # CNN RCNN Transformer MLP RNN
parser.add_argument('--word_num', type=int, default=680)    # Transformer MLP RNN: 50
parser.add_argument('--word_dim', type=int, default=50)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--not_word_strategy', type=str, default="jump") # jump表示当遇到没有对应词向量的词直接跳过，random表示随机初始化
# CNN args
parser.add_argument('--kernel_num', type=int, default=100)  
parser.add_argument('--kernel_sizes', type=str, default='3,4,5')
# RCNN RNN args 
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--model_type', type=str, default="LSTM")   # LSTM/GRU
parser.add_argument('--init_strategy', type=str, default="zero") # zero/random 
parser.add_argument('--num_layers', type=int, default=1)
# transformer args
parser.add_argument('--head_num', type=int, default=2)
parser.add_argument('--head_dim', type=int, default=25) # 每个head的其中一维，另一维为word_dim
parser.add_argument('--encoder_num', type=int, default=4)
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--choose_weight', type=str, default="weight")  # weight none
# MLP args
parser.add_argument('--hidden_1', type=int, default=512)
parser.add_argument('--hidden_2', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=5e-4) # 0.005达到74.8%

args = parser.parse_args()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
get_word_vec = keyedvectors.load_word2vec_format("./datasets/wiki_word2vec_50.bin", binary=True)
# get_word_vec[['黑人','一些']] 50维向量
if not os.path.exists('save_model'):
    os.mkdir('save_model')
model_path = 'save_model/' + args.algo + str(args.word_num) + '.pkl'

if args.algo == 'Transformer':
    assert args.head_num * args.head_dim == args.word_dim
    info = 'epochs = %d, algo = %s, hidden = %d, head_num = %d, encoder_num = %d, lr = %.3f, word_num = %d' \
        % (args.epochs, args.algo, args.hidden, args.head_num, args.encoder_num, args.lr, args.word_num)
elif args.algo == 'CNN':
    info = 'epochs = %d, algo = %s, not_word_strategy = %s, lr = %.3f, word_num = %d' \
        % (args.epochs, args.algo, args.not_word_strategy, args.lr, args.word_num)
elif args.algo == 'RCNN':
    info = 'epochs = %d, algo = %s, hidden_size = %d, model_type = %s, num_layers = %d, init_strategy = %s, lr = %.3f, word_num = %d' \
        % (args.epochs, args.algo, args.hidden_size, args.model_type, args.num_layers, args.init_strategy, args.lr, args.word_num)
elif args.algo == 'MLP':
    info = 'epochs = %d, algo = %s, hidden_1 = %d, hidden_2 = %d, lr = %.3f, word_num = %d, weight_decay = %.4f' \
        % (args.epochs, args.algo, args.hidden_1, args.hidden_2, args.lr, args.word_num, args.weight_decay)
elif args.algo == 'RNN':
    info = 'epochs = %d, algo = %s, hidden_size = %d, num_layers = %d, lr = %.3f, word_num = %d' \
        % (args.epochs, args.algo, args.hidden_size, args.num_layers, args.lr, args.word_num)
print(info)

def deal_file(input_file):
    f = open(input_file,"r") 
    lines = []
    while True: 
        line = f.readline() 
        if not line:
            break
        lines.append(line)
    random.shuffle(lines)

    now_num = 0
    label_list = []
    batch_label = torch.zeros(args.batch_size)
    sentence_list = []
    batch_sentence = torch.zeros(args.batch_size, args.word_num * args.word_dim) if args.algo == "MLP" \
                else torch.zeros(args.batch_size, args.word_num, args.word_dim)
    step = 0
    for line in (lines):
        batch_label[now_num] = 0 if line[0] == '0' else 1
        line = line[2:-1] 
        word_list = line.split(' ')
        this_sentence = torch.zeros(args.word_num, args.word_dim)
        counting = 0
        for single_word in word_list:
            if counting == args.word_num:
                break
            try:
                this_sentence[counting] = torch.as_tensor(get_word_vec[single_word].copy())
                counting += 1
            except:
                if(args.not_word_strategy == "random"):
                    this_sentence[counting] = torch.rand(args.word_dim)
                    counting += 1
        batch_sentence[now_num] = this_sentence.view(-1) if args.algo == 'MLP' else this_sentence
        now_num += 1
        if now_num == args.batch_size:
            now_num = 0
            label_list.append(batch_label)
            sentence_list.append(batch_sentence)
            step += 1
            if (step + 1) * args.batch_size > len(lines):
                left_num = len(lines) - step * args.batch_size
                batch_label = torch.zeros(left_num)
                batch_sentence = torch.zeros(left_num, args.word_num * args.word_dim) if args.algo == "MLP" \
                    else torch.zeros(left_num, args.word_num, args.word_dim)
                continue
            batch_label = torch.zeros(args.batch_size)
            batch_sentence = torch.zeros(args.batch_size, args.word_num * args.word_dim) if args.algo == "MLP" \
                else torch.zeros(args.batch_size, args.word_num, args.word_dim)
    if len(batch_label) > 0:
        label_list.append(batch_label)
        sentence_list.append(batch_sentence)
    return label_list, sentence_list

(train_label_list, train_sentence_list) = deal_file("./datasets/train.txt")
(validation_label_list, validation_sentence_list) = deal_file("./datasets/validation.txt")
(test_label_list, test_sentence_list) = deal_file("./datasets/test.txt")

NLP_model = {
    'CNN': CNN_Text(args),
    'RCNN': RCNN(args),
    'Transformer': Transformer(args),
    'MLP': MLP(args),
    'RNN': RNN(args),
}
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
net = NLP_model[args.algo]
net.to(device)

criterion = torch.nn.CrossEntropyLoss()
if args.algo == 'CNN' or args.algo == "RCNN" or args.algo == 'MLP' or args.algo == 'RNN':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.algo == 'Transformer':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def test(if_valid):
    total = 0
    if if_valid == "validation":
        label_list = validation_label_list
        sentence_list = validation_sentence_list
    elif if_valid == "test":
        label_list = test_label_list
        sentence_list = test_sentence_list
    correct = 0
    right_pos = 0
    net.eval()
    with torch.no_grad():
        for labels, inputs in zip(label_list, sentence_list):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += predicted.eq(labels).sum().item()
            for i in range(len(labels)):
                if labels[i] == 0 and labels[i] == predicted[i]:
                    right_pos += 1
    f_score = 2 / (2 + (total - correct) / (right_pos + 1e-10))
    return correct / total, f_score

def train():
    best_acc = 0.0
    for epoch in range(args.epochs):
        total = 0
        t = time.time()
        running_loss = 0.0
        net.train()
        step = 0
        for labels, inputs in zip(train_label_list, train_sentence_list): 
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.shape[0]
            step += 1
        if args.algo == 'CNN' or args.algo == 'RCNN' or args.algo == 'MLP' or args.algo == 'RNN':
            scheduler.step()
        (test_acc, f_score) = test('validation')
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), model_path)
        print('[%3d] Train data = %d  Loss = %.4f  Validation acc = %.4f  F-score = %.4f  Time = %.2fs' % \
            (epoch + 1, total, running_loss / step, test_acc, f_score, time.time() - t))

train()
net.load_state_dict(torch.load(model_path))
(acc, f_score) = test('test')
print('Test accuracy = %.4f, F-score = %.4f' % (acc, f_score))

# def get_max(input_file):
#     max_len = 0
#     f = open(input_file,"r") 
#     i = 0
#     while True: 
#         i += 1
#         line = f.readline() 
#         line = line[2:-1] #去掉换行符
#         if not line:
#             break
#         length = len(line.split(' '))
#         max_len = length if length>max_len else max_len
#     return max_len
# print(get_max("train.txt"))
# print(get_max("test.txt"))
# print(get_max("validation.txt"))