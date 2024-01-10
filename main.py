import os
import time
import torch
import argparse
import torch.nn as nn
import numpy as np
from data.data_loader import load_data
from model.FourierGNN import FGN
from utils.utils import save_model, load_model, evaluate

# 训练参数
parser = argparse.ArgumentParser(description='fourier graph network for classification')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--data', type=str, default='ABIDE-871', help='data set')
parser.add_argument('--seq_length', type=int, default=36, help='inout length')
parser.add_argument('--pre_length', type=int, default=2, help='predict length')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--feature_size', type=int, default='116', help='feature size')
parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
parser.add_argument('--batch_size', type=int, default=16, help='input data batch size')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
args = parser.parse_args()

# 直接获取数据集
train_dataloader, test_dataloader, val_dataloader = load_data(args.seq_length, args.batch_size)

# 配置训练参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FGN(pre_length=args.pre_length, embed_size=args.embed_size, feature_size=args.feature_size,
            seq_length=args.seq_length, hidden_size=args.hidden_size)
my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
forecast_loss = nn.BCEWithLogitsLoss().to(device)

# create output dir
result_train_file = os.path.join('output', args.data, 'train')
result_test_file = os.path.join('output', args.data, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)


def validate(model, vali_loader):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    sne = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")
        forecast = model(x)
#        y = y.permute(0, 2, 1).contiguous()
        loss = forecast_loss(forecast, y)
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)
    preds = np.array(preds[:-1])  # ? need to fix 最后一组预测对不齐
    trues = np.array(trues[:-1])  # ? need to fix 最后一组预测对不齐
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    model.train()
    sne.append(evaluate(trues, preds))
    return loss_total/cnt


def test(epoch: int):
    result_test_file = 'output/'+args.data+'/train'
    model = load_model(result_test_file, epoch)
    model.eval()
    preds = []
    trues = []
    sne = []
    for index, (x, y) in enumerate(train_dataloader):
        y = y.float().to("cuda:0")
        x = x.float().to("cuda:0")
        forecast = model(x)
#        y = y.permute(0, 2, 1).contiguous()
        forecast = forecast.detach().cpu().numpy()  # .squeeze()
        y = y.detach().cpu().numpy()  # .squeeze()
        preds.append(forecast)
        trues.append(y)

    preds = np.array(preds[:-1])  # ? need to fix 最后一组预测对不齐
    trues = np.array(trues[:-1])  # ? need to fix 最后一组预测对不齐
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    sne.append(evaluate(trues, preds))


if __name__ == '__main__':

    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for index, (x, y) in enumerate(train_dataloader):
            cnt += 1
            y = y.float().to("cuda:0")
            x = x.float().to("cuda:0")
            forecast = model(x)
            loss = forecast_loss(forecast, y)
            loss.backward()
            my_optim.step()
            loss_total += float(loss)

        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            val_loss = validate(model, val_dataloader)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | val_loss {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))
        save_model(model, result_train_file, epoch)
#        for i in range(0,20):
#            print(f"the epoch is {i}")
#            test(i)

