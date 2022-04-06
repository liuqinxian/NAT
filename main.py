import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random

import models
import datasets

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config import DefaultConfig, Logger

import torch.nn.functional as F

args = DefaultConfig()

sys.stdout = Logger(args.save_path + '/record.txt')

# random seed
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

def train(**kwargs):
    args.parse(kwargs)
    args.output()

    # train_dataset = getattr(datasets, args.dataset)(scaler=args.scaler, args=args.vggDataset, train=True)
    # test_dataset = getattr(datasets, args.dataset)(scaler=args.scaler, args=args.vggDataset, train=False)
    full_dataset = getattr(datasets, args.dataset)(scaler=args.scaler)
    train_size = int(args.data_size*0.8)
    test_size = args.data_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_iter = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, args.batch_size)
    print('>'*6 + 'DATA' + '<'*6)

    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.BCELoss()

    print(args.model)
    # model = getattr(models, args.model)(args.vgg16['pretrained'])
    model = getattr(models, args.model)(args.vgg16['pretrained'])
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.Adam['lr'], weight_decay = args.Adam['weight_decay'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.SGD['lr'], momentum=args.SGD['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.StepLR['step_size'], gamma=args.StepLR['gamma'])
    print('>'*6 + 'MODEL' + '<'*6)

    loss_min = 1.0
    for epoch in range(args.epochs):
        model.train()        
        loss_sum, n = 0.0, 0
        for x, y, z in train_iter:
            x, y, z = x.cuda(), y.cuda(), z.cuda()
            if args.rdrop:
                x = x.repeat(2, 1, 1, 1)
            optimizer.zero_grad()
            y_pred = model(x, z).squeeze(1).to(torch.float32)

            if args.rdrop:
                batch_size = y_pred.shape[0]
                batch_size = batch_size // 2
                y1 = y_pred[:batch_size]
                y2 = y_pred[batch_size:]
                l1 = loss_func(y1, y.long())
                l2 = loss_func(y2, y.long())
                l3 = F.kl_div(y1.softmax(dim=-1).log(), y2.softmax(dim=-1), reduction='sum') # kl散度
                loss = l1 + l2 + l3
            else:
                # y = y.to(torch.float32)
                y = y.long()
                # print(y_pred)
                # print(y)
                # print('pred:', y_pred[:5])
                # print('label:', y[:5])
                loss = loss_func(y_pred, y)
                
            # print(y_pred)
            # print(y)
            # print('P:', y_pred)
            # print('L:', y)
            
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n += 1
        scheduler.step()

        model.eval()
        loss_eval, n_eval = 0.0, 0
        acc, pre, rec, f1 = 0.0, 0.0, 0.0, 0.0
        pred, prob, gold = [], [], []
        for x, y, z in test_iter:
            x, y, z = x.cuda(), y.cuda(), z.cuda()
            y_pred = model(x).squeeze(1).to(torch.float32)

            loss_eval += loss_func(y_pred, y.long()).item()
            n_eval += 1
            prob.extend(y_pred[:, 0].tolist())
            y_pred = torch.argmax(y_pred, 1).cpu().numpy()
            y = y.cpu().numpy()

            # loss_eval += loss_func(y_pred, y.long()).item()
            # n_eval += 1
            # y, y_pred = y.cpu().numpy(), y_pred.detach().cpu().numpy()
            # prob.extend(y_pred.tolist())
            # y_pred = (y_pred > 0.5) + 0

            pred.extend(y_pred.tolist())
            gold.extend(y.tolist())

        # print(prob)
        auc = roc_auc_score(gold, prob)
        acc = accuracy_score(gold, pred)
            
        # print('epoch', epoch, ', train loss:', loss_sum / n, ', test loss:', loss_eval / n_eval, ', lr:', optimizer.param_groups[0]['lr'],
        #         'accuracy:', acc, ', precision:', pre, ', recall:', rec, ', f1:', f1, ', auc:', auc)
        print('epoch', epoch, ', train loss:', loss_sum / n, ', test loss:', loss_eval / n_eval, ', lr:', optimizer.param_groups[0]['lr'],
                'accuracy:', acc, ', auc:', auc)

        # pred = np.array(pred)
        # gold = np.array(gold)
        # acc = np.sum((pred==gold)+0) / pred.shape[0]

        # print('epoch', epoch, ', train loss:', loss_sum / n, ', test loss:', loss_eval / n_eval, ', acc', acc, ', lr:', optimizer.param_groups[0]['lr'])

        if loss_min > loss_eval / n_eval:
            loss_min = loss_eval / n_eval
            save_path = args.save_path + '/' + str(epoch)
            checkpoint = {
                'epoch': epoch,
                'min_loss': loss_eval / n_eval,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, save_path)
            print('>'*6 + 'SAVE' + '<'*6)


if __name__ == '__main__':
    import fire
    fire.Fire()