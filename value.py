import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import random
import math
import numpy as np
from Net.Model import Model
from sklearn import metrics
from data_loader import data_loader
from config import config
from utils import AverageMeter

def valid(test_loader, model):
    model.eval()

    avg_acc = AverageMeter()
    avg_pos_acc = AverageMeter()
    y_true = []
    y_pred = []

    x_label = []
    x_pred = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() if torch.is_tensor(x) else x for x in data]
            rel = data[-1]
            output = model(data[:-1])
            label = rel.argmax(-1)
            _, pred = torch.max(output, -1)
            acc = (pred == label).sum().item() / label.shape[0]
            pos_total = (label != 0).sum().item()
            pos_correct = ((pred == label) & (label != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | acc: %f, pos_acc: %f' % (i + 1, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()

            y_true.append(rel[:, 1:])
            y_pred.append(output[:, 1:])

            x_label.append(label)
            x_pred.append(pred)

    y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()
    x_label = torch.cat(x_label).reshape(-1).detach().cpu().numpy()
    x_pred  = torch.cat(x_pred).reshape(-1).detach().cpu().numpy()

    return y_true, y_pred, x_label, x_pred


if __name__ == '__main__':
    opt = vars(config())

    test_loader = data_loader(opt['test'], opt, shuffle=False, training=False)


    print("\n=== Test ===")
    # Load model
    save_dir = os.path.join(opt['save_dir'], opt['encoder'])
    model = Model(test_loader.dataset.vec_save_dir, test_loader.dataset.rel_num(), opt)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(os.path.join(save_dir, 'model.pth.tar'))['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)

    y_true, y_pred, x_label, x_pred = valid(test_loader, model)

    # F1_score
    print("\n\n[TEST] F1 score: ", metrics.f1_score(x_label, x_pred, labels=None, pos_label=1, average='weighted'))
    print("\n\n[TEST] F1 score: ", metrics.f1_score(x_label, x_pred, labels=None, pos_label=1, average='weighted'))

    # AUC
    auc = metrics.average_precision_score(y_true, y_pred)
    print("AUC: {}".format(auc))

    # P@N values
    if opt['dataset'].lower() == 'biorel':
        # P@N: 4000-8000-12000-16000
        order = np.argsort(-y_pred)
        p4000 = (y_true[order[:4000]]).mean() * 100
        p8000 = (y_true[order[:8000]]).mean() * 100
        p12000 = (y_true[order[:12000]]).mean() * 100
        p16000 = (y_true[order[:16000]]).mean() * 100
        print("P@4000: {0:.2f}, P@8000: {1:.2f}, P@12000: {2:.2f}, P@16000:{3:.2f}, Mean: {4:.2f}".format(p4000, p8000,p12000, p16000, (p4000 + p8000 + p12000 + p16000) / 4))
    elif opt['dataset'].lower() == 'tbga':
        # P@N: 50-100-250-500-1000
        order = np.argsort(-y_pred)
        p50 = (y_true[order[:50]]).mean() * 100
        p100 = (y_true[order[:100]]).mean() * 100
        p250 = (y_true[order[:250]]).mean() * 100
        p500 = (y_true[order[:500]]).mean() * 100
        p1000 = (y_true[order[:1000]]).mean() * 100
        print("P@50: {0:.2f}, P@100: {1:.2f}, P@250: {2:.2f}, P@500:{3:.2f}, P@1000:{4:.2f}, Mean: {5:.2f}".format(p50, p100, p250, p500, p1000, (p50 + p100 + p250 + p500 + p1000) / 5))


    # PR and saving
    order = np.argsort(y_pred)[::-1]
    correct = 0.
    total = y_true.sum()
    precision = []
    recall = []
    for i, o in enumerate(order):
        correct += y_true[o]
        precision.append(float(correct) / (i + 1))
        recall.append(float(correct) / total)
    precision = np.array(precision)
    recall = np.array(recall)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'precision.npy'), precision)
    np.save(os.path.join(save_dir, 'recall.npy'), recall)
