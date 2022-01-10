import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter
import csv
#from torchsummary import summary
from pytorch_model_summary import summary
import argparse
import os
import pickle
import random
import shutil
import time

import cross_val
import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
import util
from collections import Counter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
def evaluate_fir(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    jordan_center = []
    unbet = []
    discen = []
    dynage = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
        h0 = Variable(data['feats'].float()).to(device)  # cuda改
        # if model.label_dim == 13:
        #     labels.append(data['classlabel'].long().numpy())
        # else:
        #     labels.append(data['label'].long().numpy())
        labels.append(data['classlabel'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改

        jordan_center.append(data['center'].long().numpy())
        # unbet.append(data['unbet'].long().numpy())
        # discen.append(data['discen'].long().numpy())
        # dynage.append(data['dynage'].long().numpy())
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    jordan_center = np.hstack(jordan_center)
    # unbet = np.hstack(unbet)
    # discen = np.hstack(discen)
    # dynage = np.hstack(dynage)
    # class0 = {0: 0, 1: 1, 5: 2, 6: 3, 9: 4, 16: 5, 30: 6, 31: 7}
    # realpreds = []
    # for i in preds:
    #     class0_reverse = {v: k for k, v in class0.items()}
    #     realpreds.append(class0_reverse[i])
    # reallabels = []
    # for i in labels:
    #     #class0_reverse = {v: k for k, v in class0.items()}
    #     reallabels.append(class0_reverse[i])
    # read_dic = np.load("dolphins_short_path.npy", allow_pickle=True).item()
    # # print(read_dic[2][3])
    # distance = []
    # for i in range(len(labels)):
    #     a = read_dic[reallabels[i]][realpreds[i]]
    #     distance.append(a)
    # # print(distance)
    # result_dis = {}
    # for i in set(distance):
    #     result_dis[i] = distance.count(i)
    # if 0 in result_dis.keys():
    #     acc = result_dis[0]/len(labels)
    # else:
    #     acc = 0
    # print("reallabels:",reallabels)
    # print("realpreds:", realpreds)
    # print("result_distance:",result_dis)
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              # 'acc_dis':acc,
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " acc_yuan accuracy:", result['acc'])
    # print(name, " acc_dis accuracy:", result['acc_dis'])
    print(name, " prec accuracy:", result['prec'])  # 当使用第一级分类时，分类数为0-4,准确率应该为labels与preds的比

    # return result,labels,preds,jordan_center,unbet,discen,dynage
    return result, labels, preds, jordan_center
def evaluate_sec(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    jordan_center = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
        h0 = Variable(data['feats'].float()).to(device)  # cuda改
        # if model.label_dim == 13:
        #     labels.append(data['classlabel'].long().numpy())
        # else:
        #     labels.append(data['label'].long().numpy())
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改

        jordan_center.append(data['center'].long().numpy())
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    jordan_center = np.hstack(jordan_center)
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              # 'acc_dis':acc,
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " acc_yuan accuracy:", result['acc'])
    # print(name, " acc_dis accuracy:", result['acc_dis'])
    print(name, " prec accuracy:", result['prec'])  # 当使用第一级分类时，分类数为0-4,准确率应该为labels与preds的比

    # return result,labels,preds,jordan_center,unbet,discen,dynage
    return result, labels, preds, jordan_center

# 输出图片的位置
def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname  # 没用为什么？？？运行的是train1
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio * 100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name


def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'


def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)


def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # log a label-less version
    # fig = plt.figure(figsize=(8,6), dpi=300)
    # for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    # plt.tight_layout()
    # fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8, 6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i + 1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters - 1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)

def class_train(dataset, model, args, mask_nodes=True):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-5)
    iter=0
    best_train_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_train_accs = []
    best_train_epochs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)  # cuda改
            # if model.label_dim == 13:                                                           ######
            #     label = Variable(data['classlabel'].long()).to(device)
            # else:
            #     label = Variable(data['label'].long()).to(device)  # cuda()改
            label = Variable(data['label'].long()).to(device)  # cuda改
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改

            # center = Variable(data['center'].long()).to(device)#cuda改
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                # print(label.shape)
                # print(ypred.shape)
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            # if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            # elapsed=10
            total_time += elapsed
            # print(model)
            # log once per XX epochs
        avg_loss /= batch_idx + 1
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        train_result, train_labels, train_preds, train_jordancenter = evaluate_sec(dataset, model, args, name='Train',
                                                                         max_num_examples=100)
        train_accs.append(train_result['acc'])
        train_epochs.append(epoch)
        if train_result['acc'] > best_train_result['acc'] - 1e-7:
            best_train_result['acc'] = train_result['acc']
            best_train_result['epoch'] = epoch
            best_train_result['loss'] = avg_loss
        print('Best train result: ', best_train_result)
    return model, train_accs
def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-5)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)  # cuda改
            # if model.label_dim == 13:
            #     label = Variable(data['classlabel'].long()).to(device)
            # else:
            #     label = Variable(data['label'].long()).to(device)  # cuda()改
            label = Variable(data['classlabel'].long()).to(device)  # cuda改
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改

            # center = Variable(data['center'].long()).to(device)#cuda改
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                # print(label.shape)
                # print(ypred.shape)
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            # if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            # elapsed=10
            total_time += elapsed
            # print(model)
            # log once per XX epochs
            if epoch % 10 == 0 and batch_idx == len(
                    dataset) // 2 and args.method == 'soft-assign' and writer is not None:
                log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
                if args.log_graph:
                    log_graph(adj, batch_num_nodes, writer, epoch, writer_batch_idx, model.assign_tensor)
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result, train_labels, train_preds, train_jordancenter = evaluate_fir(dataset, model, args, name='Train',
                                                                         max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result, val_labels, val_preds, val_jordancenter = evaluate_fir(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
        if test_dataset is not None:
            test_result = evaluate_fir(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['acc'], epoch)

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])
        #if epoch == 99:
            # print('labels:',val_labels)#只有9个，why?？
            # print('preds:',val_preds)

    matplotlib.style.use('seaborn')  # 画图的背景图/画布风格
    plt.switch_backend('agg')  # 将图片保存，暂时不显示图片
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')  # 'bo'蓝色圆圈，'go'绿色圆圈
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)  ########报错
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##dolphins 的分类
# class0 = {0: 0, 1: 1, 5: 2, 6: 3, 9: 4, 16: 5, 30: 6, 31: 7}
# class1 = {2: 0, 4: 1, 17: 2, 18: 3, 32: 4, 33: 5, 38: 6, 39: 7, 40: 8, 42: 9, 43: 10, 44: 11, 45: 12, 46: 13,
#           56: 14, 58: 15, 59: 16, 60: 17}
# class2 = {3: 0, 22: 1, 23: 2, 34: 3, 36: 4, 37: 5, 41: 6, 47: 7, 48: 8, 52: 9, 53: 10, 54: 11}
# class3 = {12: 0, 13: 1, 19: 2, 20: 3, 21: 4, 57: 5}
# class4 = {7: 0, 8: 1, 10: 2, 11: 3, 14: 4, 15: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 35: 12, 49: 13,
#           50: 14, 51: 15, 55: 16, 61: 17}

#pyhsician117的5分类情况
# class0= {13: 0, 14: 1, 17: 2, 22: 3, 25: 4, 27: 5, 28: 6, 30: 7, 32: 8, 33: 9, 34: 10, 36: 11, 47: 12, 48: 13, 49: 14, 50: 15, 51: 16, 53: 17, 55: 18, 56: 19, 58: 20, 59: 21, 64: 22, 73: 23, 79: 24, 95: 25, 99: 26, 100: 27, 103: 28, 104: 29, 108: 30, 112: 31, 113: 32, 114: 33, 115: 34}
# class1= {1: 0, 2: 1, 4: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 37: 8, 38: 9, 39: 10, 40: 11, 41: 12, 52: 13, 57: 14, 65: 15, 66: 16, 67: 17, 68: 18, 69: 19, 70: 20, 71: 21, 72: 22, 74: 23, 76: 24, 82: 25, 84: 26, 88: 27, 90: 28, 93: 29, 97: 30, 101: 31, 102: 32, 109: 33, 110: 34}
# class2= {12: 0, 15: 1, 16: 2, 18: 3, 21: 4, 35: 5, 42: 6, 43: 7, 44: 8, 46: 9, 61: 10, 62: 11, 63: 12, 75: 13, 77: 14, 78: 15, 85: 16, 87: 17, 96: 18, 105: 19}
# class3= {19: 0, 20: 1, 23: 2, 24: 3, 80: 4, 89: 5, 92: 6, 94: 7, 111: 8, 116: 9}
# class4= {0: 0, 3: 1, 5: 2, 6: 3, 26: 4, 29: 5, 31: 6, 45: 7, 54: 8, 60: 9, 81: 10, 83: 11, 86: 12, 91: 13, 98: 14, 106: 15, 107: 16}
#35 35 20 10 17
#ws100
# class0= {0: 0, 2: 1, 4: 2, 8: 3, 9: 4, 25: 5, 59: 6, 60: 7, 61: 8, 72: 9, 73: 10, 74: 11, 75: 12, 86: 13, 89: 14, 98: 15}
# class1= {41: 0, 42: 1, 43: 2, 44: 3, 97: 4, 99: 5}
# class2= {65: 0, 66: 1, 67: 2, 68: 3, 69: 4, 70: 5, 71: 6}
# class3= {6: 0, 21: 1, 22: 2, 23: 3, 24: 4, 33: 5, 34: 6, 36: 7, 45: 8, 58: 9, 76: 10, 80: 11, 82: 12, 84: 13}
# class4= {10: 0, 46: 1, 91: 2, 92: 3, 94: 4, 95: 5, 96: 6}
# class5= {11: 0, 50: 1, 52: 2, 53: 3, 54: 4, 55: 5, 56: 6, 57: 7, 77: 8, 78: 9, 87: 10, 93: 11}
# class6= {12: 0, 13: 1, 14: 2, 47: 3, 48: 4, 49: 5}
# class7= {1: 0, 3: 1, 5: 2, 7: 3, 15: 4, 16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 62: 10, 63: 11, 64: 12, 83: 13, 85: 14}
# class8= {26: 0, 27: 1, 28: 2, 29: 3, 30: 4, 31: 5, 32: 6, 51: 7}
# class9= {35: 0, 37: 1, 38: 2, 39: 3, 40: 4, 79: 5, 81: 6, 88: 7, 90: 8}
#trap171
# class0= {11: 0, 12: 1, 14: 2, 15: 3, 33: 4, 35: 5, 36: 6, 39: 7, 42: 8, 45: 9, 66: 10, 160: 11, 161: 12, 162: 13}
# class1= {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 16: 11, 17: 12}
# class2= {13: 0, 32: 1, 34: 2, 37: 3, 38: 4, 40: 5, 41: 6, 43: 7, 44: 8, 49: 9, 129: 10, 130: 11, 131: 12, 139: 13, 140: 14}
# class3= {67: 0, 68: 1, 69: 2, 70: 3, 71: 4, 72: 5, 73: 6, 74: 7, 75: 8, 79: 9, 89: 10, 90: 11, 91: 12, 159: 13, 167: 14, 169: 15}
# class4= {76: 0, 77: 1, 78: 2, 126: 3, 132: 4, 133: 5, 134: 6, 136: 7, 146: 8, 147: 9, 148: 10, 149: 11, 150: 12, 151: 13}
# class5= {80: 0, 81: 1, 82: 2, 83: 3, 84: 4, 85: 5, 86: 6, 87: 7, 88: 8, 104: 9, 105: 10, 106: 11, 107: 12, 108: 13, 109: 14, 142: 15, 152: 16, 156: 17}
# class6= {92: 0, 93: 1, 94: 2, 95: 3, 96: 4, 123: 5, 124: 6, 125: 7, 127: 8, 128: 9, 153: 10, 154: 11, 155: 12}
# class7= {18: 0, 19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 141: 8}
# class8= {26: 0, 27: 1, 28: 2, 29: 3, 30: 4, 31: 5, 97: 6, 98: 7, 143: 8, 144: 9, 157: 10, 158: 11, 163: 12, 164: 13, 165: 14, 166: 15, 168: 16}
# class9= {99: 0, 100: 1, 101: 2, 102: 3, 103: 4, 145: 5}
# class10= {112: 0, 113: 1, 114: 2, 115: 3, 116: 4, 135: 5, 138: 6}
# class11= {46: 0, 47: 1, 48: 2, 110: 3, 111: 4, 117: 5, 118: 6, 119: 7, 120: 8, 121: 9, 122: 10, 137: 11}
# class12= {50: 0, 51: 1, 52: 2, 53: 3, 54: 4, 55: 5, 56: 6, 57: 7, 58: 8, 59: 9, 60: 10, 61: 11, 62: 12, 63: 13, 64: 14, 65: 15, 170: 16}
#ws200
# class0= {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 189: 10, 190: 11, 191: 12, 192: 13, 193: 14, 194: 15, 195: 16, 196: 17, 197: 18, 198: 19, 199: 20}
# class1= {89: 0, 90: 1, 91: 2, 92: 3, 93: 4, 94: 5, 95: 6, 96: 7, 97: 8, 98: 9, 99: 10, 100: 11}
# class2= {10: 0, 11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10, 21: 11, 22: 12, 23: 13, 24: 14, 25: 15}
# class3= {101: 0, 102: 1, 103: 2, 104: 3, 105: 4, 106: 5, 107: 6, 108: 7, 109: 8, 110: 9, 111: 10, 112: 11, 113: 12}
# class4= {26: 0, 27: 1, 28: 2, 29: 3, 30: 4, 31: 5, 32: 6, 33: 7, 34: 8, 35: 9, 36: 10, 37: 11, 38: 12, 39: 13, 40: 14}
# class5= {114: 0, 115: 1, 116: 2, 117: 3, 118: 4, 119: 5, 120: 6, 121: 7, 122: 8, 123: 9, 124: 10, 125: 11, 126: 12, 127: 13, 128: 14, 129: 15, 130: 16, 131: 17}
# class6= {132: 0, 133: 1, 134: 2, 135: 3, 136: 4, 137: 5, 138: 6, 139: 7, 140: 8, 141: 9, 142: 10, 143: 11, 144: 12, 145: 13, 146: 14, 147: 15, 148: 16, 149: 17, 150: 18}
# class7= {151: 0, 152: 1, 153: 2, 154: 3, 155: 4, 156: 5, 157: 6, 158: 7, 159: 8, 160: 9, 161: 10, 162: 11, 163: 12}
# class8= {41: 0, 42: 1, 43: 2, 44: 3, 45: 4, 46: 5, 47: 6, 48: 7, 49: 8, 50: 9, 51: 10, 52: 11, 53: 12, 188: 13}
# class9= {164: 0, 165: 1, 166: 2, 167: 3, 168: 4, 169: 5, 170: 6, 171: 7, 172: 8, 173: 9, 174: 10, 175: 11, 176: 12, 177: 13, 178: 14, 179: 15}
# class10= {54: 0, 55: 1, 56: 2, 57: 3, 58: 4, 59: 5, 60: 6, 61: 7, 62: 8, 63: 9, 64: 10, 65: 11, 66: 12, 67: 13, 68: 14, 69: 15}
# class11= {180: 0, 181: 1, 182: 2, 183: 3, 184: 4, 185: 5, 186: 6, 187: 7}
# class12= {70: 0, 71: 1, 72: 2, 73: 3, 74: 4, 75: 5, 76: 6, 77: 7, 78: 8, 79: 9, 80: 10, 81: 11, 82: 12, 83: 13, 84: 14, 85: 15, 86: 16, 87: 17, 88: 18}
# # 21 , 12 , 16 , 13 , 15 , 18 , 19 , 13 , 14 , 16 , 16 , 8 , 19
#BA500
# class0= {0: 0, 3: 1, 18: 2, 19: 3, 49: 4, 51: 5, 56: 6, 60: 7, 68: 8, 74: 9, 84: 10, 102: 11, 109: 12, 111: 13, 146: 14, 148: 15, 149: 16, 168: 17, 173: 18, 181: 19, 182: 20, 200: 21, 205: 22, 208: 23, 209: 24, 217: 25, 219: 26, 227: 27, 229: 28, 232: 29, 249: 30, 272: 31, 276: 32, 286: 33, 308: 34, 314: 35, 327: 36, 328: 37, 341: 38, 347: 39, 368: 40, 385: 41, 396: 42, 406: 43, 424: 44, 431: 45, 436: 46, 446: 47, 454: 48, 460: 49, 466: 50, 481: 51, 485: 52, 493: 53, 496: 54}
# class1= {12: 0, 28: 1, 46: 2, 80: 3, 106: 4, 143: 5, 155: 6, 160: 7, 162: 8, 164: 9, 167: 10, 185: 11, 191: 12, 192: 13, 204: 14, 237: 15, 238: 16, 278: 17, 287: 18, 290: 19, 297: 20, 305: 21, 324: 22, 325: 23, 346: 24, 348: 25, 364: 26, 365: 27, 370: 28, 393: 29, 418: 30, 439: 31, 440: 32}
# class2= {2: 0, 7: 1, 8: 2, 17: 3, 29: 4, 30: 5, 32: 6, 35: 7, 44: 8, 62: 9, 85: 10, 91: 11, 108: 12, 113: 13, 124: 14, 166: 15, 170: 16, 172: 17, 174: 18, 176: 19, 177: 20, 183: 21, 187: 22, 193: 23, 194: 24, 198: 25, 202: 26, 206: 27, 207: 28, 215: 29, 218: 30, 225: 31, 230: 32, 243: 33, 252: 34, 254: 35, 257: 36, 262: 37, 273: 38, 281: 39, 282: 40, 296: 41, 302: 42, 303: 43, 307: 44, 318: 45, 335: 46, 355: 47, 359: 48, 362: 49, 400: 50, 404: 51, 410: 52, 425: 53, 427: 54, 434: 55, 448: 56, 452: 57, 470: 58, 480: 59, 486: 60, 487: 61, 499: 62}
# class3= {4: 0, 13: 1, 25: 2, 26: 3, 31: 4, 37: 5, 40: 6, 41: 7, 42: 8, 50: 9, 57: 10, 58: 11, 66: 12, 70: 13, 81: 14, 107: 15, 119: 16, 121: 17, 123: 18, 142: 19, 151: 20, 159: 21, 180: 22, 184: 23, 203: 24, 210: 25, 221: 26, 234: 27, 246: 28, 248: 29, 279: 30, 285: 31, 300: 32, 301: 33, 306: 34, 316: 35, 317: 36, 326: 37, 339: 38, 349: 39, 353: 40, 358: 41, 360: 42, 371: 43, 377: 44, 379: 45, 402: 46, 407: 47, 409: 48, 415: 49, 416: 50, 430: 51, 457: 52, 459: 53, 471: 54, 472: 55, 476: 56, 497: 57}
# class4= {5: 0, 9: 1, 15: 2, 22: 3, 43: 4, 48: 5, 61: 6, 72: 7, 117: 8, 125: 9, 127: 10, 128: 11, 130: 12, 141: 13, 153: 14, 163: 15, 199: 16, 212: 17, 220: 18, 224: 19, 228: 20, 240: 21, 264: 22, 267: 23, 270: 24, 275: 25, 280: 26, 292: 27, 299: 28, 315: 29, 320: 30, 321: 31, 338: 32, 344: 33, 351: 34, 361: 35, 366: 36, 381: 37, 383: 38, 395: 39, 398: 40, 414: 41, 420: 42, 437: 43, 455: 44, 468: 45, 491: 46, 495: 47}
# class5= {1: 0, 90: 1, 171: 2, 188: 3, 242: 4, 263: 5, 283: 6, 304: 7, 323: 8, 329: 9, 342: 10, 352: 11, 376: 12, 388: 13, 392: 14, 403: 15, 405: 16, 428: 17, 489: 18}
# class6= {10: 0, 11: 1, 34: 2, 39: 3, 45: 4, 53: 5, 65: 6, 76: 7, 88: 8, 103: 9, 129: 10, 152: 11, 154: 12, 156: 13, 169: 14, 190: 15, 195: 16, 197: 17, 211: 18, 251: 19, 284: 20, 294: 21, 309: 22, 311: 23, 313: 24, 332: 25, 334: 26, 336: 27, 363: 28, 367: 29, 374: 30, 380: 31, 389: 32, 391: 33, 417: 34, 429: 35, 449: 36, 467: 37, 477: 38, 498: 39}
# class7= {6: 0, 16: 1, 21: 2, 27: 3, 33: 4, 64: 5, 73: 6, 83: 7, 87: 8, 92: 9, 93: 10, 96: 11, 97: 12, 101: 13, 105: 14, 110: 15, 116: 16, 120: 17, 122: 18, 135: 19, 137: 20, 147: 21, 165: 22, 175: 23, 186: 24, 189: 25, 226: 26, 236: 27, 247: 28, 260: 29, 268: 30, 277: 31, 288: 32, 295: 33, 298: 34, 310: 35, 330: 36, 331: 37, 337: 38, 357: 39, 369: 40, 372: 41, 375: 42, 390: 43, 412: 44, 423: 45, 426: 46, 438: 47, 441: 48, 445: 49, 458: 50, 462: 51, 463: 52, 469: 53, 479: 54, 488: 55, 492: 56}
# class8= {14: 0, 23: 1, 36: 2, 52: 3, 54: 4, 55: 5, 63: 6, 75: 7, 79: 8, 89: 9, 95: 10, 98: 11, 100: 12, 131: 13, 133: 14, 138: 15, 139: 16, 140: 17, 144: 18, 161: 19, 233: 20, 245: 21, 259: 22, 269: 23, 289: 24, 293: 25, 312: 26, 373: 27, 382: 28, 384: 29, 386: 30, 394: 31, 397: 32, 432: 33, 443: 34, 444: 35, 453: 36, 464: 37, 473: 38, 474: 39, 483: 40, 490: 41}
# class9= {20: 0, 47: 1, 86: 2, 94: 3, 114: 4, 118: 5, 126: 6, 136: 7, 196: 8, 239: 9, 241: 10, 256: 11, 291: 12, 340: 13, 408: 14, 411: 15, 413: 16, 421: 17, 447: 18, 450: 19, 494: 20}
# class10= {24: 0, 99: 1, 104: 2, 157: 3, 158: 4, 179: 5, 201: 6, 222: 7, 223: 8, 235: 9, 244: 10, 253: 11, 255: 12, 261: 13, 266: 14, 350: 15, 399: 16, 433: 17, 442: 18, 456: 19, 461: 20, 482: 21}
# class11= {69: 0, 78: 1, 112: 2, 150: 3, 214: 4, 231: 5, 322: 6, 333: 7, 343: 8, 378: 9, 401: 10, 419: 11, 422: 12, 465: 13, 478: 14}
# class12= {38: 0, 59: 1, 67: 2, 71: 3, 77: 4, 82: 5, 115: 6, 132: 7, 134: 8, 145: 9, 178: 10, 213: 11, 216: 12, 250: 13, 258: 14, 265: 15, 271: 16, 274: 17, 319: 18, 345: 19, 354: 20, 356: 21, 387: 22, 435: 23, 451: 24, 475: 25, 484: 26}
# #55 , 33 , 63 , 58 , 48 , 19 , 40 , 57 , 42 , 21 , 22 , 15 , 27
#food500
class0= {0: 0, 1: 1, 3: 2, 5: 3, 6: 4, 202: 5, 203: 6, 229: 7, 230: 8, 231: 9, 326: 10, 327: 11, 328: 12, 329: 13, 330: 14, 345: 15, 346: 16, 347: 17, 348: 18, 368: 19, 399: 20, 400: 21, 402: 22, 403: 23, 404: 24, 415: 25, 450: 26, 451: 27, 471: 28}
class1= {2: 0, 4: 1, 10: 2, 11: 3, 12: 4, 24: 5, 25: 6, 26: 7, 27: 8, 28: 9, 43: 10, 54: 11, 65: 12, 66: 13, 92: 14, 93: 15, 94: 16, 95: 17, 96: 18, 98: 19, 105: 20, 172: 21, 174: 22, 179: 23, 180: 24, 181: 25, 182: 26, 183: 27, 184: 28, 186: 29, 188: 30, 205: 31, 207: 32, 208: 33, 239: 34, 240: 35, 243: 36, 244: 37, 245: 38, 246: 39, 249: 40, 257: 41, 260: 42, 261: 43, 262: 44, 274: 45, 284: 46, 286: 47, 287: 48, 289: 49, 290: 50, 291: 51, 295: 52, 298: 53, 324: 54, 361: 55, 363: 56, 382: 57, 390: 58, 393: 59, 398: 60, 409: 61, 410: 62, 411: 63, 413: 64, 423: 65, 424: 66, 425: 67, 426: 68, 428: 69, 429: 70, 430: 71, 435: 72, 455: 73, 472: 74, 474: 75, 478: 76, 479: 77}
class2= {226: 0, 372: 1, 487: 2}
class3= {13: 0, 14: 1, 15: 2, 16: 3, 18: 4, 19: 5, 20: 6, 21: 7, 22: 8, 23: 9, 85: 10, 191: 11, 192: 12, 193: 13, 194: 14, 383: 15, 394: 16, 449: 17}
class4= {17: 0, 77: 1, 79: 2, 110: 3, 111: 4, 161: 5, 162: 6, 163: 7, 164: 8, 166: 9, 167: 10, 168: 11, 169: 12, 171: 13, 173: 14, 247: 15, 252: 16, 297: 17, 369: 18, 370: 19, 391: 20, 416: 21, 417: 22, 418: 23, 419: 24, 458: 25, 459: 26, 473: 27, 486: 28, 490: 29}
class5= {187: 0, 320: 1, 321: 2, 414: 3, 491: 4}
class6= {31: 0, 47: 1, 51: 2, 52: 3, 53: 4, 55: 5, 56: 6, 57: 7, 58: 8, 59: 9, 60: 10, 61: 11, 62: 12, 63: 13, 64: 14, 67: 15, 68: 16, 69: 17, 70: 18, 71: 19, 72: 20, 73: 21, 80: 22, 82: 23, 83: 24, 97: 25, 104: 26, 107: 27, 108: 28, 109: 29, 134: 30, 141: 31, 165: 32, 170: 33, 175: 34, 176: 35, 177: 36, 178: 37, 209: 38, 223: 39, 225: 40, 227: 41, 232: 42, 233: 43, 234: 44, 235: 45, 236: 46, 241: 47, 258: 48, 259: 49, 263: 50, 265: 51, 268: 52, 271: 53, 272: 54, 273: 55, 275: 56, 276: 57, 300: 58, 301: 59, 335: 60, 353: 61, 364: 62, 374: 63, 384: 64, 385: 65, 386: 66, 461: 67, 470: 68, 475: 69, 485: 70, 495: 71}
class7= {211: 0, 212: 1, 213: 2, 214: 3, 215: 4, 444: 5}
class8= {38: 0, 112: 1, 113: 2, 116: 3, 118: 4, 119: 5, 120: 6, 121: 7, 122: 8, 124: 9, 126: 10, 127: 11, 128: 12, 129: 13, 130: 14, 131: 15, 132: 16, 135: 17, 136: 18, 137: 19, 139: 20, 145: 21, 146: 22, 147: 23, 148: 24, 149: 25, 150: 26, 151: 27, 152: 28, 153: 29, 154: 30, 155: 31, 156: 32, 157: 33, 158: 34, 159: 35, 160: 36, 189: 37, 216: 38, 217: 39, 218: 40, 254: 41, 255: 42, 277: 43, 278: 44, 288: 45, 314: 46, 317: 47, 318: 48, 319: 49, 343: 50, 344: 51, 373: 52, 397: 53, 405: 54, 406: 55, 407: 56, 437: 57, 462: 58, 463: 59, 464: 60, 465: 61, 480: 62, 483: 63, 488: 64, 489: 65, 492: 66, 494: 67, 496: 68, 497: 69, 498: 70, 499: 71}
class9= {39: 0, 40: 1, 41: 2, 42: 3, 44: 4, 45: 5, 114: 6, 115: 7, 219: 8, 220: 9, 221: 10, 222: 11, 322: 12}
class10= {33: 0, 34: 1, 35: 2, 36: 3, 37: 4, 46: 5, 48: 6, 49: 7, 50: 8, 75: 9, 76: 10, 78: 11, 308: 12, 309: 13, 310: 14, 311: 15, 312: 16, 313: 17, 336: 18, 337: 19, 338: 20, 339: 21, 378: 22, 379: 23, 387: 24, 388: 25, 389: 26, 396: 27, 412: 28, 432: 29, 469: 30, 481: 31}
class11= {7: 0, 8: 1, 9: 2, 29: 3, 30: 4, 32: 5, 81: 6, 84: 7, 86: 8, 87: 9, 88: 10, 89: 11, 90: 12, 91: 13, 106: 14, 117: 15, 123: 16, 125: 17, 133: 18, 138: 19, 140: 20, 142: 21, 143: 22, 144: 23, 185: 24, 190: 25, 195: 26, 196: 27, 197: 28, 198: 29, 199: 30, 200: 31, 201: 32, 206: 33, 210: 34, 224: 35, 228: 36, 237: 37, 238: 38, 242: 39, 250: 40, 251: 41, 253: 42, 266: 43, 279: 44, 280: 45, 285: 46, 296: 47, 307: 48, 316: 49, 323: 50, 331: 51, 332: 52, 333: 53, 334: 54, 340: 55, 341: 56, 342: 57, 351: 58, 352: 59, 354: 60, 355: 61, 356: 62, 357: 63, 358: 64, 359: 65, 360: 66, 362: 67, 365: 68, 366: 69, 375: 70, 381: 71, 392: 72, 401: 73, 408: 74, 422: 75, 427: 76, 431: 77, 433: 78, 434: 79, 436: 80, 438: 81, 439: 82, 440: 83, 441: 84, 442: 85, 443: 86, 445: 87, 446: 88, 447: 89, 448: 90, 456: 91, 457: 92, 460: 93, 466: 94, 467: 95, 468: 96, 477: 97, 482: 98, 484: 99}
class12= {74: 0, 99: 1, 100: 2, 101: 3, 102: 4, 103: 5, 204: 6, 248: 7, 256: 8, 264: 9, 267: 10, 269: 11, 270: 12, 281: 13, 282: 14, 283: 15, 292: 16, 293: 17, 294: 18, 299: 19, 302: 20, 303: 21, 304: 22, 305: 23, 306: 24, 315: 25, 325: 26, 349: 27, 350: 28, 367: 29, 371: 30, 376: 31, 377: 32, 380: 33, 395: 34, 420: 35, 421: 36, 452: 37, 453: 38, 454: 39, 476: 40, 493: 41}
#29 , 78 , 3 , 18 , 30 , 5 , 72 , 6 , 72 , 13 , 32 , 100 , 42
def benchmark_task_val(args, writer=None, feat='node-label'):

    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)  # 加了num_claaes参数
    for G in graphs:
        if G.graph['classlabel'] == 0:
            for j in class0:
                if j == G.graph['label']:
                    G.graph['label'] = class0[G.graph['label']]
        elif G.graph['classlabel'] == 1:
            for j in class1:
                if j == G.graph['label']:
                    G.graph['label'] = class1[G.graph['label']]
        elif G.graph['classlabel'] == 2:
            for j in class2:
                if j == G.graph['label']:
                    G.graph['label'] = class2[G.graph['label']]
        elif G.graph['classlabel'] == 3:
            for j in class3:
                if j == G.graph['label']:
                    G.graph['label'] = class3[G.graph['label']]
        elif G.graph['classlabel'] == 4:
            for j in class4:
                if j == G.graph['label']:
                    G.graph['label'] = class4[G.graph['label']]
        elif G.graph['classlabel'] == 5:
            for j in class5:
                if j == G.graph['label']:
                    G.graph['label'] = class5[G.graph['label']]
        elif G.graph['classlabel'] == 6:
            for j in class6:
                if j == G.graph['label']:
                    G.graph['label'] = class6[G.graph['label']]
        elif G.graph['classlabel'] == 7:
            for j in class7:
                if j == G.graph['label']:
                    G.graph['label'] = class7[G.graph['label']]
        elif G.graph['classlabel'] == 8:
            for j in class8:
                if j == G.graph['label']:
                    G.graph['label'] = class8[G.graph['label']]
        elif G.graph['classlabel'] == 9:
            for j in class9:
                if j == G.graph['label']:
                    G.graph['label'] = class9[G.graph['label']]
        elif G.graph['classlabel'] == 10:
            for j in class10:
                if j == G.graph['label']:
                    G.graph['label'] = class10[G.graph['label']]
        elif G.graph['classlabel'] == 11:
            for j in class11:
                if j == G.graph['label']:
                    G.graph['label'] = class11[G.graph['label']]
        elif G.graph['classlabel'] == 12:
            for j in class12:
                if j == G.graph['label']:
                    G.graph['label'] = class12[G.graph['label']]
    example_node = util.node_dict(graphs[0])[0]

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])  # 给node本来就有的label属性，再加上feat
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
    all_vals = []
    #num_class = [8,18,12,6,18]  #dolphins 分类
    #num_class = [35,35,20,10,17]  #physician117 分类
    #num_class = [16, 6, 7, 14, 7, 12, 6, 15, 8, 9] #ws100
    #num_class = [14 , 13 , 15 , 16 , 14 , 18 , 13 , 9 , 17 , 6 , 7 , 12 , 17] #trap171
    #num_class = [21 , 12 , 16 , 13 , 15 , 18 , 19 , 13 , 14 , 16 , 16 , 8 , 19] #ws200
    #num_class = [55 , 33 , 63 , 58 , 48 , 19 , 40 , 57 , 42 , 21 , 22 , 15 , 27] #BA500
    num_class = [29 , 78 , 3 , 18 , 30 , 5 , 72 , 6 , 72 , 13 , 32 , 100 , 42] #food500
    model_c = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(1):
        # train_dataset, val_dataset, val_dataset_1, max_num_nodes, input_dim, assign_input_dim = \
        #         cross_val.prepare_val_data_init(graphs, args, i, max_nodes=args.max_nodes)
        train_dataset, train_c0, train_c1, train_c2, train_c3, train_c4, train_c5, train_c6, train_c7, train_c8, \
        train_c9, train_c10, train_c11, train_c12, \
        val_dataset, val_dataset_1, max_num_nodes, input_dim, assign_input_dim = cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        if args.method == 'soft-assign':
            print("i=", i)
            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim).to(device)
            for j in range(len(num_class)):
                model_c[j] = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes,
                    input_dim, args.hidden_dim, args.output_dim, num_class[j], args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim).to(device)

        elif args.method == 'base-set2set':
            print('Method: base-set2set')
            model = encoders.GcnSet2SetEncoder(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
        else:
            print("i=", i)
            print('Method: base,first floor')
            model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
            print('1')
            #summary(model,(500,20,20))
            para = sum([np.prod(list(p.size())) for p in model.parameters()])
            print(para)
            print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4/ 1000 / 1000))

            for j in range(len(num_class)):
                model_c[j] = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, num_class[j],
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改

        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                            writer=writer)
        all_vals.append(np.array(val_accs))

        m_c0, train_accs_c0 = class_train(train_c0, model_c[0], args)
        m_c1, train_accs_c1 = class_train(train_c1, model_c[1], args)
        m_c2, train_accs_c2 = class_train(train_c2, model_c[2], args)
        m_c3, train_accs_c3 = class_train(train_c3, model_c[3], args)
        m_c4, train_accs_c4 = class_train(train_c4, model_c[4], args)
        m_c5, train_accs_c5 = class_train(train_c5, model_c[5], args)
        m_c6, train_accs_c6 = class_train(train_c6, model_c[6], args)
        m_c7, train_accs_c7 = class_train(train_c7, model_c[7], args)
        m_c8, train_accs_c8 = class_train(train_c8, model_c[8], args)
        m_c9, train_accs_c9 = class_train(train_c9, model_c[9], args)
        m_c10, train_accs_c10 = class_train(train_c10, model_c[10], args)
        m_c11, train_accs_c11 = class_train(train_c11, model_c[11], args)
        m_c12, train_accs_c12 = class_train(train_c12, model_c[12], args)
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    #print(all_vals)
    print("fist floor:", np.max(all_vals))
    #print(np.argmax(all_vals))
    for i in range(12):
        para = sum([np.prod(list(p.size())) for p in model_c[i].parameters()])
        print(para)
        print(i,'m_c0 {} : params: {:4f}M'.format(model_c[i]._get_name(), para * 4 / 1000 / 1000))

    print(m_c0)
    print(train_accs_c0)
    print(m_c1)
    print(train_accs_c1)
    print(m_c2)
    print(train_accs_c2)
    print(m_c3)
    print(train_accs_c3)
    print(m_c4)
    print(train_accs_c4)
    # torch.save(model, 'model/'+args.bmname)
    # torch.save(model.state_dict(),'model/'+args.bmname+'.pt')

    # val_dataset_1, max_num_nodes, input_dim, assign_input_dim = \
    #             cross_val.prepare_val_data_validation(graphs, args, max_nodes=args.max_nodes)
    #model_dict = torch.load('model/'+args.bmname)
    #checkpoint = torch.load('model/'+args.bmname+'.pt')
    # model_dict = torch.load('model/dolphins_SI_z0.5_m100')
    # checkpoint = torch.load('model/dolphins_SI_z0.5_m100.pt')
    # model_dict.load_state_dict(checkpoint)

    # model_dict_class0 = torch.load('model/dolphins_SI_z0.5_class0')
    # checkpoint_class0 = torch.load('model/dolphins_SI_z0.5_class0.pt')
    # model_dict_class0.load_state_dict(checkpoint_class0)
    eva_result, eva_labels, eva_realpreds, eva_jordan_center, eva_unbet, eva_discen, eva_dynage = \
        evaluate_class_pro(val_dataset_1, model, m_c0, m_c1, m_c2, m_c3, m_c4, m_c5, m_c6, m_c7, m_c8, m_c9,m_c10,m_c11,m_c12, args)
    print("result:",eva_result)
    #print("labels:", eva_labels)
    #print("preds:", eva_preds)
    print('labels:', eva_labels, 'len(labels):', len(eva_labels))
    print('realpreds_c:', eva_realpreds, 'len(realpreds_c):', len(eva_realpreds))
    print('jc:', eva_jordan_center, 'len(jc):', len(eva_jordan_center))
    # bmname_labels = 'labels/' + args.bmname + '/' + args.bmname + '_labels.txt'
    bmname_labels = 'labels/' + args.bmname + '_labels.txt'
    bmname_preds = 'labels/' + args.bmname + '_preds.txt'
    bmname_val_jordancenter = 'labels/' + args.bmname + '_val_jordancenter.txt'
    bmname_unbet = 'labels/' + args.bmname + '_unbet.txt'
    bmname_discen = 'labels/' + args.bmname + '_discen.txt'
    bmname_dynage = 'labels/' + args.bmname + '_dynage.txt'
    with open(bmname_labels, 'w') as f:  # writer 和前面的247行 重合了，报错
        for i in eva_labels:
            f.write(str(i))  # 整数i不能直接写入
            f.write('\n')
    with open(bmname_preds, 'w') as f:
        for i in eva_realpreds:
            f.write(str(i))  # 整数i不能直接写入
            f.write('\n')
    with open(bmname_val_jordancenter, 'w') as f:
        for i in eva_jordan_center:
            f.write(str(i))  # 整数i不能直接写入
            f.write('\n')
    with open(bmname_unbet, 'w') as f:
        for i in eva_unbet:
            f.write(str(i))  # 整数i不能直接写入
            f.write('\n')
    with open(bmname_discen, 'w') as f:
        for i in eva_discen:
            f.write(str(i))  # 整数i不能直接写入
            f.write('\n')
    with open(bmname_dynage, 'w') as f:
        for i in eva_dynage:
            f.write(str(i))  # 整数i不能直接写入
            f.write('\n')
def evaluate_class_pro(dataset, model, model_c0, model_c1,model_c2,model_c3,model_c4,model_c5,
                       model_c6, model_c7, model_c8, model_c9, model_c10, model_c11, model_c12,
                       args, name='Validation', max_num_examples=None):

    class0_reverse = {value: i for i, value in class0.items()}
    class1_reverse = {value: i for i, value in class1.items()}
    class2_reverse = {value: i for i, value in class2.items()}
    class3_reverse = {value: i for i, value in class3.items()}
    class4_reverse = {value: i for i, value in class4.items()}
    class5_reverse = {value: i for i, value in class5.items()}
    class6_reverse = {value: i for i, value in class6.items()}
    class7_reverse = {value: i for i, value in class7.items()}
    class8_reverse = {value: i for i, value in class8.items()}
    class9_reverse = {value: i for i, value in class9.items()}
    class10_reverse = {value: i for i, value in class10.items()}
    class11_reverse = {value: i for i, value in class11.items()}
    class12_reverse = {value: i for i, value in class12.items()}
    class_reverse = [class0_reverse, class1_reverse, class2_reverse, class3_reverse, class4_reverse, class5_reverse,
                     class6_reverse, class7_reverse, class8_reverse, class9_reverse, class10_reverse, class11_reverse,
                     class12_reverse]  #dict的key不支持[]或{},他们是不可哈希的
    model_dic = [model_c0, model_c1, model_c2, model_c3, model_c4, model_c5, model_c6, model_c7, model_c8, model_c9,
                 model_c10, model_c11, model_c12]
    model.eval()
    classlabels = []
    preds = []
    jordan_center = []
    unbet = []  ###
    discen = []  ###
    dynage = []  ###
    labels = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    preds_c = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
        h0 = Variable(data['feats'].float()).to(device)  # cuda改
        classlabels.append(data['classlabel'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改
        jordan_center.append(data['center'].long().numpy())

        unbet.append(data['unbet'].long().numpy())     ###
        discen.append(data['discen'].long().numpy())   ###
        dynage.append(data['dynage'].long().numpy())   ###
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())
        for i in range(len(model_dic)):
            #print(i)
            if indices.cpu().data.numpy() == i:
                model_dic[i].eval()
                # print(data['label'])
                # print(data['classlabel'])
                # print(class_reverse[i][data['label'].item()])
                # print(data['classlabel'].item())
                labels[i].append(class_reverse[data['classlabel'].item()][data['label'].item()])
                ypred_c = model_dic[i](h0, adj, batch_num_nodes, assign_input=assign_input)
                _, indices_c = torch.max(ypred_c, 1)
                preds_c[i].append(indices_c.cpu().data.numpy())
    classlabels = np.hstack(classlabels)
    preds = np.hstack(preds)
    jordan_center = np.hstack(jordan_center)
    unbet = np.hstack(unbet)   ###
    discen = np.hstack(discen)   ###
    dynage = np.hstack(dynage)   ###
    realpreds_c = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    #reallabels = [[], [], [], [], []]
    for i in range(len(model_dic)):
        if len(labels[i]) != 0:
            labels[i] = np.hstack(labels[i])
            preds_c[i] = np.hstack(preds_c[i])
        for m in preds_c[i]:
            realpreds_c[i].append(class_reverse[i][m])
        #print("labels_c:", i, labels[i])
        #print("realpreds_c:", i, realpreds_c[i])
        # for n in labels[i]:
        #     reallabels[i].append(class_reverse[i][n])
    labels = np.hstack(labels)
    #reallabels = np.hstack(reallabels)
    realpreds_c = np.hstack(realpreds_c)
    print('labels:', labels, 'len(labels)', len(labels))
    #print('reallabels:', reallabels, 'len(reallabels)', len(reallabels))
    # print('realpreds_c:', realpreds_c, 'len(realpreds_c):', len(realpreds_c))

    read_dic = np.load("food500_short_path.npy", allow_pickle=True).item()
    distance_pred = []
    distance_jc = []
    distance_un = []   ###
    distance_discen = []  ###
    distance_dy =[]    ###
    count = 0
    for i in range(len(labels)):
        a = read_dic[labels[i]][realpreds_c[i]]
        b = read_dic[labels[i]][jordan_center[i]]
        h = read_dic[labels[i]][unbet[i]]     ###
        k = read_dic[labels[i]][discen[i]]    ###
        q = read_dic[labels[i]][dynage[i]]    ###
        distance_pred.append(a)
        distance_jc.append(b)
        distance_un.append(h)
        distance_discen.append(k)
        distance_dy.append(q)
        if labels[i] == realpreds_c[i]:
            count = count+1
    print('perdict accuracy:', count/len(labels))
    result_dis = {}
    ave_class = 0
    for i in set(distance_pred):
        result_dis[i] = distance_pred.count(i)
        ave_class = i*result_dis[i]+ave_class
    result_jc = {}
    ave_jc = 0
    for i in set(distance_jc):
        result_jc[i] = distance_jc.count(i)
        ave_jc = i*result_jc[i]+ave_jc
    result_un = {}
    ave_un = 0
    for i in set(distance_un):
        result_un[i] = distance_un.count(i)
        ave_un = i*result_un[i]+ave_un
    result_discen = {}
    ave_discen = 0
    for i in set(distance_discen):
        result_discen[i] = distance_discen.count(i)
        ave_discen = i*result_discen[i]+ave_discen
    result_dy = {}
    ave_dy = 0
    for i in set(distance_dy):
        result_dy[i] = distance_dy.count(i)
        ave_dy = i*result_dy[i]+ave_dy
    result = {'prec': metrics.precision_score(classlabels, preds, average='macro'),  #第一级分类的准确率，preds代表第一级分类的预测结果（0-4）
              'recall': metrics.recall_score(classlabels, preds, average='macro'),
              'acc_yuan': metrics.accuracy_score(classlabels, preds),
              # 'acc_dis':acc,
              'GNN distance' : ave_class/len(labels),
              'jc distance' : ave_jc/len(labels),
              'unbet distance': ave_un / len(labels),
              'discen distance': ave_discen / len(labels),
              'dynage distance': ave_dy / len(labels)
              }
    return result, labels, realpreds_c, jordan_center, unbet, discen, dynage
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
                        const=False, default=True,
                        help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        # dataset='syn1v2',
                        dataset='wormpro72',
                        max_nodes=100,
                        cuda='cpu',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        # num_epochs=1000,
                        num_epochs=1,
                        train_ratio=0.9,
                        test_ratio=0.1,
                        # num_workers=1,
                        num_workers=0,  # 调试train时，要改为0，本来是1
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=20,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                        )
    return parser.parse_args()


def main():
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    # writer = None

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        benchmark_task_val(prog_args, writer=writer)
        # benchmark_task(prog_args, writer=writer)

    writer.close()


if __name__ == "__main__":
    main()
