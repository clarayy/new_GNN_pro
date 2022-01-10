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
def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
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
        if model.label_dim == 6:
            labels.append(data['classlabel'].long().numpy())
        else:
            labels.append(data['label'].long().numpy())
        #labels.append(data['classlabel'].long().numpy())
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
            if model.label_dim == 6:                                                           ######
                label = Variable(data['classlabel'].long()).to(device)
            else:
                label = Variable(data['label'].long()).to(device)  # cuda()改
            #label = Variable(data['classlabel'].long()).to(device)  # cuda改
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
        train_result, train_labels, train_preds, train_jordancenter = evaluate(dataset, model, args, name='Train',
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
            if model.label_dim == 6:
                label = Variable(data['classlabel'].long()).to(device)
            else:
                label = Variable(data['label'].long()).to(device)  # cuda()改
            #label = Variable(data['classlabel'].long()).to(device)  # cuda改
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
        result, train_labels, train_preds, train_jordancenter = evaluate(dataset, model, args, name='Train',
                                                                         max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result, val_labels, val_preds, val_jordancenter = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
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


def syn_community1v2(args, writer=None, export_graphs=False):
    # data
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 500,
                             featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    for G in graphs1:
        G.graph['label'] = 0
    if export_graphs:
        util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3,
                                        [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))])
    for G in graphs2:
        G.graph['label'] = 1
    if export_graphs:
        util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim).to(device)  # cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                                           args.num_gc_layers, bn=args.bn).to(device)  # cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers, bn=args.bn).to(device)

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)


def syn_community2hier(args, writer=None):
    # data
    feat_gen = [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))]
    graphs1 = datagen.gen_2hier(1000, [2, 4], 10, range(4, 5), 0.1, 0.03, feat_gen)
    graphs2 = datagen.gen_2hier(1000, [3, 3], 10, range(4, 5), 0.1, 0.03, feat_gen)
    graphs3 = datagen.gen_2community_ba(range(28, 33), range(4, 7), 1000, 0.25, feat_gen)

    for G in graphs1:
        G.graph['label'] = 0
    for G in graphs2:
        G.graph['label'] = 1
    for G in graphs3:
        G.graph['label'] = 2

    graphs = graphs1 + graphs2 + graphs3

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)

    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, linkpred=args.linkpred, args=args, assign_input_dim=assign_input_dim).to(device)  # cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                                           args.num_gc_layers, bn=args.bn, args=args,
                                           assign_input_dim=assign_input_dim).to(device)  # cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers, bn=args.bn, args=args).to(device)  # cuda()
    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)

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
#BA200   10类
# class0= {22: 0, 62: 1, 95: 2, 97: 3, 112: 4, 135: 5, 141: 6, 159: 7, 170: 8, 175: 9, 191: 10}
# class1= {1: 0, 11: 1, 15: 2, 24: 3, 35: 4, 42: 5, 51: 6, 53: 7, 68: 8, 73: 9, 77: 10, 78: 11, 81: 12, 109: 13, 111: 14, 120: 15, 122: 16, 139: 17, 147: 18, 149: 19, 151: 20, 157: 21, 162: 22, 178: 23, 183: 24, 187: 25, 189: 26, 190: 27}
# class2= {5: 0, 8: 1, 31: 2, 56: 3, 74: 4, 80: 5, 104: 6, 106: 7, 108: 8, 125: 9, 148: 10, 156: 11, 188: 12}
# class3= {4: 0, 9: 1, 25: 2, 63: 3, 66: 4, 75: 5, 82: 6, 89: 7, 92: 8, 93: 9, 98: 10, 110: 11, 117: 12, 119: 13, 121: 14, 127: 15, 128: 16, 129: 17, 130: 18, 133: 19, 138: 20, 154: 21, 167: 22}
# class4= {2: 0, 3: 1, 16: 2, 20: 3, 26: 4, 30: 5, 37: 6, 49: 7, 69: 8, 79: 9, 85: 10, 90: 11, 115: 12, 116: 13, 124: 14, 131: 15, 136: 16, 140: 17, 143: 18, 150: 19, 179: 20}
# class5= {6: 0, 23: 1, 43: 2, 48: 3, 65: 4, 88: 5, 102: 6, 107: 7, 123: 8, 134: 9, 153: 10, 163: 11, 164: 12, 166: 13, 180: 14, 182: 15, 184: 16, 194: 17, 196: 18}
# class6= {12: 0, 29: 1, 34: 2, 55: 3, 72: 4, 99: 5, 100: 6, 101: 7, 113: 8, 137: 9, 144: 10, 152: 11, 165: 12, 172: 13, 174: 14, 181: 15, 185: 16, 195: 17}
# class7= {7: 0, 10: 1, 13: 2, 19: 3, 32: 4, 36: 5, 38: 6, 39: 7, 46: 8, 47: 9, 58: 10, 61: 11, 64: 12, 70: 13, 76: 14, 83: 15, 94: 16, 118: 17, 126: 18, 142: 19, 145: 20, 155: 21, 161: 22, 171: 23, 176: 24, 192: 25, 198: 26, 199: 27}
# class8= {14: 0, 27: 1, 40: 2, 41: 3, 57: 4, 60: 5, 71: 6, 84: 7, 158: 8, 169: 9, 193: 10, 197: 11}
# class9= {0: 0, 17: 1, 18: 2, 21: 3, 28: 4, 33: 5, 44: 6, 45: 7, 50: 8, 52: 9, 54: 10, 59: 11, 67: 12, 86: 13, 87: 14, 91: 15, 96: 16, 103: 17, 105: 18, 114: 19, 132: 20, 146: 21, 160: 22, 168: 23, 173: 24, 177: 25, 186: 26}
#11 , 28 , 13 , 23 , 21 , 19 , 18 , 28 , 12 , 27
#BA100_3
class0= {0: 0, 2: 1, 4: 2, 9: 3, 21: 4, 40: 5, 49: 6, 50: 7, 59: 8, 63: 9, 64: 10, 70: 11, 72: 12, 75: 13, 79: 14, 83: 15, 88: 16, 94: 17, 96: 18, 99: 19}
class1= {1: 0, 11: 1, 25: 2, 36: 3, 39: 4, 45: 5, 47: 6, 52: 7, 53: 8, 66: 9, 71: 10, 78: 11, 93: 12, 95: 13}
class2= {3: 0, 12: 1, 20: 2, 30: 3, 31: 4, 38: 5, 42: 6, 54: 7, 57: 8, 73: 9, 76: 10, 82: 11, 86: 12, 87: 13}
class3= {5: 0, 7: 1, 13: 2, 14: 3, 18: 4, 19: 5, 29: 6, 33: 7, 34: 8, 41: 9, 43: 10, 60: 11, 61: 12, 62: 13, 65: 14, 67: 15, 80: 16, 89: 17, 90: 18, 91: 19}
class4= {6: 0, 15: 1, 16: 2, 17: 3, 22: 4, 23: 5, 24: 6, 26: 7, 32: 8, 35: 9, 44: 10, 46: 11, 55: 12, 69: 13, 74: 14, 77: 15, 81: 16, 92: 17, 98: 18}
class5= {8: 0, 10: 1, 27: 2, 28: 3, 37: 4, 48: 5, 51: 6, 56: 7, 58: 8, 68: 9, 84: 10, 85: 11, 97: 12}
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
        # elif G.graph['classlabel'] == 6:
        #     for j in class6:
        #         if j == G.graph['label']:
        #             G.graph['label'] = class6[G.graph['label']]
        # elif G.graph['classlabel'] == 7:
        #     for j in class7:
        #         if j == G.graph['label']:
        #             G.graph['label'] = class7[G.graph['label']]
        # elif G.graph['classlabel'] == 8:
        #     for j in class8:
        #         if j == G.graph['label']:
        #             G.graph['label'] = class8[G.graph['label']]
        # elif G.graph['classlabel'] == 9:
        #     for j in class9:
        #         if j == G.graph['label']:
        #             G.graph['label'] = class9[G.graph['label']]
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
    #num_class = [11 , 28 , 13 , 23 , 21 , 19 , 18 , 28 , 12 , 27]#BA200,10classes
    num_class = [20 , 14 , 14 , 20 , 19 , 13] #BA100_3
    model_c = [[], [], [], [], [], []]
    for i in range(10):
        # train_dataset, val_dataset, val_dataset_1, max_num_nodes, input_dim, assign_input_dim = \
        #         cross_val.prepare_val_data_init(graphs, args, i, max_nodes=args.max_nodes)
        train_dataset, train_c0, train_c1, train_c2, train_c3, train_c4, train_c5, \
        val_dataset, val_dataset_1, max_num_nodes, input_dim,\
        assign_input_dim = cross_val.prepare_val_data_c6(graphs, args, i, max_nodes=args.max_nodes)
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
            print(model)
            for j in range(len(num_class)):
                model_c[j] = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, num_class[j],
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改

            # model_c1 = encoders.GcnEncoderGraph(
            #     input_dim, args.hidden_dim, args.output_dim, 18,
            #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
            #
            # model_c2 = encoders.GcnEncoderGraph(
            #     input_dim, args.hidden_dim, args.output_dim, 12,
            #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
            #
            # model_c3 = encoders.GcnEncoderGraph(
            #     input_dim, args.hidden_dim, args.output_dim, 6,
            #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
            #
            # model_c4 = encoders.GcnEncoderGraph(
            #     input_dim, args.hidden_dim, args.output_dim, 18,
            #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                            writer=writer)
        all_vals.append(np.array(val_accs))

        m_c0, train_accs_c0 = class_train(train_c0, model_c[0], args)
        m_c1, train_accs_c1 = class_train(train_c1, model_c[1], args)
        m_c2, train_accs_c2 = class_train(train_c2, model_c[2], args)
        m_c3, train_accs_c3 = class_train(train_c3, model_c[3], args)
        m_c4, train_accs_c4 = class_train(train_c4, model_c[4], args)
        m_c5, train_accs_c5 = class_train(train_c5, model_c[5], args)
        # m_c6, train_accs_c6 = class_train(train_c6, model_c[6], args)
        # m_c7, train_accs_c7 = class_train(train_c7, model_c[7], args)
        # m_c8, train_accs_c8 = class_train(train_c8, model_c[8], args)
        # m_c9, train_accs_c9 = class_train(train_c9, model_c[9], args)
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    #print(all_vals)
    print("fist floor:", np.max(all_vals))
    #print(np.argmax(all_vals))

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
        evaluate_class_pro(val_dataset_1, model, m_c0, m_c1, m_c2, m_c3, m_c4, m_c5,  args)
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
                       args, name='Validation', max_num_examples=None):

    class0_reverse = {value: i for i, value in class0.items()}
    class1_reverse = {value: i for i, value in class1.items()}
    class2_reverse = {value: i for i, value in class2.items()}
    class3_reverse = {value: i for i, value in class3.items()}
    class4_reverse = {value: i for i, value in class4.items()}
    class5_reverse = {value: i for i, value in class5.items()}
    # class6_reverse = {value: i for i, value in class6.items()}
    # class7_reverse = {value: i for i, value in class7.items()}
    # class8_reverse = {value: i for i, value in class8.items()}
    # class9_reverse = {value: i for i, value in class9.items()}
    class_reverse = [class0_reverse, class1_reverse, class2_reverse, class3_reverse, class4_reverse, class5_reverse,
                     ]  #dict的key不支持[]或{},他们是不可哈希的
    model_dic = [model_c0, model_c1, model_c2, model_c3, model_c4, model_c5 ]
    model.eval()
    classlabels = []
    preds = []
    jordan_center = []
    unbet = []  ###
    discen = []  ###
    dynage = []  ###
    labels = [[], [], [], [], [], []]
    preds_c = [[], [], [], [], [], []]
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
    realpreds_c = [[], [], [], [], [], []]
    #reallabels = [[], [], [], [], []]
    for i in range(len(model_dic)):
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

    read_dic = np.load("BA100_3_short_path.npy", allow_pickle=True).item()
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
    elif prog_args.dataset is not None:
        if prog_args.dataset == 'syn1v2':
            syn_community1v2(prog_args, writer=writer)
        if prog_args.dataset == 'syn2hier':
            syn_community2hier(prog_args, writer=writer)

    writer.close()


if __name__ == "__main__":
    main()
