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

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    jordan_center = []
    unbet=[]
    discen=[]
    dynage=[]
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)#cuda改
        h0 = Variable(data['feats'].float()).to(device)#cuda改
        labels.append(data['classlabel'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)#cuda改

        jordan_center.append(data['center'].long().numpy())
        # unbet.append(data['unbet'].long().numpy())
        # discen.append(data['discen'].long().numpy())
        # dynage.append(data['dynage'].long().numpy())
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
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
              #'acc_dis':acc,
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " acc_yuan accuracy:", result['acc'])
    #print(name, " acc_dis accuracy:", result['acc_dis'])
    print(name, " prec accuracy:", result['prec'])#当使用第一级分类时，分类数为0-4,准确率应该为labels与preds的比

    #return result,labels,preds,jordan_center,unbet,discen,dynage
    return result, labels, preds, jordan_center
#输出图片的位置
def evaluate_class(dataset, model, model_class0, args, name='Validation', max_num_examples=None):
    model.eval()
    model_class0.eval()
    classlabels = []
    labels_c0 = []
    preds = []
    jordan_center = []
    preds_c0 = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)#cuda改
        h0 = Variable(data['feats'].float()).to(device)#cuda改
        classlabels.append(data['classlabel'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)#cuda改
        jordan_center.append(data['center'].long().numpy())
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())
        if indices.cpu().data.numpy() == 0:
            labels_c0.append(data['label'].long().numpy())
            ypred_c0 = model_class0(h0, adj, batch_num_nodes, assign_input=assign_input)
            _, indices_c0 = torch.max(ypred_c0, 1)
            preds_c0.append(indices_c0.cpu().data.numpy())

    classlabels = np.hstack(classlabels)
    preds = np.hstack(preds)
    labels_c0 = np.hstack(labels_c0)
    preds_c0 = np.hstack(preds_c0)
    jordan_center = np.hstack(jordan_center)
    class0 = {0: 0, 1: 1, 5: 2, 6: 3, 9: 4, 16: 5, 30: 6, 31: 7}
    realpreds_c0 = []
    for i in preds_c0:
        class0_reverse = {v: k for k, v in class0.items()}
        realpreds_c0.append(class0_reverse[i])
    print("classlabels:", classlabels)
    print("preds:", preds)
    print("labels_c0:", labels_c0)
    print("realpreds_c0:", realpreds_c0)
    print("preds_c0:", preds_c0)
    read_dic = np.load("dolphins_short_path.npy", allow_pickle=True).item()
    # print(read_dic[2][3])
    distance_c0 = []
    distance_jc = []
    for i in range(len(labels_c0)):
        a = read_dic[labels_c0[i]][realpreds_c0[i]]
        distance_c0.append(a)
        b = read_dic[labels_c0[i]][jordan_center[i]]
        distance_jc.append(b)
    # print(distance)
    result_dis = {}
    ave_c0 = 0
    for i in set(distance_c0):
        result_dis[i] = distance_c0.count(i)
        ave_c0 = i*result_dis[i]+ave_c0
    result_jc = {}
    ave_jc = 0
    for i in set(distance_jc):
        result_jc[i] = distance_jc.count(i)
        ave_jc = i*result_jc[i]+ave_jc
    # if 0 in result_dis.keys():
    #     acc = result_dis[0]/len(labels_c0)   #计算的是距离为0的准确率，但此时应该计算的是class0得到的几个数的预测平均距离
    # else:
    #     acc = 0
    print("result_distance:",result_dis)
    print("ave_c0:", ave_c0/len(labels_c0))
    print("ave_jc:", ave_jc/len(labels_c0))
    result = {'prec': metrics.precision_score(classlabels, preds, average='macro'),
              'recall': metrics.recall_score(classlabels, preds, average='macro'),
              'acc_yuan': metrics.accuracy_score(classlabels, preds),
              #'acc_dis':acc,
              'F1': metrics.f1_score(classlabels, preds, average="micro")}
    result_c0 ={
        'acc_class0': metrics.accuracy_score(labels_c0, realpreds_c0)
        #'acc_class0': acc
    }
    print(name, " acc_yuan accuracy:", result['acc_yuan'])
    #print(name, " acc_dis accuracy:", result['acc_dis'])
    print(name, " prec accuracy:", result['prec'])#当使用第一级分类时，分类数为0-4,准确率应该为labels与preds的比
    print('class0 accuracy:',result_c0['acc_class0'])
    return result, labels_c0, preds, jordan_center
def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname  #没用为什么？？？运行的是train1
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))
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
    fig = plt.figure(figsize=(8,6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i+1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)

def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # log a label-less version
    #fig = plt.figure(figsize=(8,6), dpi=300)
    #for i in range(len(batch_idx)):
    #    ax = plt.subplot(2, 2, i+1)
    #    num_nodes = batch_num_nodes[batch_idx[i]]
    #    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    #    G = nx.from_numpy_matrix(adj_matrix)
    #    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color='#336699',
    #            edge_color='grey', width=0.5, node_size=25,
    #            alpha=0.8)

    #plt.tight_layout()
    #fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #writer.add_image('graphs_no_label', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8,6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters-1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)


def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True):
    writer_batch_idx = [0, 3, 6, 9]
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001,weight_decay=1e-5)
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
    #bmname_labels = 'labels/' + args.bmname + '/' + args.bmname + '_labels.txt'
    bmname_labels = 'labels/' + args.bmname + '_labels.txt'
    bmname_preds = 'labels/' + args.bmname + '_preds.txt'
    bmname_val_jordancenter = 'labels/' + args.bmname + '_val_jordancenter.txt'
    # bmname_unbet = 'labels/' + args.bmname + '_unbet.txt'
    # bmname_discen = 'labels/' + args.bmname + '_discen.txt'
    # bmname_dynage = 'labels/' + args.bmname + '_dynage.txt'
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)#cuda改
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)#cuda改
            label = Variable(data['classlabel'].long()).to(device)#cuda改
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)#cuda改

            #center = Variable(data['center'].long()).to(device)#cuda改
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
                #print(label.shape)
                #print(ypred.shape)
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            #if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])
            elapsed = time.time() - begin_time
            #elapsed=10
            total_time += elapsed
            #print(model)
            # log once per XX epochs
            if epoch % 10 == 0 and batch_idx == len(dataset) // 2 and args.method == 'soft-assign' and writer is not None:
                log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
                if args.log_graph:
                    log_graph(adj, batch_num_nodes, writer, epoch, writer_batch_idx, model.assign_tensor)
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result ,train_labels,train_preds,train_jordancenter = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result,val_labels,val_preds,val_jordancenter = evaluate(val_dataset, model, args, name='Validation')
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
        if epoch==99:
            #print('labels:',val_labels)#只有9个，why?？
            #print('preds:',val_preds)
            print(val_labels)
            with open(bmname_labels,'w') as f:#writer 和前面的247行 重合了，报错
                for i in val_labels:
                    f.write(str(i)) # 整数i不能直接写入
                    f.write('\n')
            with open(bmname_preds,'w') as f:
                for i in val_preds:
                    f.write(str(i)) # 整数i不能直接写入
                    f.write('\n')
            with open(bmname_val_jordancenter,'w') as f:
                for i in val_jordancenter:
                    f.write(str(i)) # 整数i不能直接写入
                    f.write('\n')
            # with open(bmname_unbet, 'w') as f:
            #     for i in val_unbet:
            #         f.write(str(i))  # 整数i不能直接写入
            #         f.write('\n')
            # with open(bmname_discen, 'w') as f:
            #     for i in val_discen:
            #         f.write(str(i))  # 整数i不能直接写入
            #         f.write('\n')
            # with open(bmname_dynage, 'w') as f:
            #     for i in val_dynage:
            #         f.write(str(i))  # 整数i不能直接写入
            #         f.write('\n')
    matplotlib.style.use('seaborn') #画图的背景图/画布风格
    plt.switch_backend('agg') #将图片保存，暂时不显示图片
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')#'bo'蓝色圆圈，'go'绿色圆圈
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)          ########报错
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs

def prepare_data(graphs, args, test_graphs=None, max_nodes=0):

    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs),
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

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
                bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim).to(device)#cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn).to(device)#cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn).to(device)

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)

def syn_community2hier(args, writer=None):

    # data
    feat_gen = [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))]
    graphs1 = datagen.gen_2hier(1000, [2,4], 10, range(4,5), 0.1, 0.03, feat_gen)
    graphs2 = datagen.gen_2hier(1000, [3,3], 10, range(4,5), 0.1, 0.03, feat_gen)
    graphs3 = datagen.gen_2community_ba(range(28, 33), range(4,7), 1000, 0.25, feat_gen)

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
                bn=args.bn, linkpred=args.linkpred, args=args, assign_input_dim=assign_input_dim).to(device)#cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args, assign_input_dim=assign_input_dim).to(device)#cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                args.num_gc_layers, bn=args.bn, args=args).to(device)#cuda()
    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)


def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph['label'] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph['label'] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(graphs, args, test_graphs=test_graphs)
    model = encoders.GcnEncoderGraph(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
            args.num_gc_layers, bn=args.bn).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, 'Validation')

def benchmark_task(args, writer=None, feat='node-label'):
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    example_node = util.node_dict(graphs[0])[0]
    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = np.array(G.nodes[u]['label'])#G.nodes
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = \
            prepare_data(graphs, args, max_nodes=args.max_nodes)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        model = encoders.SoftPoolingGcnEncoder(
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim).to(device)
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)#cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, 
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)#cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)
    evaluate(test_dataset, model, args, 'Validation')


def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)#加了num_claaes参数

    example_node = util.node_dict(graphs[0])[0]
    
    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])  #给node本来就有的label属性，再加上feat
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)


    for i in range(1):
        # train_dataset, val_dataset, val_dataset_1, max_num_nodes, input_dim, assign_input_dim = \
        #         cross_val.prepare_val_data_init(graphs, args, i, max_nodes=args.max_nodes)
        train_dataset, train_c0, train_c1, train_c2, train_c3, train_c4, val_dataset, val_dataset_1, max_num_nodes, \
        input_dim, assign_input_dim = cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        if args.method == 'soft-assign':
            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes,
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim).to(device)
        elif args.method == 'base-set2set':
            print('Method: base-set2set')
            model = encoders.GcnSet2SetEncoder(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)#cuda()改
        else:
            print('Method: base')
            model = encoders.GcnEncoderGraph(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)#cuda()改
            print(model)
            print("i=",i)
        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)

    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))
    # torch.save(model, 'model/'+args.bmname)
    # torch.save(model.state_dict(),'model/'+args.bmname+'.pt')

    # val_dataset_1, max_num_nodes, input_dim, assign_input_dim = \
    #             cross_val.prepare_val_data_validation(graphs, args, max_nodes=args.max_nodes)
    # #model_dict = torch.load('model/'+args.bmname)
    # #checkpoint = torch.load('model/'+args.bmname+'.pt')
    # model_dict = torch.load('model/dolphins_SI_z0.5_m100')
    # checkpoint = torch.load('model/dolphins_SI_z0.5_m100.pt')
    # model_dict.load_state_dict(checkpoint)
    #
    # model_dict_class0 = torch.load('model/dolphins_SI_z0.5_class0')
    # checkpoint_class0 = torch.load('model/dolphins_SI_z0.5_class0.pt')
    # model_dict_class0.load_state_dict(checkpoint_class0)
    # eva_result, eva_labels, eva_preds, eva_jordan_center = evaluate_class(val_dataset_1, model_dict, model_dict_class0, args)
    # print("result:",eva_result)
    # print("labels:", eva_labels)
    # print("preds:", eva_preds)
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
                        #dataset='syn1v2',
                        dataset='wormpro72',
                        max_nodes=100,
                        cuda='cpu',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        #num_epochs=1000,
                        num_epochs=1,
                        train_ratio=0.9,
                        test_ratio=0.1,
                        #num_workers=1,
                        num_workers=0,#调试train时，要改为0，本来是1
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=20,
                        num_gc_layers=3,
                        dropout=0.3,
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
    #writer = None

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        benchmark_task_val(prog_args, writer=writer)
        #benchmark_task(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == 'syn1v2':
            syn_community1v2(prog_args, writer=writer)
        if prog_args.dataset == 'syn2hier':
            syn_community2hier(prog_args, writer=writer)

    writer.close()

if __name__ == "__main__":
    main()

