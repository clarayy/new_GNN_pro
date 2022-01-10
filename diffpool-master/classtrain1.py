#从原程序中开始写的
#只写关键部分
#先不考虑其他方法的实验
#必要时可以删除没用的feature参数

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
import load_data_source
import util
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                        dataset='BA500_SI_r3',
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
def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
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

    writer.close()
def benchmark_task_val(args,writer=None,feat='node-label'):
    all_vals = []
    # graphs0 = load_data_source.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes) #用graph_label_class.txt
    # featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    # for G in graphs0:
    #     featgen_const.gen_node_features(G)
    # train_c, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
    #     cross_val.prepare_val_data_init(graphs0, args, 1, max_nodes=args.max_nodes)   #0-4标签，0-4标签，62,10,10

    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    for G in graphs:
        featgen_const.gen_node_features(G)

    train_dataset, train_c0, train_c1, train_c2, train_c3, train_c4, val_dataset, val_dataset_1, max_num_nodes,\
        input_dim, assign_input_dim = cross_val.prepare_val_data(graphs, args, 1, max_nodes=args.max_nodes)   #这一部分val_dataset与上一部分不一样，数据泄漏导致结果过拟合

    print('Method: base,first floor')
    model = encoders.GcnEncoderGraph(
        input_dim, args.hidden_dim, args.output_dim, args.num_classes,
        args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
    print(model)
    _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                        writer=writer)
    print(_)
    print(val_accs)
    all_vals.append(np.array(val_accs))
    print(all_vals)
    # for i in range(1):
    #     print('i=',i)
    #     model_c0 = encoders.GcnEncoderGraph(
    #         input_dim, args.hidden_dim, args.output_dim, 8,
    #         args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改   num_classes=8时报错，分为8类标签不能大于8,改为62好像又不对
    #     print(model_c0)
    #     _c0, val_accs_c0 = fit(train_c0, model_c0, args, val_dataset=val_dataset, test_dataset=None)
    # #print(_c0)
    # print(val_accs_c0)
    # all_vals.append(np.array(val_accs_c0))
    # all_vals = np.vstack(all_vals)
    # print(all_vals)
    #
    # model_c1 = encoders.GcnEncoderGraph(
    #     input_dim, args.hidden_dim, args.output_dim, 18,
    #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改   18类
    # print(model_c1)
    # _c1, val_accs_c1 = fit(train_c1, model_c1, args, val_dataset=val_dataset, test_dataset=None)
    # print(_c1)
    # print(val_accs_c1)
    #
    # model_c2 = encoders.GcnEncoderGraph(
    #     input_dim, args.hidden_dim, args.output_dim, 12,
    #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改   12类
    # print(model_c2)
    # _c2, val_accs_c2 = fit(train_c2, model_c2, args, val_dataset=val_dataset, test_dataset=None)
    # print(_c2)
    # print(val_accs_c2)
    #
    # model_c3 = encoders.GcnEncoderGraph(
    #     input_dim, args.hidden_dim, args.output_dim, 6,
    #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改   6类
    # print(model_c3)
    # _c3, val_accs_c3 = fit(train_c3, model_c3, args, val_dataset=val_dataset, test_dataset=None)
    # print(_c3)
    # print(val_accs_c3)
    #
    # model_c4 = encoders.GcnEncoderGraph(
    #     input_dim, args.hidden_dim, args.output_dim, 18,
    #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改   18类
    # print(model_c4)
    # _c4, val_accs_c4 = fit(train_c4, model_c4, args, val_dataset=val_dataset, test_dataset=None)
    # print(_c4)
    # print(val_accs_c4)
    #总的验证程序
   # model.eval()
    model_c0.eval()
    # model_c1.eval()
    # model_c2.eval()
    # model_c3.eval()
    # model_c4.eval()
    labels = []
    #
    preds_c0 = []
    preds_c1 = []
    # preds_c2 = []
    # preds_c3 = []
    # preds_c4 = []
    # realpreds_c0 = []
    # class0_dataset = []
    # result = evaluate(val_dataset, model, args, name='Validation', max_num_examples=100)
    # print("model val result:",result)
    # for batch_idx, data in enumerate(val_dataset_1):
    #     adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda()改
    #     h0 = Variable(data['feats'].float()).to(device)  # cuda()改
    #     labels.append(data['label'].long().numpy())
    #     batch_num_nodes = data['num_nodes'].int().numpy()
    #     assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda()改
    #     classlabels = data['classlabel'].long().numpy()
    #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
    #     _, indices = torch.max(ypred, 1)#1表示返回ypred中每一行中的最大值及其索引
    #     preds = indices.cpu().data.numpy()
    #
    #     if preds == classlabels:
    #         if preds == 0:
    #             ypred_c0 = model_c0(h0, adj, batch_num_nodes, assign_x=assign_input)
    #             _, indices = torch.max(ypred_c0, 1)
    #             preds_c0.append(indices.cpu().data.numpy())
    #             continue
    #         # elif preds == 1:
    #         #     ypred_c1 = model_c1(h0, adj, batch_num_nodes, assign_x=assign_input)
    #         #     _, indices = torch.max(ypred_c1, 1)
    #         #     preds_c1.append(indices.cpu().data.numpy())
    #         # elif preds == 2:
    #         #     ypred_c2 = model_c2(h0, adj, batch_num_nodes, assign_x=assign_input)
    #         #     _, indices = torch.max(ypred_c2, 1)
    #         #     preds_c2.append(indices.cpu().data.numpy())
    #         # elif preds == 3:
    #         #     ypred_c3 = model_c3(h0, adj, batch_num_nodes, assign_x=assign_input)
    #         #     _, indices = torch.max(ypred_c3, 1)
    #         #     preds_c3.append(indices.cpu().data.numpy())
    #         # elif preds == 4:
    #         #     ypred_c4 = model_c4(h0, adj, batch_num_nodes, assign_x=assign_input)
    #         #     _, indices = torch.max(ypred_c4, 1)
    #         #     preds_c4.append(indices.cpu().data.numpy())
    #     else:
    #         continue

        # 1表示返回ypred中每一行中的最大值及其索引

        #
        # ypred_c1 = model_c1(h0, adj, batch_num_nodes, assign_x=assign_input)
        # _, indices = torch.max(ypred_c1, 1)  # 1表示返回ypred中每一行中的最大值及其索引
        # preds_c1.append(indices.cpu().data.numpy())
        #
        # ypred_c2 = model_c2(h0, adj, batch_num_nodes, assign_x=assign_input)
        # _, indices = torch.max(ypred_c2, 1)  # 1表示返回ypred中每一行中的最大值及其索引
        # preds_c2.append(indices.cpu().data.numpy())
        #
        # ypred_c3 = model_c3(h0, adj, batch_num_nodes, assign_x=assign_input)
        # _, indices = torch.max(ypred_c3, 1)  # 1表示返回ypred中每一行中的最大值及其索引
        # preds_c3.append(indices.cpu().data.numpy())
        #
        # ypred_c4 = model_c4(h0, adj, batch_num_nodes, assign_x=assign_input)
        # _, indices = torch.max(ypred_c4, 1)  # 1表示返回ypred中每一行中的最大值及其索引
        # preds_c4.append(indices.cpu().data.numpy())

    #preds = np.hstack(preds)
    #preds_c0 = np.hstack(preds_c0)
    # preds_c1 = np.hstack(preds_c1)
    # preds_c2 = np.hstack(preds_c2)
    # preds_c3 = np.hstack(preds_c3)
    # preds_c4 = np.hstack(preds_c4)

    labels = np.hstack(labels)
    # class0 = {0: 0, 1: 1, 5: 2, 6: 3, 9: 4, 16: 5, 30: 6, 31: 7}
    # for i in preds_c0:
    #     class0_reverse = {v: k for k, v in class0.items()}
    #     realpreds_c0.append(class0_reverse[i])

    #print('preds:', preds)
    # #print('preds_c0:', preds_c0)
    # print('preds_c1:', preds_c1)
    # print('preds_c2:', preds_c2)
    # print('preds_c3:', preds_c3)
    # print('preds_c4:', preds_c4)

    print("labels:", labels)
    # print(realpreds_c0)
    # count = 0
    # for i in range(len(labels)):
    #     if labels[i] == realpreds_c0[i]:
    #         count = count+1
    # print(count/len(labels))
    # labels = np.hstack(labels)
    # preds = np.hstack(preds)
    #     for i in preds:
    #         if preds[i] == labels[i]:
    #             if preds[i] == 0:
    #                 class0_dataset.append(data[i])

#fit和evaluate_class是一对，分开写的话出现问题，label_dim如果一样的话怎么区分模型呢？因此产生fitpro
def fit(dataset,model,args,val_dataset=None,test_dataset=None,
        mask_nodes=True):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda()改
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)  # cuda()改
            label = Variable(data['label'].long()).to(device)  # cuda()改
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda()改

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            loss = model.loss(ypred, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        avg_loss /= batch_idx + 1
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        #val_result = evaluate_class(val_dataset, model, args, name='Validation')#val_dataset的标签是0-61,不能用上边的evaluate
        #val_accs.append(val_result['acc'])
    #     if val_result['acc'] > best_val_result['acc'] - 1e-7:
    #         best_val_result['acc'] = val_result['acc']
    #         best_val_result['epoch'] = epoch
    #         best_val_result['loss'] = avg_loss
    #     print('Best val result: ', best_val_result)
    #     best_val_epochs.append(best_val_result['epoch'])
    #     best_val_accs.append(best_val_result['acc'])
    # matplotlib.style.use('seaborn')
    # plt.switch_backend('agg')
    # plt.figure()
    # plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    # plt.plot(best_val_epochs, best_val_accs, 'bo')
    # plt.legend(['train', 'val'])
    # plt.savefig(gen_train_plt_name(args), dpi=600)
    # plt.close()
    # matplotlib.style.use('default')

    return model, train_accs
def fitpro(dataset,model,args,mask_nodes=True):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda()改
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)  # cuda()改
            label = Variable(data['label'].long()).to(device)  # cuda()改
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda()改

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            loss = model.loss(ypred, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        avg_loss /= batch_idx + 1
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result = evaluate_class(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):
    writer_batch_idx = [0, 3, 6, 9]
    #print(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
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
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda()改
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)  # cuda()改
            if model.label_dim == 5:
                label = Variable(data['classlabel'].long()).to(device)
            else:
                label = Variable(data['label'].long()).to(device)  # cuda()改
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda()改

            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if not args.method == 'soft-assign' or not args.linkpred:
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
            total_time += elapsed

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
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
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

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, 'bo', test_epochs, test_accs, 'go')
        plt.legend(['train', 'val', 'test'])
    else:
        plt.plot(best_val_epochs, best_val_accs, 'bo')
        plt.legend(['train', 'val'])
    plt.savefig(gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    #print(model)
    model.eval()

    labels = []
    preds = []

    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda()改
        h0 = Variable(data['feats'].float()).to(device)  # cuda()改
        if model.label_dim == 5:
            labels.append(data['classlabel'].long().numpy())
        #elif model.label_dim == 8:
        else:
            labels.append(data['label'].long().numpy())
        #labels.append(data['classlabel'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda()改

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)#1表示返回ypred中每一行中的最大值及其索引
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return result

def evaluate_class(dataset, model, args, name='Validation', max_num_examples=None):
    #print(model)
    dolphins_class = {0: 0, 1: 0, 2: 1, 3: 2, 4: 1, 5: 0, 6: 0, 7: 4, 8: 4, 9: 0, 10: 4, 11: 4, 12: 3, 13: 3, 14: 4,
                      15: 4,
                      16: 0, 17: 1, 18: 1, 19: 3, 20: 3, 21: 3, 22: 2, 23: 2, 24: 4, 25: 4, 26: 4, 27: 4, 28: 4, 29: 4,
                      30: 0, 31: 0, 32: 1, 33: 1, 34: 2, 35: 4, 36: 2, 37: 2, 38: 1, 39: 1, 40: 1, 41: 2, 42: 1, 43: 1,
                      44: 1, 45: 1, 46: 1, 47: 2, 48: 2, 49: 4, 50: 4, 51: 4, 52: 2, 53: 2, 54: 2, 55: 4, 56: 1, 57: 3,
                      58: 1, 59: 1, 60: 1, 61: 4}
    class0 = {0: 0, 1: 1, 5: 2, 6: 3, 9: 4, 16: 5, 30: 6, 31: 7}
    class1 = {2: 0, 4: 1, 17: 2, 18: 3, 32: 4, 33: 5, 38: 6, 39: 7, 40: 8, 42: 9, 43: 10, 44: 11, 45: 12, 46: 13,
              56: 14, 58: 15, 59: 16, 60: 17}
    class2 = {3: 0, 22: 1, 23: 2, 34: 3, 36: 4, 37: 5, 41: 6, 47: 7, 48: 8, 52: 9, 53: 10, 54: 11}
    class3 = {12: 0, 13: 1, 19: 2, 20: 3, 21: 4, 57: 5}
    class4 = {7: 0, 8: 1, 10: 2, 11: 3, 14: 4, 15: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 35: 12, 49: 13,
              50: 14, 51: 15, 55: 16, 61: 17}
    # class0 = {0: 0, 3: 1, 18: 2, 19: 3, 49: 4, 51: 5, 56: 6, 60: 7, 68: 8, 74: 9, 84: 10, 102: 11, 109: 12, 111: 13,
    #           146: 14, 148: 15, 149: 16, 168: 17, 173: 18, 181: 19, 182: 20, 200: 21, 205: 22, 208: 23, 209: 24,
    #           217: 25, 219: 26, 227: 27, 229: 28, 232: 29, 249: 30, 272: 31, 276: 32, 286: 33, 308: 34, 314: 35,
    #           327: 36, 328: 37, 341: 38, 347: 39, 368: 40, 385: 41, 396: 42, 406: 43, 424: 44, 431: 45, 436: 46,
    #           446: 47, 454: 48, 460: 49, 466: 50, 481: 51, 485: 52, 493: 53, 496: 54}
    # class1 = {12: 0, 28: 1, 46: 2, 80: 3, 106: 4, 143: 5, 155: 6, 160: 7, 162: 8, 164: 9, 167: 10, 185: 11, 191: 12,
    #           192: 13, 204: 14, 237: 15, 238: 16, 278: 17, 287: 18, 290: 19, 297: 20, 305: 21, 324: 22, 325: 23,
    #           346: 24, 348: 25, 364: 26, 365: 27, 370: 28, 393: 29, 418: 30, 439: 31, 440: 32}
    # class2 = {2: 0, 7: 1, 8: 2, 17: 3, 29: 4, 30: 5, 32: 6, 35: 7, 44: 8, 62: 9, 85: 10, 91: 11, 108: 12, 113: 13,
    #           124: 14, 166: 15, 170: 16, 172: 17, 174: 18, 176: 19, 177: 20, 183: 21, 187: 22, 193: 23, 194: 24,
    #           198: 25, 202: 26, 206: 27, 207: 28, 215: 29, 218: 30, 225: 31, 230: 32, 243: 33, 252: 34, 254: 35,
    #           257: 36, 262: 37, 273: 38, 281: 39, 282: 40, 296: 41, 302: 42, 303: 43, 307: 44, 318: 45, 335: 46,
    #           355: 47, 359: 48, 362: 49, 400: 50, 404: 51, 410: 52, 425: 53, 427: 54, 434: 55, 448: 56, 452: 57,
    #           470: 58, 480: 59, 486: 60, 487: 61, 499: 62}
    # class3 = {4: 0, 13: 1, 25: 2, 26: 3, 31: 4, 37: 5, 40: 6, 41: 7, 42: 8, 50: 9, 57: 10, 58: 11, 66: 12, 70: 13,
    #           81: 14, 107: 15, 119: 16, 121: 17, 123: 18, 142: 19, 151: 20, 159: 21, 180: 22, 184: 23, 203: 24, 210: 25,
    #           221: 26, 234: 27, 246: 28, 248: 29, 279: 30, 285: 31, 300: 32, 301: 33, 306: 34, 316: 35, 317: 36,
    #           326: 37, 339: 38, 349: 39, 353: 40, 358: 41, 360: 42, 371: 43, 377: 44, 379: 45, 402: 46, 407: 47,
    #           409: 48, 415: 49, 416: 50, 430: 51, 457: 52, 459: 53, 471: 54, 472: 55, 476: 56, 497: 57}
    # class4 = {5: 0, 9: 1, 15: 2, 22: 3, 43: 4, 48: 5, 61: 6, 72: 7, 117: 8, 125: 9, 127: 10, 128: 11, 130: 12, 141: 13,
    #           153: 14, 163: 15, 199: 16, 212: 17, 220: 18, 224: 19, 228: 20, 240: 21, 264: 22, 267: 23, 270: 24,
    #           275: 25, 280: 26, 292: 27, 299: 28, 315: 29, 320: 30, 321: 31, 338: 32, 344: 33, 351: 34, 361: 35,
    #           366: 36, 381: 37, 383: 38, 395: 39, 398: 40, 414: 41, 420: 42, 437: 43, 455: 44, 468: 45, 491: 46,
    #           495: 47}

    model.eval()

    labels = []
    preds = []
    labels_new = []
    labels_newlist = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda()改
        h0 = Variable(data['feats'].float()).to(device)  # cuda()改
        if model.label_dim == 5:
            labels.append(data['classlabel'].long().numpy())
        elif model.label_dim == 8:
            labels.append(data['label'].long().numpy())
            # print(type(labels))   #list
            # print(type(data['label']))    #torch.tensor
            # print(type(data['label'].long()))
            # print(type(data['label'].long().numpy()))    #numpy.ndarray
            labels_new = []
            for i in labels[batch_idx]:
                class0_reverse = {v: k for k, v in class0.items()}
                labels_new.append(class0_reverse[i])
            labels_newlist.append(torch.Tensor(labels_new).long().numpy())
        elif model.label_dim == 18:
            labels.append(data['label'].long().numpy())
            labels_new = []
            for i in labels[batch_idx]:
                class1_reverse = {v: k for k, v in class1.items()}
                labels_new.append(class1_reverse[i])
            labels_newlist.append(torch.Tensor(labels_new).long().numpy())
        elif model.label_dim == 12:
            labels.append(data['label'].long().numpy())
            labels_new = []
            for i in labels[batch_idx]:
                class2_reverse = {v: k for k, v in class2.items()}
                labels_new.append(class2_reverse[i])
            labels_newlist.append(torch.Tensor(labels_new).long().numpy())
        elif model.label_dim == 6:
            labels.append(data['label'].long().numpy())
            labels_new = []
            for i in labels[batch_idx]:
                class3_reverse = {v: k for k, v in class3.items()}
                labels_new.append(class3_reverse[i])
            labels_newlist.append(torch.Tensor(labels_new).long().numpy())
        elif model.label_dim == 48:
            labels.append(data['label'].long().numpy())
            labels_new = []
            for i in labels[batch_idx]:
                class4_reverse = {v: k for k, v in class4.items()}
                labels_new.append(class4_reverse[i])
            labels_newlist.append(torch.Tensor(labels_new).long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda()改

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)#1表示返回ypred中每一行中的最大值及其索引
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels_newlist)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return result

if __name__ == "__main__":
    main()