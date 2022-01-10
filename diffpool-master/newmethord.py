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
import util


def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
        h0 = Variable(data['feats'].float()).to(device)  # cuda改
        labels.append(data['classlabel'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改

        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
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


def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
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


def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):
    writer_batch_idx = [0, 3, 6, 9]

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
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)  # cuda改
            label = Variable(data['classlabel'].long()).to(device)  # cuda改
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改

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
        # if val_dataset is not None:
        #     val_result = evaluate(val_dataset, model, args, name='Validation')
        #     val_accs.append(val_result['acc'])
        # if val_result['acc'] > best_val_result['acc'] - 1e-7:
        #     best_val_result['acc'] = val_result['acc']
        #     best_val_result['epoch'] = epoch
        #     best_val_result['loss'] = avg_loss
        # if test_dataset is not None:
        #     test_result = evaluate(test_dataset, model, args, name='Test')
        #     test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            #writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
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
    plt.savefig(gen_train_plt_name(args), dpi=600)  ########报错
    plt.close()
    matplotlib.style.use('default')

    return model, val_accs


def prepare_data(graphs, args, test_graphs=None, max_nodes=0):
    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
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
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
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

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
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
            assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
        print('Method: base-set2set')
        model = encoders.GcnSet2SetEncoder(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()
    else:
        print('Method: base')
        model = encoders.GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)
    evaluate(test_dataset, model, args, 'Validation')

def benchmark_task_my(graphs, args, writer=None, feat='node-label',flag=1):

    print('Using node labels')
    for G in graphs:
        for u in G.nodes():
            util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    # val_size = len(graphs) // 10
    # train_graphs = graphs[:1 * val_size]
    # train_graphs = train_graphs + graphs[2 * val_size :]
    # val_graphs = graphs[1*val_size: (1+1)*val_size]
    # print('Num training graphs: ', len(train_graphs),
    #       '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(graphs, normalize=False, max_num_nodes=args.max_nodes,
            features=args.feature_type)
    print("graphsampler")
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)
    print("dataloader")
    model = encoders.GcnEncoderGraph(
        dataset_sampler.feat_dim, args.hidden_dim, args.output_dim, args.num_classes,
        args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()
    print("model")
    if flag == 1:
        _, val_accs = train(train_dataset_loader, model, args, val_dataset=None, test_dataset=None,
                            writer=writer)
        print("model:", model)
        # print(model.state_dict()) # 参数
        torch.save(model.state_dict(), 'model/'+args.bmname+'_Mt.pt')
    elif flag == 2:
        #args.bmname = 'food500_SI_z0.1_m10_rd1'
        checkpoint = torch.load('model/'+args.bmname+'_Mt.pt')
        model.load_state_dict(checkpoint)
        _, val_accs = train(train_dataset_loader, model, args, val_dataset=None, test_dataset=None,
                            writer=writer)
        torch.save(model.state_dict(), 'model/'+args.bmname+'_Mt.pt')
        print("model:", model)
    elif flag == 3:
        checkpoint = torch.load('model/' + args.bmname + '_Mt.pt')
        model.load_state_dict(checkpoint)
        result = evaluate(train_dataset_loader, model, args, 'Validation')
        print("result:", result)
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

def sec_class(graphs, args, writer=None, feat='node-label',flag=1):
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
    print('Using node labels')
    for G in graphs:
        for u in G.nodes():
            util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
    train_graphs_class0 = []
    train_graphs_class1 = []
    train_graphs_class2 = []
    train_graphs_class3 = []
    train_graphs_class4 = []
    train_graphs_class5 = []
    train_graphs_class6 = []
    train_graphs_class7 = []
    train_graphs_class8 = []
    train_graphs_class9 = []
    train_graphs_class10 = []
    train_graphs_class11 = []
    train_graphs_class12 = []
    for G in graphs:                    #以下程序中G的label值被改变
        if G.graph['classlabel'] == 0:
            train_graphs_class0.append(G)
        elif G.graph['classlabel'] == 1:
            train_graphs_class1.append(G)
        elif G.graph['classlabel'] == 2:
            train_graphs_class2.append(G)
        elif G.graph['classlabel'] == 3:
            train_graphs_class3.append(G)
        elif G.graph['classlabel'] == 4:
            train_graphs_class4.append(G)
        elif G.graph['classlabel'] == 5:
            train_graphs_class5.append(G)
        elif G.graph['classlabel'] == 6:
            train_graphs_class6.append(G)
        elif G.graph['classlabel'] == 7:
            train_graphs_class7.append(G)
        elif G.graph['classlabel'] == 8:
            train_graphs_class8.append(G)   ###这一行也要改
        elif G.graph['classlabel'] == 9:
            train_graphs_class9.append(G)       ###这一行也要改
        elif G.graph['classlabel'] == 10:
            train_graphs_class10.append(G)    ###这一行也要改
        elif G.graph['classlabel'] == 11:
            train_graphs_class11.append(G)      ###这一行也要改
        elif G.graph['classlabel'] == 12:
            train_graphs_class12.append(G)     ###这一行也要改
    print('3')
    dataset_sampler = GraphSampler(train_graphs_class0, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c0_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('4')
    dataset_sampler = GraphSampler(train_graphs_class1, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c1_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('5')
    dataset_sampler = GraphSampler(train_graphs_class2, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c2_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('6')
    dataset_sampler = GraphSampler(train_graphs_class3, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c3_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('7')
    dataset_sampler = GraphSampler(train_graphs_class4, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c4_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('8')
    dataset_sampler = GraphSampler(train_graphs_class5, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c5_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('9')

    dataset_sampler = GraphSampler(train_graphs_class6, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c6_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('10')
    dataset_sampler = GraphSampler(train_graphs_class7, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c7_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('11')
    dataset_sampler = GraphSampler(train_graphs_class8, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c8_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('12')
    dataset_sampler = GraphSampler(train_graphs_class9, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c9_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('13')
    dataset_sampler = GraphSampler(train_graphs_class10, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c10_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('14')
    dataset_sampler = GraphSampler(train_graphs_class11, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c11_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('15')
    dataset_sampler = GraphSampler(train_graphs_class12, normalize=False, max_num_nodes=args.max_nodes,
                                   features=args.feature_type)
    train_c12_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    num_class = [29 , 78 , 3 , 18 , 30 , 5 , 72 , 6 , 72 , 13 , 32 , 100 , 42] #food500
    model_c = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    for j in range(len(num_class)):
        model_c[j] = encoders.GcnEncoderGraph(
            dataset_sampler.feat_dim, args.hidden_dim, args.output_dim, num_class[j],
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改

    if flag == 1:
        m_c0, train_accs_c0 = class_train(train_c0_dataset_loader, model_c[0], args)
        print("model:", m_c0)
        # print(model.state_dict()) # 参数
        torch.save(m_c0.state_dict(), 'model/' + args.bmname + '_m_c0_' + '.pt')
        m_c1, train_accs_c1 = class_train(train_c1_dataset_loader, model_c[1], args)
        torch.save(m_c1.state_dict(), 'model/' + args.bmname + '_m_c1_' + '.pt')
        m_c2, train_accs_c2 = class_train(train_c2_dataset_loader, model_c[2], args)
        torch.save(m_c2.state_dict(), 'model/' + args.bmname + '_m_c2_' + '.pt')
        m_c3, train_accs_c3 = class_train(train_c3_dataset_loader, model_c[3], args)
        torch.save(m_c3.state_dict(), 'model/' + args.bmname + '_m_c3_' + '.pt')
        m_c4, train_accs_c4 = class_train(train_c4_dataset_loader, model_c[4], args)
        torch.save(m_c4.state_dict(), 'model/' + args.bmname + '_m_c4_' + '.pt')
        m_c5, train_accs_c5 = class_train(train_c5_dataset_loader, model_c[5], args)
        m_c6, train_accs_c6 = class_train(train_c6_dataset_loader, model_c[6], args)
        m_c7, train_accs_c7 = class_train(train_c7_dataset_loader, model_c[7], args)
        m_c8, train_accs_c8 = class_train(train_c8_dataset_loader, model_c[8], args)
        m_c9, train_accs_c9 = class_train(train_c9_dataset_loader, model_c[9], args)
        m_c10, train_accs_c10 = class_train(train_c10_dataset_loader, model_c[10], args)
        m_c11, train_accs_c11 = class_train(train_c11_dataset_loader, model_c[11], args)
        m_c12, train_accs_c12 = class_train(train_c12_dataset_loader, model_c[12], args)
        torch.save(m_c5.state_dict(), 'model/' + args.bmname + '_m_c5_' + '.pt')
        torch.save(m_c6.state_dict(), 'model/' + args.bmname + '_m_c6_' + '.pt')
        torch.save(m_c7.state_dict(), 'model/' + args.bmname + '_m_c7_' + '.pt')
        torch.save(m_c8.state_dict(), 'model/' + args.bmname + '_m_c8_' + '.pt')
        torch.save(m_c9.state_dict(), 'model/' + args.bmname + '_m_c9_' + '.pt')
        torch.save(m_c10.state_dict(), 'model/' + args.bmname + '_m_c10_' + '.pt')
        torch.save(m_c11.state_dict(), 'model/' + args.bmname + '_m_c11_' + '.pt')
        torch.save(m_c12.state_dict(), 'model/' + args.bmname + '_m_c12_' + '.pt')
    elif flag == 2:
        args.bmname = 'food500_SI_z0.1_m10_rd1'
        model_c[0].load_state_dict(torch.load('model/' + args.bmname + '_m_c0_' + '.pt'))
        m_c0, train_accs_c0 = class_train(train_c0_dataset_loader, model_c[0], args)
        torch.save(m_c0.state_dict(), 'model/' + args.bmname + '_m_c0_' + '.pt')
        print("model:", m_c0)
        model_c[1].load_state_dict(torch.load('model/' + args.bmname + '_m_c1_' + '.pt'))
        m_c1, train_accs_c1 = class_train(train_c1_dataset_loader, model_c[1], args)
        torch.save(m_c1.state_dict(), 'model/' + args.bmname + '_m_c1_' + '.pt')
        model_c[2].load_state_dict(torch.load('model/' + args.bmname + '_m_c2_' + '.pt'))
        m_c2, train_accs_c2 = class_train(train_c2_dataset_loader, model_c[2], args)
        torch.save(m_c2.state_dict(), 'model/' + args.bmname + '_m_c2_' + '.pt')
        model_c[3].load_state_dict(torch.load('model/' + args.bmname + '_m_c3_' + '.pt'))
        m_c3, train_accs_c3 = class_train(train_c3_dataset_loader, model_c[3], args)
        torch.save(m_c3.state_dict(), 'model/' + args.bmname + '_m_c3_' + '.pt')
        model_c[4].load_state_dict(torch.load('model/' + args.bmname + '_m_c4_' + '.pt'))
        m_c4, train_accs_c4 = class_train(train_c4_dataset_loader, model_c[4], args)
        torch.save(m_c4.state_dict(), 'model/' + args.bmname + '_m_c4_' + '.pt')
        model_c[5].load_state_dict(torch.load('model/' + args.bmname + '_m_c5_' + '.pt'))
        m_c5, train_accs_c5 = class_train(train_c5_dataset_loader, model_c[5], args)
        torch.save(m_c5.state_dict(), 'model/' + args.bmname + '_m_c5_' + '.pt')
        model_c[6].load_state_dict(torch.load('model/' + args.bmname + '_m_c6_' + '.pt'))
        m_c6, train_accs_c6 = class_train(train_c6_dataset_loader, model_c[6], args)
        torch.save(m_c6.state_dict(), 'model/' + args.bmname + '_m_c6_' + '.pt')
        model_c[7].load_state_dict(torch.load('model/' + args.bmname + '_m_c7_' + '.pt'))
        m_c7, train_accs_c7 = class_train(train_c7_dataset_loader, model_c[7], args)
        torch.save(m_c7.state_dict(), 'model/' + args.bmname + '_m_c7_' + '.pt')
        model_c[8].load_state_dict(torch.load('model/' + args.bmname + '_m_c8_' + '.pt'))
        m_c8, train_accs_c8 = class_train(train_c8_dataset_loader, model_c[8], args)
        torch.save(m_c8.state_dict(), 'model/' + args.bmname + '_m_c8_' + '.pt')
        model_c[9].load_state_dict(torch.load('model/' + args.bmname + '_m_c9_' + '.pt'))
        m_c9, train_accs_c9 = class_train(train_c9_dataset_loader, model_c[9], args)
        torch.save(m_c9.state_dict(), 'model/' + args.bmname + '_m_c9_' + '.pt')
        model_c[10].load_state_dict(torch.load('model/' + args.bmname + '_m_c10_' + '.pt'))
        m_c10, train_accs_c10 = class_train(train_c10_dataset_loader, model_c[10], args)
        torch.save(m_c10.state_dict(), 'model/' + args.bmname + '_m_c10_' + '.pt')
        model_c[11].load_state_dict(torch.load('model/' + args.bmname + '_m_c11_' + '.pt'))
        m_c11, train_accs_c11 = class_train(train_c11_dataset_loader, model_c[11], args)
        torch.save(m_c11.state_dict(), 'model/' + args.bmname + '_m_c11_' + '.pt')
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

def evaluate_test(args, writer=None, feat='node-label'):
    graphs = load_data.read_graphfile('data', args.bmname, max_nodes=args.max_nodes)
    random.shuffle(graphs)
    graphs = graphs[:1000]
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
    print('Using node labels')
    for G in graphs:
        for u in G.nodes():
            util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
    num_class = [29 , 78 , 3 , 18 , 30 , 5 , 72 , 6 , 72 , 13 , 32 , 100 , 42] #food500
    model_c = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    dataset_sampler = GraphSampler(graphs, normalize=False, max_num_nodes=args.max_nodes,
            features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers)
    model_fir = encoders.GcnEncoderGraph(
        dataset_sampler.feat_dim, args.hidden_dim, args.output_dim, args.num_classes,
        args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()
    checkpoint = torch.load('model/' + 'food500_nSI_z0.1_m20_11' + '_Mt.pt')
    model_fir.load_state_dict(checkpoint)
    print("model_fir:",model_fir)
    for j in range(len(num_class)):
        model_c[j] = encoders.GcnEncoderGraph(
            dataset_sampler.feat_dim, args.hidden_dim, args.output_dim, num_class[j],
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
    # checkpoint = torch.load('model/' + 'food500_SI_z0.1_m30_c0' + '.pt')
    # model_c[0].load_state_dict(checkpoint)
    # print(0, model_c[0])
    # checkpoint = torch.load('model/' + 'food500_SI_z0.1_m30_c1' + '.pt')
    # model_c[1].load_state_dict(checkpoint)
    # print(1, model_c[1])
    for i in range(0, args.num_classes):
        checkpoint = torch.load('model/' + 'food500_nSI_z0.1_m20_c' + str(i)+'.pt')
        model_c[i].load_state_dict(checkpoint)
        print(i, model_c[i])
    # model_c[0].load_state_dict((torch.load('model/'+args.bmname+'_m_c0_'+'.pt')))
    # print("0:",model_c[0])
    # model_c[1].load_state_dict((torch.load('model/'+args.bmname+'_m_c1_'+'.pt')))
    # print("1:",model_c[1])
    eva_result, eva_labels, eva_realpreds, eva_jordan_center, eva_unbet, eva_discen, eva_dynage = \
        evaluate_class_pro(test_dataset_loader,model_fir,model_c[0],model_c[1],model_c[2],model_c[3],model_c[4],
                       model_c[5],model_c[6],model_c[7],model_c[8],model_c[9],model_c[10],model_c[11],model_c[12])
    print("result:",eva_result)
    #print("labels:", eva_labels)
    #print("preds:", eva_preds)
    print('labels:', eva_labels, 'len(labels):', len(eva_labels))
    #print('realpreds_c:', eva_realpreds, 'len(realpreds_c):', len(eva_realpreds))
    #print('jc:', eva_jordan_center, 'len(jc):', len(eva_jordan_center))
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
                       ):

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
    labels_init = []  # 原始classlabels对应的labels
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
        labels_init.append(class_reverse[data['classlabel'].item()][data['label'].item()])
        for i in range(len(model_dic)):
            #print(i)
            #print(indices.cpu().data.numpy())
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
    labels_init = np.hstack(labels_init)
    print("labels_init:", labels_init)
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
        b = read_dic[labels_init[i]][jordan_center[i]]
        h = read_dic[labels_init[i]][unbet[i]]     ###
        k = read_dic[labels_init[i]][discen[i]]    ###
        q = read_dic[labels_init[i]][dynage[i]]    ###
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
def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    example_node = util.node_dict(graphs[0])[0]

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:   #####
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(10):
        train_dataset, val_dataset,val_1, max_num_nodes, input_dim, assign_input_dim = \
            cross_val.prepare_val_data_init(graphs, args, i, max_nodes=args.max_nodes)     #初始分数据的方式
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
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改
        else:
            print('Method: base')
            model = encoders.GcnEncoderGraph(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda()改

        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))
# def newmethord(graphs, args, writer=None, feat='node-label'):
#     print('Number of graphs: ', len(graphs))
#     print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
#     print('Max, avg, std of graph size: ',
#             max([G.number_of_nodes() for G in graphs]), ', '
#             "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
#             "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
#     dataset_sampler = GraphSampler(graphs, normalize=False, max_num_nodes=args.max_nodes,
#             features=args.feature_type)
#     train_dataset_loader = torch.utils.data.DataLoader(
#             dataset_sampler,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=args.num_workers)
#     model_t1 = encoders.GcnEncoderGraph(
#         dataset_sampler.feat_dim, args.hidden_dim, args.output_dim, args.num_classes,
#         args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).to(device)  # cuda() #2000张训练集训练模型
#
#     train_losses,train_acc = [],[]
#     val_losses, val_acc = [],[]
#     for epoch in range(args.epochs):
#         epoch_loss,epoch_acc = fit(epoch,model_t1,train_dataset_loader,phase='training')
#         #val_epoch_loss, val_epoch_acc = fit(epoch,model_t1,test_loader,phase='validation')
#         train_losses.append(epoch_loss)
#         train_acc.append(epoch_acc)
#     plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
# def fit(epoch,model,dataset,phase='training',volatile=False):
#
#     runnung_loss=0.0
#     running_correct = 0
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
#     total_time = 0
#     avg_loss = 0.0
#     if phase == 'training':
#         model.train()
#     if phase == 'validation':
#         model.eval()
#         volatile = True
#     print('Epoch: ', epoch)
#     for batch_idx, data in enumerate(dataset):
#         begin_time = time.time()
#         model.zero_grad()
#         adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
#         h0 = Variable(data['feats'].float(), requires_grad=False).to(device)  # cuda改
#         label = Variable(data['classlabel'].long()).to(device)  # cuda改
#         batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
#         assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改
#
#         ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
#         loss = model.loss(ypred, label)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), '2.0')
#         optimizer.step()
#         iter += 1
#         avg_loss += loss
#         # if iter % 20 == 0:
#         #    print('Iter: ', iter, ', loss: ', loss.data[0])
#         elapsed = time.time() - begin_time
#         total_time += elapsed
#     avg_loss /= batch_idx + 1
#     print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
#     return loss,acc
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
                                 help='ratio of number of nodes in consecutive layers')   #连续层中节点数的比例
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
                        dataset='wormpro7',
                        max_nodes=1000,
                        cuda='cpu',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        # num_epochs=1000,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=0,
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
         graphs = load_data.read_graphfile('data', prog_args.bmname, max_nodes=prog_args.max_nodes)
         graphs = graphs[:1000]
         random.shuffle(graphs)
         #benchmark_task_val(prog_args, writer=writer) #原始训练方法
         #newmethord(graphs[:2000], prog_args, writer=writer)
         print('prog_args.bmname:',prog_args.bmname)
         if prog_args!='food500_nSI_z0.1_m20_11':
             prog_args.bmname='food500_nSI_z0.1_m20_11'
         print('prog_args.bmname:',prog_args.bmname)
         benchmark_task_my(graphs,prog_args,writer=writer,flag=2)  #第一级分类的训练程序，1为第一次训练，2为之后的几次训练，3为验证准确率
         #benchmark_task_my(graphs[2000:4000],prog_args,writer=writer,flag=2)

         #benchmark_task_my(graphs[9000:],prog_args,writer=writer,flag=3)  #testdataset部分可以改，改成边训练边验证
         #sec_class(graphs,prog_args,writer=writer,flag=1)   #先将13类数据分开，再训练得到第二级每一类的模型，不需要这么麻烦
         #evaluate_test(prog_args,writer)  #现在每次运行则重新打乱，train与test是混合的是不对的。
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)

    writer.close()


if __name__ == "__main__":
    main()