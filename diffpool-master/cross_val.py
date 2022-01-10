import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler
dolphins_class = {0: 0, 1: 0, 2: 1, 3: 2, 4: 1, 5: 0, 6: 0, 7: 4, 8: 4, 9: 0, 10: 4, 11: 4, 12: 3, 13: 3, 14: 4, 15: 4,
                  16: 0, 17: 1, 18: 1, 19: 3, 20: 3, 21: 3, 22: 2, 23: 2, 24: 4, 25: 4, 26: 4, 27: 4, 28: 4, 29: 4,
                  30: 0, 31: 0, 32: 1, 33: 1, 34: 2, 35: 4, 36: 2, 37: 2, 38: 1, 39: 1, 40: 1, 41: 2, 42: 1, 43: 1,
                  44: 1, 45: 1, 46: 1, 47: 2, 48: 2, 49: 4, 50: 4, 51: 4, 52: 2, 53: 2, 54: 2, 55: 4, 56: 1, 57: 3,
                  58: 1, 59: 1, 60: 1, 61: 4}
class0 = {0: 0, 1: 1, 5: 2, 6: 3, 9: 4, 16: 5, 30: 6, 31: 7}
class1 = {2: 0,4: 1,17:2, 18:3, 32:4, 33:5, 38:6, 39:7, 40:8, 42:9, 43:10, 44:11, 45:12, 46:13, 56:14, 58:15, 59:16, 60:17}
class2 = {3:0,  22:1, 23:2, 34:3, 36:4, 37:5, 41:6,  47:7, 48:8, 52:9, 53:10, 54:11}
class3 = {12:0, 13:1, 19:2, 20:3, 21:4,  57:5}
class4 = {7:0, 8:1, 10:2, 11:3, 14:4, 15:5, 24:6, 25:7, 26:8, 27:9, 28:10, 29:11, 35:12,  49:13, 50:14, 51:15, 55:16,  61:17}
# class0 = {value: i for i, value in enumerate(class0)}
# class1 = {value: i for i, value in enumerate(class1)}
# class2 = {value: i for i, value in enumerate(class2)}
# class3 = {value: i for i, value in enumerate(class3)}
# class4 = {value: i for i, value in enumerate(class4)}
def prepare_val_data_init(graphs, args, val_idx, max_nodes=0):
    #graphs = graphs[:1749]
    random.shuffle(graphs)
    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

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
    # return train_dataset_loader, val_dataset_loader, \
    #         dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim
    val_dataset_loader_1 = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)
    return train_dataset_loader, val_dataset_loader, val_dataset_loader_1, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


def prepare_val_data(graphs, args, val_idx, max_nodes=0):

    random.shuffle(graphs)
    print(val_idx)
    #graphs=graphs[:6000]
    val_size = len(graphs) // 10

    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
    #train_graphs_class = []
    # for G in train_graphs:
    #     G.graph['classlabel'] = dolphins_class[G.graph['label']]
        #train_graphs_class.append(G)
    # minibatch
    print('1')
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)  # >=1时多线程读取数据         #label是正常范围0-61
    print('2')
    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)                       #label是正常范围0-61
    val_dataset_loader_1 = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)
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
    for G in train_graphs:                    #以下程序中G的label值被改变
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
    dataset_sampler = GraphSampler(train_graphs_class0, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c0_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('4')
    dataset_sampler = GraphSampler(train_graphs_class1, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c1_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('5')
    dataset_sampler = GraphSampler(train_graphs_class2, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c2_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('6')
    dataset_sampler = GraphSampler(train_graphs_class3, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c3_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('7')
    dataset_sampler = GraphSampler(train_graphs_class4, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c4_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('8')
    dataset_sampler = GraphSampler(train_graphs_class5, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c5_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('9')
    
    dataset_sampler = GraphSampler(train_graphs_class6, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c6_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('10')
    dataset_sampler = GraphSampler(train_graphs_class7, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c7_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('11')
    dataset_sampler = GraphSampler(train_graphs_class8, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c8_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('12')
    dataset_sampler = GraphSampler(train_graphs_class9, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c9_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('13')
    dataset_sampler = GraphSampler(train_graphs_class10, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c10_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('14')
    dataset_sampler = GraphSampler(train_graphs_class11, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c11_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    print('15')
    dataset_sampler = GraphSampler(train_graphs_class12, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c12_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    # dataset_sampler = GraphSampler(train_graphs_class, normalize=False, max_num_nodes=max_nodes,
    #                                features=args.feature_type)
    # train_c_dataset_loader = torch.utils.data.DataLoader(
    #     dataset_sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers)
    return train_dataset_loader, train_c0_dataset_loader, train_c1_dataset_loader,\
           train_c2_dataset_loader, train_c3_dataset_loader, train_c4_dataset_loader, \
           train_c5_dataset_loader, train_c6_dataset_loader, train_c7_dataset_loader, \
           train_c8_dataset_loader, train_c9_dataset_loader, train_c10_dataset_loader,\
           train_c11_dataset_loader, train_c12_dataset_loader, \
           val_dataset_loader, \
            val_dataset_loader_1, dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim
def prepare_val_data_c10(graphs, args, val_idx, max_nodes=0):

    random.shuffle(graphs)
    print(val_idx)
    #graphs=graphs[:10000]
    val_size = len(graphs) // 10

    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
    #train_graphs_class = []
    # for G in train_graphs:
    #     G.graph['classlabel'] = dolphins_class[G.graph['label']]
        #train_graphs_class.append(G)
    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)  # >=1时多线程读取数据         #label是正常范围0-61
    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)                       #label是正常范围0-61
    val_dataset_loader_1 = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)
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

    for G in train_graphs:                    #以下程序中G的label值被改变
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

    dataset_sampler = GraphSampler(train_graphs_class0, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c0_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    dataset_sampler = GraphSampler(train_graphs_class1, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c1_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    dataset_sampler = GraphSampler(train_graphs_class2, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c2_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(train_graphs_class3, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c3_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(train_graphs_class4, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c4_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(train_graphs_class5, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c5_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    dataset_sampler = GraphSampler(train_graphs_class6, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c6_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    dataset_sampler = GraphSampler(train_graphs_class7, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c7_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(train_graphs_class8, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c8_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(train_graphs_class9, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c9_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    # dataset_sampler = GraphSampler(train_graphs_class, normalize=False, max_num_nodes=max_nodes,
    #                                features=args.feature_type)
    # train_c_dataset_loader = torch.utils.data.DataLoader(
    #     dataset_sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers)
    return train_dataset_loader, train_c0_dataset_loader, train_c1_dataset_loader,\
           train_c2_dataset_loader, train_c3_dataset_loader, train_c4_dataset_loader, \
           train_c5_dataset_loader, train_c6_dataset_loader, train_c7_dataset_loader, \
           train_c8_dataset_loader, train_c9_dataset_loader,  \
           val_dataset_loader, \
            val_dataset_loader_1, dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


def prepare_val_data_c6(graphs, args, val_idx, max_nodes=0):

    random.shuffle(graphs)
    print(val_idx)
    #graphs=graphs[:10000]
    val_size = len(graphs) // 10

    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
    #train_graphs_class = []
    # for G in train_graphs:
    #     G.graph['classlabel'] = dolphins_class[G.graph['label']]
        #train_graphs_class.append(G)
    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)  # >=1时多线程读取数据         #label是正常范围0-61
    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)                       #label是正常范围0-61
    val_dataset_loader_1 = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)
    train_graphs_class0 = []
    train_graphs_class1 = []
    train_graphs_class2 = []
    train_graphs_class3 = []
    train_graphs_class4 = []
    train_graphs_class5 = []
    # train_graphs_class6 = []
    # train_graphs_class7 = []
    # train_graphs_class8 = []
    # train_graphs_class9 = []

    for G in train_graphs:                    #以下程序中G的label值被改变
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
        # elif G.graph['classlabel'] == 6:
        #     train_graphs_class6.append(G)
        # elif G.graph['classlabel'] == 7:
        #     train_graphs_class7.append(G)
        # elif G.graph['classlabel'] == 8:
        #     train_graphs_class8.append(G)   ###这一行也要改
        # elif G.graph['classlabel'] == 9:
        #     train_graphs_class9.append(G)       ###这一行也要改

    dataset_sampler = GraphSampler(train_graphs_class0, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c0_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    dataset_sampler = GraphSampler(train_graphs_class1, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c1_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    dataset_sampler = GraphSampler(train_graphs_class2, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c2_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(train_graphs_class3, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c3_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(train_graphs_class4, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c4_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(train_graphs_class5, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_c5_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    return train_dataset_loader, train_c0_dataset_loader, train_c1_dataset_loader,\
           train_c2_dataset_loader, train_c3_dataset_loader, train_c4_dataset_loader, \
           train_c5_dataset_loader, \
           val_dataset_loader, \
            val_dataset_loader_1, dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

