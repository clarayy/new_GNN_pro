from torch.autograd import Variable
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate_class(dataset, model, model_c0, model_c1,model_c2,model_c3,model_c4,args, name='Validation', max_num_examples=None):
    class0 = {0: 0, 1: 1, 5: 2, 6: 3, 9: 4, 16: 5, 30: 6, 31: 7}
    class1 = {2: 0, 4: 1, 17: 2, 18: 3, 32: 4, 33: 5, 38: 6, 39: 7, 40: 8, 42: 9, 43: 10, 44: 11, 45: 12, 46: 13,
              56: 14, 58: 15, 59: 16, 60: 17}
    class2 = {3: 0, 22: 1, 23: 2, 34: 3, 36: 4, 37: 5, 41: 6, 47: 7, 48: 8, 52: 9, 53: 10, 54: 11}
    class3 = {12: 0, 13: 1, 19: 2, 20: 3, 21: 4, 57: 5}
    class4 = {7: 0, 8: 1, 10: 2, 11: 3, 14: 4, 15: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 35: 12, 49: 13,
              50: 14, 51: 15, 55: 16, 61: 17}
    class0_reverse = {value: i for i, value in class0.items()}
    class1_reverse = {value: i for i, value in class1.items()}
    class2_reverse = {value: i for i, value in class2.items()}
    class3_reverse = {value: i for i, value in class3.items()}
    class4_reverse = {value: i for i, value in class4.items()}
    class_reverse = [class0_reverse, class1_reverse, class2_reverse, class3_reverse, class4_reverse]  #dict的key不支持[]或{},他们是不可哈希的
    model_dic = [model_c0, model_c1, model_c2, model_c3, model_c4]
    model.eval()
    classlabels = []
    preds = []
    jordan_center = []
    labels = [[], [], [], [], []]
    preds_c = [[], [], [], [], []]
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)  # cuda改
        h0 = Variable(data['feats'].float()).to(device)  # cuda改
        classlabels.append(data['classlabel'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).to(device)  # cuda改
        jordan_center.append(data['center'].long().numpy())
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())
        for i in range(5):
            #print(i)
            if indices.cpu().data.numpy() == i:
                model_dic[i].eval()
                labels[i].append(data['label'].long().numpy())
                ypred_c = model_dic[i](h0, adj, batch_num_nodes, assign_input=assign_input)
                _, indices_c = torch.max(ypred_c, 1)
                preds_c[i].append(indices_c.cpu().data.numpy())
    classlabels = np.hstack(classlabels)
    preds = np.hstack(preds)
    jordan_center = np.hstack(jordan_center)
    realpreds_c = [[], [], [], [], []]
    for i in range(5):
        labels[i] = np.hstack(labels[i])
        preds_c[i] = np.hstack(preds_c[i])
        for m in preds_c[i]:
            realpreds_c[i].append(class_reverse[i][m])
        print("labels_c:", i, labels[i])
        print("realpreds_c:", i, realpreds_c[i])
    print(realpreds_c)
    result = {'prec': metrics.precision_score(classlabels, preds, average='macro'),
              'recall': metrics.recall_score(classlabels, preds, average='macro'),
              'acc_yuan': metrics.accuracy_score(classlabels, preds),
              # 'acc_dis':acc,
              'F1': metrics.f1_score(classlabels, preds, average="micro")}
    return result, labels, preds, jordan_center