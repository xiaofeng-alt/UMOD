# -*- coding: utf-8 -*-
# @Author  : Xiaofeng Feng
# @Time    : 2023/11/9 19:08
# @Email: xfeng_fxf@163.com
# @University: Guangdong University of Technology
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score, accuracy_score, f1_score
from munkres import Munkres
import numpy as np


def print_log(message, file_name='./log.txt'):
    print(message)
    with open(file_name, 'a') as f:
        f.write(message)
        f.write('\n')

def cluster_accuracy(y_true, y_pre):
        Label1 = np.unique(y_true)
        nClass1 = len(Label1)
        Label2 = np.unique(y_pre)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = y_true == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = y_pre == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        y_best = np.zeros(y_pre.shape)
        for i in range(nClass2):
            y_best[y_pre == Label2[i]] = Label1[c[i]]

        # # calculate accuracy
        err_x = np.sum(y_true[:] != y_best[:])
        missrate = err_x.astype(float) / (y_true.shape[0])
        acc = 1. - missrate
        nmi = normalized_mutual_info_score(y_true, y_pre)
        kappa = cohen_kappa_score(y_true, y_best)

        # ca = self.class_acc(y_true, y_best)
        return (acc, nmi, kappa), y_best

def mode_to_center_inertia(X, center,labels):
    # mode到center的距离的平方和
    modes_to_c = []
    for i in range(len(center)):
        cluster_members = np.where(labels == i)[0]
        dists_to_center = (center[i,:] - X[cluster_members, :]) ** 2
        dists_mode_to_c = np.min(dists_to_center)
        modes_to_c.append(dists_mode_to_c)

    inertia_mode_to_c = np.sum(modes_to_c)
    return inertia_mode_to_c

def inertia_(X, centers, labels):
    inertia = np.sum((X - centers[labels,:]) ** 2)
    return inertia

def Accurry(true_labels, predicted_labels):
    # 创建一个字典，将预测标签映射到真实标签
    label_mapping = {}
    unique_labels = set(true_labels)
    unique_pre_labels = set(predicted_labels)
    for label in unique_pre_labels:
        mapping = true_labels[predicted_labels == label]

        if len(mapping)> 0:
            most_common_true_label = max(set(mapping), key=list(mapping).count)
            label_mapping[label] = most_common_true_label
        
        else:
            label_mapping[label] = ''

    # 将预测标签映射到真实标签
    mapped_predicted_labels = [label_mapping[label] for label in predicted_labels]

    # 计算准确率
    accuracy = sum(mapped_predicted_labels == true_labels) / len(true_labels)
    
    return accuracy,mapped_predicted_labels


def F1Score(true_labels, predicted_labels, method):
    # 获取唯一的类别标签
    unique_labels = np.unique(true_labels)
    n_classes = len(unique_labels)

    # 初始化精度、召回率和 F1 分数数组
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)

    # 计算每个类别的 TP, FP, FN
    for i in range(n_classes):
        class_label = unique_labels[i]

        TP = np.sum((true_labels == class_label) & (predicted_labels == class_label))
        FP = np.sum((true_labels != class_label) & (predicted_labels == class_label))
        FN = np.sum((true_labels == class_label) & (predicted_labels != class_label))

        # 计算 Precision 和 Recall
        if TP + FP > 0:
            precision[i] = TP / (TP + FP)
        else:
            precision[i] = 0

        if TP + FN > 0:
            recall[i] = TP / (TP + FN)
        else:
            recall[i] = 0

        # 计算 F1 Score
        if precision[i] + recall[i] > 0:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_score[i] = 0

    # 根据方法选择输出 Macro F1 或 Micro F1
    if method.lower() == 'macro':
        # 计算 Macro F1
        f1_score = np.mean(f1_score)
    elif method.lower() == 'micro':
        # 计算 Micro F1
        total_TP = np.sum(true_labels == predicted_labels)
        total_FP = np.sum((true_labels != predicted_labels) & (predicted_labels != 0))
        total_FN = np.sum((true_labels != predicted_labels) & (true_labels != 0))

        if total_TP + total_FP > 0:
            micro_precision = total_TP / (total_TP + total_FP)
        else:
            micro_precision = 0

        if total_TP + total_FN > 0:
            micro_recall = total_TP / (total_TP + total_FN)
        else:
            micro_recall = 0

        if micro_precision + micro_recall > 0:
            f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        else:
            f1_score = 0
    else:
        raise ValueError("Invalid method. Choose either 'macro' or 'micro'.")

    return f1_score


def print_mess(labels,labels_true,file_name='./log.txt',pre_mess=''):
    ami = AMI(labels_true, labels)
    ari = ARI(labels_true, labels)
    nmi = NMI(labels_true, labels)
    acc,mapped_predicted_labels = Accurry(labels_true, labels)
    F1 = f1_score(labels_true, mapped_predicted_labels,average='weighted')
    F1_macro = F1Score(labels_true, mapped_predicted_labels, 'macro')

    print_log(pre_mess + ' nmi:{:.4f},  acc:{:.4f}, ami:{:.4f} ari:{:.4f} F1:{:.4f},F1_macro:{:.4f}'.format(nmi,acc,ami,ari,F1,F1_macro), file_name=file_name)
    return nmi,acc,ami,ari,F1,F1_macro
