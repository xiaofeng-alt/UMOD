from AUMOD import AUMOD
import numpy as np
import time
import scipy.io as scio
import math

import utils.metric as m_u
from utils import Dataprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as SC


def test_ATG_clustering():
    log_name = './result/clustering_avg_log.txt'
    iter_log_name = './result/clustering_log.txt'
    o = 500
    r = 5
    s=0
    l=20
    file_name = 'USPS'
    m_u.print_log(
        '***************************************' + file_name + '*******************************************',
        file_name=log_name)
    m_u.print_log(
        '***************************************' + file_name + '*******************************************',
        file_name=iter_log_name)

    X, labels_true = Dataprocessing.get_datasets(file_name)
    n, dim = X.shape
    n_clusters = len(np.unique(labels_true.copy()))
    m_u.print_log('n:{},s:{},o:{},r:{},n_clusters:{},l:{}'.format(n, s, o, r, n_clusters, l),
                  file_name=log_name)
    t_l = [0, 0, 0]
    nmi_l = [0, 0, 0]
    acc_l = [0, 0, 0]
    ami_l = [0, 0, 0]
    ari_l = [0, 0, 0]
    F1_l = [0, 0, 0]
    F1_macro_l = [0, 0, 0]
    tt = 10
    for j in range(tt):
        m_u.print_log('seed:{},n:{},s:{},o:{},r:{},n_clusters:{},l:{}'.format(j, n, s, o, r, n_clusters, l),
                      file_name=iter_log_name)
        start = time.time()
        amod = AUMOD(s, o=o, r=r, random_state=j, l=l)
        amod.ATG_SC(X, n_clusters=n_clusters)
        end = time.time()

        nmi, acc, ami, ari, F1, F1_macro = m_u.print_mess(amod.labels_, labels_true, file_name=iter_log_name,
                                                          pre_mess='SC  WITH LTG')
        t_l[0] += end - start
        nmi_l[0] += nmi
        acc_l[0] += acc
        ami_l[0] += ami
        ari_l[0] += ari
        F1_l[0] += F1
        F1_macro_l[0] += F1_macro

        start = time.time()
        amod = AUMOD(s, o=o, r=r, random_state=j, l=l)
        amod.ATG_DPC(X, n_clusters=n_clusters)
        end = time.time()

        nmi, acc, ami, ari, F1, F1_macro = m_u.print_mess(amod.labels_, labels_true, file_name=iter_log_name,
                                                          pre_mess='DPC WITH LTG')
        t_l[1] += end - start
        nmi_l[1] += nmi
        acc_l[1] += acc
        ami_l[1] += ami
        ari_l[1] += ari
        F1_l[1] += F1
        F1_macro_l[1] += F1_macro

        start = time.time()
        amod = AUMOD(s, o=o, r=r, random_state=j, l=l)
        amod.ATG_KM(X, n_clusters=n_clusters)
        end = time.time()
        nmi, acc, ami, ari, F1, F1_macro = m_u.print_mess(amod.labels_, labels_true, file_name=iter_log_name,
                                                          pre_mess='KM  WITH LTG')
        t_l[2] += end - start
        nmi_l[2] += nmi
        acc_l[2] += acc
        ami_l[2] += ami
        ari_l[2] += ari
        F1_l[2] += F1
        F1_macro_l[2] += F1_macro
    m_u.print_log('######################################################################', file_name=log_name)
    m_u.print_log(
        'ATG_SC,  nmi:{:.2f}%, acc:{:.2f}%, ami:{:.2f}%, ari:{:.2f}%, f1:{:2f}%, f1_macro:{:2f}%, RT:{:6f}'.format(
            nmi_l[0] * 100 / tt,
            acc_l[0] * 100 / tt,
            ami_l[0] * 100 / tt,
            ari_l[0] * 100 / tt,
            F1_l[0] * 100 / tt,
            F1_macro_l[0] * 100 / tt,
            t_l[0] / tt), file_name=log_name)
    m_u.print_log(
        'ATG_DPC, nmi:{:.2f}%, acc:{:.2f}%, ami:{:.2f}%, ari:{:.2f}%, f1:{:2f}%, f1_macro:{:2f}%, RT:{:6f}'.format(
            nmi_l[1] * 100 / tt,
            acc_l[1] * 100 / tt,
            ami_l[1] * 100 / tt,
            ari_l[1] * 100 / tt,
            F1_l[1] * 100 / tt,
            F1_macro_l[1] * 100 / tt,
            t_l[1] / tt),
        file_name=log_name)
    m_u.print_log(
        'ATG_KM,  nmi:{:.2f}%, acc:{:.2f}%, ami:{:.2f}%, ari:{:.2f}%, f1:{:2f}%, f1_macro:{:2f}%, RT:{:6f}'.format(
            nmi_l[2] * 100 / tt,
            acc_l[2] * 100 / tt,
            ami_l[2] * 100 / tt,
            ari_l[2] * 100 / tt,
            F1_l[2] * 100 / tt,
            F1_macro_l[2] * 100 / tt,
            t_l[2] / tt),
        file_name=log_name)
    m_u.print_log('\n', file_name=log_name)

if __name__ == '__main__':
    test_ATG_clustering()