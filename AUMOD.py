# -*- coding: utf-8 -*-
# @Author  : Xiaofeng Feng
# @Time    : 2024/10/12 17:02
# @Email: xfeng_fxf@163.com
# @University: Guangdong University of Technology

import numpy as np
from sklearn.cluster import KMeans

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh

from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils import check_array, check_random_state
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances as EuDist2
from utils import Dataprocessing
import numba as nb
import time
import os


def row_min_max_scaler(arr):
    scaler = MinMaxScaler()
    transposed_arr = arr.T
    scaled_arr = scaler.fit_transform(transposed_arr).T
    return scaled_arr

@nb.njit()
def accelerated_knn(n, r, dists_n_to_o):
    nn_dists = np.full((n, r), np.inf, dtype=np.float32)
    nn_indices = np.full((n, r), -1, dtype=np.int32)
    for i in nb.prange(n):
        idx = np.argsort(dists_n_to_o[i, :])  # 对距离排序
        nn_idx = idx[:r]  # 选取前 r 个最小值的索引
        nn_indices[i, :] = nn_idx  # 存储这些索引
        nn_dists[i, :] = np.sort(dists_n_to_o[i, :])[:r]  # 存储这些索引对应的距离
    return nn_indices, nn_dists


class AUMOD:
    def __init__(self, s=None, o=500, r=5, l=20, random_state=None):
        self.s = s
        self.o = o
        self.r = r
        self.l = l
        self.random_state = random_state

    def calculate_AUMOD(self, X):
        self.n = X.shape[0]
        # observers selection
        if self.s == 0 or self.s is None or self.s > self.n:
            data =  X
        else:
            np.random.seed(self.random_state)
            s_idx = np.random.choice(self.n, self.s)
            data = X[s_idx,:]
        km = KMeans(n_clusters=self.o, random_state=self.random_state, init='random', max_iter=3, n_init=1).fit(data)

        observers = km.cluster_centers_
        self.objects_km_label = km.labels_
        self.observers = observers

        # Z^S Construction
        Z_S = self.get_Z_S(data, observers, self.r)
        # Z^O Construction
        Z_O, row, col = self.get_Z_O(Z_S, observers, self.r)
        return observers, Z_O, row, col

    def get_Z_S(self, data, observers, r):
        s = data.shape[0]
        o = observers.shape[0]
        dists_n_to_o = EuDist2(data, observers, squared=False)
        nn_dists = np.full((s, r), np.inf, dtype=np.float32)
        nn_indices = np.full((s, r), -1, dtype=np.int32)
        for i in nb.prange(s):
            idx = np.argsort(dists_n_to_o[i, :])
            nn_idx = idx[:r]
            nn_indices[i, :] = nn_idx
            nn_dists[i, :] = dists_n_to_o[i, nn_idx]
        row = np.repeat(np.arange(s), r)
        col = nn_indices.flatten()
        sigma_x = np.mean(nn_dists, axis=1).reshape(-1, 1)
        sigma_o = np.zeros(o)
        # sigma_o 计算：仅计算出现过的核心
        clu_idx = nn_indices[:, 0]
        clu_d = nn_dists[:, 0]
        for i in np.unique(clu_idx):
            sigma_o[i] = np.mean(clu_d[clu_idx == i])
        # 避免 sigma 值为零
        sigma_x[sigma_x == 0] = 1
        sigma_o[sigma_o == 0] = np.max(sigma_o)

        # 直接在 CSR 矩阵上计算距离缩放
        scaled_dists = np.exp(- (nn_dists.flatten() ** 2) / (sigma_x[row].flatten() * sigma_o[col].flatten()))
        Z_S = csr_matrix((scaled_dists, col, range(0, s * r + 1, r)), shape=(s, o))

        self.Z_S = Z_S
        return Z_S

    def get_Z_O(self, Z_X, observers, r):
        o = observers.shape[0]
        d_O = np.full((o, o), np.inf, dtype=np.float64)
        sigmas = np.full(o, np.inf, dtype=np.float64)
        dists_o_to_o = EuDist2(observers, observers, squared=False)
        phi = Z_X.T * Z_X
        phi = phi.toarray()
        self.phi = phi
        self.dists_o_to_o = dists_o_to_o
        idx_list = []
        for i in nb.prange(o):
            phi_ = phi[i, :]
            phi_[i] = 0
            idx = np.where(phi_ > 0)[0]
            idx_list.append(idx)
            phi_ = phi_ / np.min(phi_[idx])
            d = dists_o_to_o[i, idx] / phi_[idx]
            iddd = np.where(d == 0)[0]
            if len(iddd) != 0:
                print(d)
            d_O[i, idx] = d
            sigma_idx = np.argsort(d)
            if len(sigma_idx) >= r:
                sigmas[i] = d[sigma_idx[r - 1]]
            elif len(sigma_idx) == 0:
                id = np.argsort(dists_o_to_o[i, :])
                d_O[i, id[:r]] = dists_o_to_o[i, id[:r]]
                d_O[id[:r], i] = d_O[i, id]
                sigmas[i] = d_O[i, id[r - 1]]
            else:

                sigmas[i] = d[sigma_idx[len(sigma_idx) - 1]]
            if sigmas[i] == 0:
                sigmas[i] = 1

        Z_O = np.full((o, o), 0.0)
        Z_O_idx = np.full((o, o), 0.0)
        for i in range(o):
            idx = idx_list[i]
            scale_dists = d_O[i,idx] ** 2 / (sigmas[i]*sigmas[idx])

            Z_O[i, idx] = scale_dists
            Z_O_idx[i, idx] = 1
        Z_O = csr_matrix(Z_O)
        Z_O_idx = coo_matrix(Z_O_idx)
        row = Z_O_idx.row
        col = Z_O_idx.col
        return Z_O, row, col

    def ATG_SC(self, X, n_clusters):
        self.n = X.shape[0]
        # G^O Construction
        observers, Z_O, row, col = self.calculate_AUMOD(X)

        # L^O Construction and eigenvalue decomposition
        o = observers.shape[0]
        G_O = np.full((o, o), 0.0)
        Z_O = Z_O.toarray()
        G_O[row, col] = np.exp(-Z_O[row, col])
        G_O = (G_O + G_O.transpose()) / 2
        Y_O = self.spectral_clustering_G_O(G_O, n_clusters=n_clusters, random_state=self.random_state)

        # assign non-observers
        labels = self.assign(X, observers,Y_O)

        self.labels_ = labels
        self.Y_O = Y_O
        self.G_O = G_O
        return labels

    def ATG_DPC(self, X, n_clusters, pre_density=None):
        self.n_ = X.shape[0]
        # G^O Construction
        observers, Z_O, row, col = self.calculate_AUMOD(X)

        # (G_P)^O Construction
        o = observers.shape[0]
        G_O = np.full((o, o), 0.0)
        Z_O = Z_O.toarray()
        G_O[row, col] = np.exp(-Z_O[row, col])
        for i in range(o):
            idx = G_O[i,:]>0
            G_O[i,idx] = G_O[i,idx] / np.sum(G_O[i,idx])
        temp = G_O.copy()
        for i in range(self.l):
            temp += np.dot(temp, G_O)
        temp = (temp + temp.T) / 2
        G_O = temp

        # Calculate density
        if pre_density is not None:
            self.rho = pre_density
        else:
            self.rho = np.sum(G_O, axis=0)
        # Calculating delta
        deltas, o_ndh = self.get_delta_sim(self.rho, G_O)
        sig = np.full(o, np.inf, dtype=np.float32)
        sig[deltas > 0] = self.rho[deltas > 0] / deltas[deltas > 0]
        idx = np.argsort(-sig)
        centers_idx = idx[:n_clusters]
        centers = observers[centers_idx]

        # Calculate significance
        Y_O = np.full(o, -1)
        o_ndh[centers_idx] = -1
        for i, center_idx in enumerate(centers_idx):
            Y_O[center_idx] = i
            next_idx = np.where(o_ndh == center_idx)[0]
            while len(next_idx) != 0:
                Y_O[next_idx] = i
                next_idx = np.where(np.in1d(o_ndh, next_idx))[0]

        # assign non-observers
        labels = self.assign(X, observers, Y_O)
        self.deltas = deltas
        self.centers_idx = centers_idx
        self.centers_ = centers
        self.Y_O = Y_O
        self.labels_ = labels

        return labels

    def ATG_KM(self, X, n_clusters):
        self.n_ = X.shape[0]
        # G^O Construction
        observers, Z_O, row, col = self.calculate_AUMOD(X)

        # (G_P)^O Construction
        o = observers.shape[0]
        G_O = np.full((o, o), 0.0)
        Z_O = Z_O.toarray()
        G_O[row, col] = np.exp(-Z_O[row, col])
        for i in range(o):
            idx = G_O[i,:]>0
            G_O[i,idx] = G_O[i,idx] / np.sum(G_O[i,idx])
        temp = G_O.copy()
        for i in range(self.l):
            temp += np.dot(temp, G_O.T)
        temp = (temp + temp.T) / 2
        G_O = temp
        np.fill_diagonal(G_O, np.max(G_O))

        # K-MEANS on (G_P)^O
        centers_indices, Y_O = self.km_iters_sim(G_O, n_clusters=n_clusters, n_iters=20, random_state=None)
        
        # assign non-observers
        labels = self.assign(X, observers, Y_O)
        self.Y_O = Y_O
        self.labels_ = labels

        return labels

    def assign(self, X, observers,Y_O):
        if self.s == 0 or self.s is None or self.s > self.n:
            near_o_idx = self.objects_km_label
        else:

            n_o_d = EuDist2(X, observers, squared=True)
            near_o_idx= np.full(self.n, -1, dtype=np.int32)
            for i in nb.prange(self.n):
                idx = np.argmin(n_o_d[i, :])
                near_o_idx[i] = idx
        labels = Y_O[near_o_idx]
        return labels

    def spectral_clustering_G_O(self, G_O, n_clusters, random_state=None):
        if not Dataprocessing._graph_is_connected(G_O):
            warnings.warn(
                "Graph is not fully connected, spectral embedding may not work as expected."
            )
        random_state = check_random_state(random_state)
        o = G_O.shape[0]
        L_O = G_O.copy()
        np.fill_diagonal(G_O, 0)
        w = L_O.sum(axis=0)
        isolated_node_mask = (w == 0)
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        L_O /= w
        L_O /= w[:, np.newaxis]
        L_O = (L_O + L_O.T) / 2
        v0 = random_state.uniform(-1, 1, o)
        val, diffusion_map = eigsh(
            L_O, k=n_clusters + 1, which="LM", tol=0.0, v0=v0
        )
        U_O = diffusion_map[:, 1:n_clusters + 1]
        U_O = U_O / np.sqrt(np.sum(U_O**2,axis=1)).reshape(-1, 1)
        os.environ["OMP_NUM_THREADS"] = '1'
        est = KMeans(n_clusters=n_clusters, random_state=random_state,n_init=10).fit(U_O)
        return est.labels_

    def get_delta_sim(self, rho, G_O):

        o = rho.shape[0]
        deltas = np.full(o, 0.0, dtype=np.float32)
        o_ndh = np.full(o, -1, dtype=np.int32)
        rho_order_index = np.argsort(-rho)  # rho 排序索引
        for i in range(1, o):
            rho_index = rho_order_index[i]  # 对应 rho 的索引（点的编号）
            j_list = rho_order_index[:i]  # j < i 的排序的索引值 -> rho > i 的列表
            max_sim_index = np.argmax(G_O[rho_index, j_list])
            max_sim_index = j_list[max_sim_index]
            max_sim = G_O[rho_index, max_sim_index]

            deltas[rho_index] = max_sim
            o_ndh[rho_index] = max_sim_index
        return deltas, o_ndh

    def km_init_sim(self, G_O, n_clusters, random_state=None):
        indices = []
        n_local_trials = 2 + int(np.log(n_clusters))

        inertias = np.sum(G_O, axis=1)

        center_0 = np.argmax(inertias)
        indices.append(center_0)
        closest_sim_sq = G_O[center_0, :].copy()
        closest_sim_sq[center_0] = np.inf
        idx = np.where(closest_sim_sq == 0)[0]
        closest_sim_sq[center_0] = 0
        while idx.size != 0:
            # candidate_id = np.random.choice(idx)
            sim_to_candidate = G_O[idx, :].copy()
            np.maximum(closest_sim_sq, sim_to_candidate, out=sim_to_candidate)
            best_candidate = np.argmax(np.sum(sim_to_candidate, axis=1))
            closest_sim_sq = sim_to_candidate[best_candidate, :]
            indices.append(idx[best_candidate])
            closest_sim_sq[indices] = np.inf
            idx = np.where(closest_sim_sq == 0)[0]
            closest_sim_sq[indices] = 0

        current_pot = closest_sim_sq.sum()
        for c in range(len(indices), n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            # closest_sim_sq[indices] = np.inf
            random_state = check_random_state(random_state)
            rand_vals = random_state.uniform(size=n_local_trials) * current_pot
            candidate_ids = np.searchsorted(
                stable_cumsum(closest_sim_sq), rand_vals
            )
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_sim_sq.size - 1, out=candidate_ids)
            # Compute distances to center candidates
            sim_to_candidates = G_O[candidate_ids, :].copy()
            sim_to_candidates[:, indices] = 0
            # update closest distances squared and potential for each candidate
            np.maximum(closest_sim_sq, sim_to_candidates, out=sim_to_candidates)
            # print(sim_to_candidates)
            candidates_pot = np.sum(sim_to_candidates, axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmax(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_sim_sq = sim_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            if best_candidate in indices:
                continue
            indices.append(best_candidate)
        return indices

    def km_iters_sim(self, G_O, n_clusters, n_iters, random_state=None):
        o = G_O.shape[0]
        # 初始centers
        G_O_ = G_O.copy()
        indices = self.km_init_sim(G_O, n_clusters, random_state)
        c_to_m_d = G_O_[indices, :]
        labels = np.argmax(c_to_m_d, axis=0)
        for t in range(n_iters):
            new_indices = []
            # new_inertias=[]
            for id in range(n_clusters):
                labels[id] = id
                idx = np.where(labels == id)[0]
                candidate_inertias = []
                for j in idx:
                    candidate_inertia = np.sum(G_O[j, idx])
                    candidate_inertias.append(candidate_inertia)
                candidate_idx = np.argmax(candidate_inertias)
                new_indices.append(idx[candidate_idx])
            c_to_o_d = G_O[new_indices, :]
            new_labels = np.argmax(c_to_o_d, axis=0)
            if (labels == new_labels).all():
                return indices, labels
            labels = new_labels
            indices = new_indices

        return indices, labels