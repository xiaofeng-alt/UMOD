import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import numbers

import csv
from sklearn import preprocessing
from sklearn import datasets
import scipy.io as scio

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node.

    Parameters
    ----------
    graph : array-like of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    node_id : int
        The index of the query node of the graph.

    Returns
    -------
    connected_components_matrix : array-like of shape (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node.
    """
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes

def _graph_is_connected(graph):
    """Return whether the graph is connected (True) or Not (False).

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]
    
    
def _deterministic_vector_sign_flip(u):
        """Modify the sign of vectors for reproducibility.

        Flips the sign of elements of all the vectors (rows of u) such that
        the absolute maximum element of each vector is positive.

        Parameters
        ----------
        u : ndarray
            Array with vectors as its rows.

        Returns
        -------
        u_flipped : ndarray with same shape as u
            Array with the sign flipped vectors as its rows.
        """
        max_abs_rows = np.argmax(np.abs(u), axis=1)
        signs = np.sign(u[range(u.shape[0]), max_abs_rows])
        u *= signs[:, np.newaxis]
        return u


def get_data(filename):
    data = []
    label = []

    labs = []
    with open(filename, 'r') as file_obj:
        csv_reader = csv.reader(file_obj)
        csv_reader = list(csv_reader)
        for row in csv_reader:
            row = list(row)
            point = []
            for d in row[:-1]:
                
                if d == np.nan or d == '':
                    point = []
                    break
                point.append(float(d))
            if len(point) == 0:
                continue  
            data.append(point)

            lab = row[-1]
            if isinstance(lab, str):
                 if lab not in labs:
                     labs.append(lab)
                 idx = labs.index(lab)
                 label.append(idx)
            else:
                label.append(int(float(row[-1])))
    X = np.array(data, dtype=np.float32)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_minMax = min_max_scaler.fit_transform(X)
    # X_minMax =X
    return X, np.array(label, np.int8)

def get_datasets(file_name, n_samples =1500, seed=170):
    if file_name == 'mooons':
        X, label_true = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

    elif file_name == 'circles':
        X, label_true = datasets.make_circles(
            n_samples=n_samples, factor=0.8, noise=0.05, random_state=seed
        )
    elif file_name == 'varied':
        X, label_true = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.0, 0.5], random_state=seed)

    elif file_name == 'aniso':

        X, label_true = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)

    elif file_name == 'blobs':
        X, label_true = datasets.make_blobs(n_samples=n_samples, random_state=seed)

    elif file_name == 'no_structure':
        X, label_true = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        rng = np.random.RandomState(seed)
        X = rng.rand(n_samples, 2)

    elif file_name in ['USPS','MNIST','PenDigits','Optdigits','emnist-letter','emnist-digits','Covtype''Balanced','Byclass']:
        DATA_File = './datasets/' + file_name + '.mat'
        data = scio.loadmat(DATA_File)
        X = data['fea']
        label_true = data['gnd']
        label_true = label_true.flatten()

    elif file_name == 'mnsit_8m':
        print(file_name)
        files_Name = ['mnist1q.mat','mnist2q.mat',
                'mnist3q.mat','mnist4q.mat',
                'mnist5q.mat','mnist6q.mat',
                'mnist7q.mat','mnist8q.mat',
                'mnist9q.mat','mnist10q.mat',
                'mnist11q.mat','mnist12q.mat',
                'mnist13q.mat','mnist14q.mat',
                'mnist15q.mat','mnist16q.mat']
        for i,file in enumerate(files_Name):
            DATA_File = 'D:/data/数据集/mnist_8m/' + file
            data = scio.loadmat(DATA_File)
            X_ = data['fea']
            label_true_ = data['gnd']
            X_ = X_.toarray()
            label_true_ = label_true_.flatten()
            if i ==0 :
                X = X_
                label_true = label_true_
            else:
                X = np.concatenate((X,X_), axis=0)
                label_true = np.concatenate((label_true,label_true_))

    X = X.astype(np.float64)
    return X, label_true


def standardize_label(y):
        """
        standardize the classes label into 0-k
        :param y:
        :return:
        """
        import copy
        classes = np.unique(y)
        standardize_y = copy.deepcopy(y)
        label_map = {}
        for i in range(classes.shape[0]):
            standardize_y[np.nonzero(y == classes[i])] = i
            label_map[i] = classes[i]
        return standardize_y, label_map
def order_sam_for_diag(x, y):
    """
    rearrange samples
    :param x: feature sets
    :param y: ground truth
    :return:
    """
    x_new = np.zeros(x.shape)
    y_new = np.zeros(y.shape)
    index = np.zeros(y.shape,dtype=np.int32)
    start = 0
    for i in np.unique(y):
        idx = np.nonzero(y == i)
        stop = start + idx[0].shape[0]
        x_new[start:stop] = x[idx]
        y_new[start:stop] = y[idx]
        index[idx] = np.arange(start,stop)
        start = stop
    return x_new, y_new, index