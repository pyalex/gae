import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj, test=0.01, val=0.005):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert adj.diagonal().sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = set(map(tuple, sparse_to_tuple(adj)[0]))
    num_test = int(np.floor(edges.shape[0] * test))
    num_val = int(np.floor(edges.shape[0] * val))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    train_edges_set = set(map(tuple, train_edges))
    val_edges_set = set(map(tuple, val_edges))
    test_edges_set = set(map(tuple, val_edges))

    def ismember(a, b):
        if isinstance(a, set):
            return bool(a & b)

        return a in b

    test_edges_false = set()
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        candidate = (idx_i, idx_j)

        if ismember(candidate, edges_all):
            continue
        if test_edges_false:
            if ismember(tuple(reversed(candidate)), test_edges_false):
                continue
            if ismember(candidate, test_edges_false):
                continue
        test_edges_false.add(candidate)

    val_edges_false = set()
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        candidate = (idx_i, idx_j)

        if ismember(candidate, train_edges_set):
            continue
        if ismember(tuple(reversed(candidate)), train_edges_set):
            continue
        if ismember(candidate, val_edges_set):
            continue
        if ismember(tuple(reversed(candidate)), val_edges_set):
            continue
        if val_edges_false:
            if ismember(tuple(reversed(candidate)), val_edges_false):
                continue
            if ismember(candidate, val_edges_false):
                continue

        val_edges_false.add(candidate)

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges_set, train_edges_set)
    assert ~ismember(test_edges_set, train_edges_set)
    assert ~ismember(val_edges_set, test_edges_set)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
