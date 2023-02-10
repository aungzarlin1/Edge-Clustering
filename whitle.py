import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


def cluster_whitle(G, K, method='PRE'):

    print('starting')

    M = len(G.edges)
    N = len(G.nodes)
    df = nx.to_pandas_edgelist(G)

    # incidence matrix
    B = -nx.incidence_matrix(G, oriented=True)

    w = np.array(df['weight'])
    norm_w = np.linalg.norm(w, 1)

    print('sigma calculate')

    sigma = np.diag(np.abs(B) @ np.diag(np.ones(M)) @ np.abs(B).T)

    print('sigma complete')

    nu = np.diag(np.abs(B) @ np.diag(np.abs(w)) @ np.abs(B).T) / norm_w

    print('nu complete')

    Le = B.T @ np.diag(nu) @ B
    print("Edge Laplacian computed.")

    # FLOW LAPLACIAN
    dualW = Le - np.diag(np.diag(Le))
    dualD = np.abs(dualW) @ np.ones((M,1))
    dualD = np.diag(dualD.reshape(-1))

    if method == 'PRE':
        Lf = dualD - dualW
    elif method == 'DPE':
        Lf = dualD + dualW
    elif method == 'RGE':
        Lf = dualD - np.abs(dualW)
    else:
        raise ValueError('Method must be one of the following: "PRE", "DPE" or "RGE".')

    # Force flow Laplacian to be symmetric
    Lf = np.triu(Lf)
    Lf = Lf + Lf.T - np.diag(np.diag(Lf))
    print('Flow Laplacian computed.')

    # Normalized Flow Laplacian

    # Edge volume vector
    Bout, Bin = (B > 0), (B < 0)
    Dout = Bout @ np.abs(w)    # vector of sum of absolute values of out-weights.              
    Din = Bin @ np.abs(w)      # vector of sum of absolute values of in-weights.

    sigma = sigma * nu
    Fdiag = 0.5 * np.abs(w) * (Bout.T @ np.divide(sigma, Dout) + Bin.T @ np.divide(sigma, Din))
    Fdiag = np.diag(Fdiag**(-0.5))
    
    # Normalized flow Laplacian
    normLf = Fdiag @ Lf @ Fdiag

    # Force normalized flow Laplacian to be symmetric.
    normLf = np.triu(normLf)
    normLf = normLf + normLf.T - np.diag(np.diag(normLf))
    print('Normalized flow Laplacian computed.')

    # Smallest K eigenvalue and eigenvector
    eigval, eigvec = np.linalg.eig(normLf)
    idx = np.argsort(eigval)[:K]
    eigval = eigval[idx]
    DR = eigvec[:, idx]
    
    # Normalized eigenvectors
    l2 = np.linalg.norm(DR, axis=1).reshape(-1,1)
    DR = DR / l2


    # K-means Clustering
    model = KMeans(n_clusters=K, random_state=42, n_init=350)
    Elabels = model.fit_predict(DR) 
    print('Edge Labels assigned.')
    
    return Elabels 





    






