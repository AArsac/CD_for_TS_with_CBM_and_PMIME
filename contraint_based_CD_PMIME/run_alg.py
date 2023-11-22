# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:23:19 2023

@author: AA270673
"""

import CB_PMIME as cbalg
# import FCI_assaad as fcim
import pandas as pd
import numpy as np
import networkx as nx
# import time
# from joblib import Parallel, delayed

def run_algo(data,method= 'PC', nlags=5, nb_process = 1, verbose=True, **kwargs):
    '''
    Parameters
    ----------
    data : pandas dataframe containing the data 
    method : str, optional
        PC or FCI. The default is 'PC'.
    nlags : int, optional
        max lag considered. The default is 5.
    nb_process : int, optional
        how many processors involved in the parallel computation. The default is 1.
    verbose : bool, optional
        display info or not.

    Returns
    -------
    g : nx.Digraph
        oriented graph with auto-causation considered
    og : nx.Digraph
        oriented graph without auto-causation
    sg : nx.Digraph
        only self-causation 

    '''
    bootstrap_bool = kwargs.get('bootstrap', False)
    stopping_threshold = kwargs.get('sig_pmime', 0.03)
    sig_bs = kwargs.get('sig_bs', 0.05)
    nb_bs = kwargs.get('nb_bs', 100)
    knnei = kwargs.get('knnei', 0.01)

    if method == 'FCI':
        ci = cbalg.FCI(data,  lag_max=nlags, p_value=False, rank_using_p_value=False, verbose=verbose,
                   num_processor=nb_process, sig_pmime=stopping_threshold, knnei = knnei,
                   bs = bootstrap_bool, sig_CMI = sig_bs, nb_bs = nb_bs )
        _ = ci.fit()
        
    elif method == 'PC':
        ci = cbalg.PC(data,  lag_max=nlags, p_value=False, rank_using_p_value=False, verbose=verbose,
                   num_processor=nb_process, sig_pmime=stopping_threshold, knnei = knnei,
                   bs = bootstrap_bool, sig_CMI = sig_bs, nb_bs = nb_bs )
        _ = ci.fit()
    else:
        print('enter valid method name, PC or FCI')
        
    g = ci.graph.to_summary()

    nodes = list(ci.graph.map_names_nodes.keys())

    og = nx.DiGraph()
    sg = nx.DiGraph()
    og.add_nodes_from(nodes)
    sg.add_nodes_from(nodes)
    for cause, effects in g.adj.items():
        for effect, _ in effects.items():
            if cause != effect:
                og.add_edges_from([(cause, effect)])
            else:
                sg.add_edges_from([(cause, effect)])

    print(g.edges)
    print(og.edges)
    print(sg.edges)
    return g, og, sg

if __name__ == '__main__':
    n = 2000
    X=1/10*np.ones((n,4))
    X[1,1] = np.random.randn()
    # for j in range(1,n):
    #     X[j,0] = np.random.random()*X[j-1,0] + 0.01*np.random.randn()
   
    # for k in range(1,n):
    #     X[k,2] = np.random.uniform()*np.abs(X[k-1,0]) + np.random.uniform() *X[k-1,2] + 0.01*np.random.randn()
    # for k in range(1,n):
    #     X[k,3] = np.random.uniform()*X[k-1,3] + 0.01*np.random.randn()
    # for t in range(2,n):
    #     X[t,1] = np.random.uniform()*np.tanh(X[t-3,0]**2)  + np.random.uniform()*np.sin(X[t-2,2]) +  np.random.uniform()*X[t-1,1] + 0.01*np.random.randn()
    #X[t,1] =  np.random.uniform()*X[t-2,2]**2 +  np.random.uniform()*X[t-1,1] + np.random.randn()
    for j in range(1,n):
        X[j,0] = 0.7*X[j-1,0] + 0.01*np.random.randn()
    for k in range(1,n):
        X[k,3] = 0.8*X[k-1,3] + 0.01*np.random.randn()
    for k in range(1,n):
        X[k,2] = 0.6*np.abs(X[k-1,3]) + 0.5 *X[k-1,2] + 0.01*np.random.randn()

    for t in range(2,n):
        X[t,1] = 0.9*np.tanh(X[t-3,0])  + 0.5*np.sin(X[t-2,2]) +  0.4*X[t-1,1] + 0.01*np.random.randn()
    df = pd.DataFrame(data = X, columns = ['X_0','X_1','X_2','X_3'])
    
    g, og, sg = run_algo(df, method = 'PC', nb_process = 10, knnei = 0.01, sig_pmime = 0.03)

