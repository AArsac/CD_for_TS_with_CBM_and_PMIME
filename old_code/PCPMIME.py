# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:26:11 2022

@author: Antonin Arsac

This file holds the code for the PC algorithm for time series, coupled
with a measure of links between time series : PMIME.


 Markus Kalisch and Peter Bhlmann. 
Estimating high-dimensional directed acyclic graphs with the pc-algorithm. 
In The Journal of Machine Learning Research, Vol. 8, pp. 613-636, 2007.

"""

import pandas as pd
import numpy as np
from math import *


import networkx as nx


from itertools import combinations, permutations 

import PMIME_pc as pm 


def array_to_digraph(array, nodes):
    '''
    Parameters
    ----------
    array : adjacency matrix (numpy array)
    nodes : list of nodes (variable labels)

    Returns
    -------
    g_hat : corresponding networkx Digraph

    '''
    g_hat = nx.DiGraph()
    g_hat.add_nodes_from(nodes)
    for var in range(array.shape[0]):
        # if array[var,var]>0 :
        #     g_hat.add_edges_from([(nodes[var],nodes[var])])
        for col in range(array.shape[1]):
            if (nodes[var] != nodes[col]):
                if (array[var,col] >0):
                    g_hat.add_edges_from([(nodes[var],nodes[col])])
    return g_hat

def array_to_graph(array, nodes):
    '''
    Parameters
    ----------
    array : adjacency matrix (numpy array)
    nodes : list of nodes (variable labels)

    Returns
    -------
    g_hat : corresponding networkx graph

    '''
    g_hat = nx.Graph()
    g_hat.add_nodes_from(nodes)
    for var in range(array.shape[0]):
        # if array[var,var]>0 :
        #     g_hat.add_edges_from([(nodes[var],nodes[var])])
        for col in range(array.shape[1]):
            if (nodes[var] != nodes[col]):
                if (array[var,col] >0):
                    g_hat.add_edges_from([(nodes[var],nodes[col])])
    return g_hat

def generate_varnames(nb_nodes, Lmax = 0):
    '''
    A function to automatically generate names of variables for a dataset containing
    a number nb_nodes of variables.
    
    Parameters
    ----------
    nb_nodes : number of nodes
    Lmax: if Lmax >0, then the function generates a lsit of lagged variables (default = 0)
    
    Returns
    -------
    l : list of variables labels $X_{0}$...$X_{nb_nodes}$

    '''
    
    l = []
    if Lmax ==0:
        for j in range(nb_nodes):
            l.append('$X_{'+str(j)+'}$')
    else:
        for j in range(nb_nodes):
            for t in range(Lmax+1):
                if t ==0:
                    l.append('$X_{t}^{'+str(j)+'}$')
                else:
                    l.append('$X_{t-'+str(t)+'}^{'+str(j)+'}$')
    return l




def create_TS_graph(nbnodes):
    '''
    inputs :
        nodes : list of nodes, the last in the list is the response variable Y
        
    output :
        A partially completed graph from TS, from potential causes to 
        the response variable
    '''
    
    ## form the adjacency matrix :
    A = np.zeros((nbnodes,nbnodes))
    A[:,-1] = 1
    g = array_to_graph(A,generate_varnames(nbnodes))
    return g




def create_complete_digraph(nodes):
    """Create a complete graph from the list of node ids.
    Args:
        node_ids: a list of node ids
    Returns:
        An undirected graph (as a networkx.Graph)
        
    Anto : En vrai, networkx a deja une fonction qui fait ça :  nx.complete_graph(nb_noeuds)
    _create_complete_graph permet juste de choisir le nom des noeuds, mais si je crois que celle de nx le fait aussi
    """
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    ## Combinations(noeuds, 2) : fait toutes les combinaisons avec 2 éléments possibles entre les noeuds
    for (i, j) in permutations(nodes, 2):
        g.add_edge(i, j, weight =  1)
    return g

# =============================================================================
# PCPMIME
# =============================================================================


def PCPMIME_all_out(data, verbose = 1 ,**kwargs ):
    '''
    data : dataframe of time series size (N x g+1), 
        N = time series length, g : number of potentially causal variables
        The last column of data = response variable (Y)
    kwargs : see parameters of PMIME
    
    We consider that there exists links between the inputs. 
            
    '''
    # g = data.shape[1]
    
    #nodes = generate_varnames(g) ## list of nodes [X_0,X_1...,X_g]
    nodes = data.columns
    
    G = create_complete_digraph(nodes) ## complete graph
    #dep_l = nodes[:-1]  ## dependency list : variables that potentially cause Y
    remove_edge = [] # unconomment for 'PC-stable'
    dfsep_set = pd.DataFrame(columns = nodes, index= nodes)
    for (effect,node) in permutations(nodes,2):
        if verbose :
            print('X = ',node, 'Y =',effect)
        ## measure R, the link between 'node' and the response var
        #r = pm.MIME(data,node,effect,**kwargs)
        r = pm.PMIME_measure(data,node,effect,Z=[],**kwargs)
        print('measure r :', r)
        G[node][effect]['weight'] = r
        
        ## if r is too small, then the link is not significant
        # then remove the edge 
        if r < 1*10**(-10) :
            remove_edge.append((node,effect)) #unconomment for 'PC-stable'
            #G.remove_edge((node,effect)) # comment for PC stable
            print('remove the edge from ', node, 'to', effect,'\n')
    

    
    G.remove_edges_from(remove_edge)
    l =1
    stop = 0
    
    
    while stop ==0 : 
        stop =1
         ## conditionning set size
        print('new loop l = ', l)
        remove_edge = [] # unconomment for 'PC-stable'
        
        for (effect,node) in permutations(nodes,2):
            adj_resp = list(G.predecessors(effect)) ## nodes pointing to effect var
            
            if node not in adj_resp :
                continue
            else :
                adj_resp.remove(node)
            
            if len(adj_resp) >= l: ## stopping criterion 

                for parents in combinations(adj_resp,l) :
                    parents = list(parents)
                    if verbose :
                        print('X = ',node, 'Y =',effect, 'Z =',parents)
                    # cond = np.where(G.has_edge(tmp_node,effect) for tmp_node in parents)[0]
                    r = pm.PMIME_measure(data, node,effect,Z = parents,verbose = verbose, **kwargs)
                    print('r =', r)
                    
                    if r < 1*10**(-10) :  
                        # if G.has_edge(node,effect):
                            # G.remove_edge(node,effect)  # comment for PC stable
                        remove_edge.append((node,effect))  #unconomment for 'PC-stable'
                        remove_edge.append((effect,node))  #unconomment for 'PC-stable'
                        print('remove the edge between ', node, 'and', effect,'\n')
                        dfsep_set[node][effect] = set(parents)
                        
                        break

                stop = 0
        
        G.remove_edges_from(remove_edge)
        if stop ==1:
            break
        l +=1
        
    return G,dfsep_set


if __name__ == '__main__':
    import timeit
    import numpy.random as rand
    
    n = 2000
    X=1/10*np.ones((n,4))
    X[1,1] = np.random.randn()
    for j in range(1,n):
        X[j,0] = np.random.random()*X[j-1,0] + 0.01* np.random.randn()
   
    for k in range(1,n):
        X[k,2] =  np.random.uniform() *X[k-1,2] + 0.01*np.random.randn()
    for k in range(1,n):
        X[k,3] = np.random.uniform()*X[k-1,3] + 0.01*np.random.randn()
    for t in range(2,n):
        X[t,1] = np.random.uniform()*np.tanh(X[t-3,0])**2  + np.random.uniform()*X[t-2,2]**2 +  np.random.uniform()*X[t-1,1] + np.random.randn()
        #X[t,1] =  np.random.uniform()*X[t-2,2]**2 +  np.random.uniform()*X[t-1,1] + np.random.randn()
    df = pd.DataFrame(data = X, columns = ['X_0','X_1','X_2','X_3'])

    
    # all_lagM, indlagM = build_lagged_matrix(df.values[:-1,:], 5, 1)
    
    #print('PMIME :')
    #r1_ksg = PMIME(df,'X_0','X_1', Z = [], sig_pmime = 0.05)
    #print('measure :',r1_ksg)
    # print('\n PMIME bootstrap :')
    # for j in range(10):
    mat = np.zeros((4,4))
    mat[0,[1,2]] = 1
    mat[2,1] = 1
    G_true = array_to_graph(mat, nodes = ['X_0','X_1','X_2','X_3'])
    k = np.int64(0.03*n)    
    # timebs = []
    # timenb = []
    # F1_bs = []
    # F1_nb = []
# for j in range(6):
    # start1 = timeit.default_timer()
    # Ghat1,dfs= PCPMIME_all_out(df,bootstrap = True, A = 0.03, nnei = k)
    # stop1 = timeit.default_timer()
    # F1_bs +=[pp.F1_score(Ghat1,G_true) ]
    start2 = timeit.default_timer()
    Ghat2,dfsp= PCPMIME_all_out(df,bootstrap = False, A = 0.03, nnei = k)
    stop2 = timeit.default_timer()
    # F1_nb +=[pp.F1_score(Ghat2,G_true) ]
    # timebs.append(stop1-start1)
    # timenb.append(stop2-start2)
        
    # print('bootstrapped time:',stop1-start1)
    # print('not bootstrapped time:',stop2-start2)
        
