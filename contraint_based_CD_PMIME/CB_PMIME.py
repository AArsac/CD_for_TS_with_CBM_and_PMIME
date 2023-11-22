

# -*- coding: utf-8 -*-
"""
Discovery of summary causal graphs for time series: script implementing
the PC and FCI methods with PMIME measure of conditional dependence.
Parallelization is done across variables for the skeleton construction step and for the rule origin of causality.

code based on https://github.com/ckassaad/PCGCE/blob/main/baselines/scripts_python/python_packages/CITCE/citce.py

This presents an adaptation to use the FCI algorithm coded on python by C. K. Assaad with the measure of 
conditional dependence


"""

import networkx as nx
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from datetime import datetime


# from tigramite.independence_tests import CMIknn


from tqdm import tqdm

import PMIME as pm

class PseudoSummaryGraph:
    """
    Graph structure
    0: no edge
    1: a tail -
    2: arrow head ->
    """
    def __init__(self, nodes):
        super(PseudoSummaryGraph, self).__init__()
        self.nodes_present, self.nodes_past, self.map_names_nodes = self.get_nodes(nodes)
        self.ghat = nx.DiGraph()
        self.ghat.add_nodes_from(self.nodes_present + self.nodes_past)

        for node_present in self.nodes_present:
            for node_past in self.nodes_past:
                self.ghat.add_edges_from([(node_past, node_present)])
            for node_present_2 in self.nodes_present:
                if node_present != node_present_2:
                    self.ghat.add_edges_from([(node_present_2, node_present)])

        self.d = len(nodes)*2
        self.sep = dict()
        for tup in self.ghat.edges:
            self.sep[tup] = []

    @staticmethod
    def get_nodes(names):
        nodes_present = []
        nodes_past = []
        map_names_nodes = dict()
        for name_p in names:
            try:
                int(name_p)
                node_p_present = "V" + str(name_p) + '_t'
                node_p_past = "V" + str(name_p) + '_t-'
                nodes_present.append(node_p_present)
                nodes_past.append(node_p_past)
                map_names_nodes[name_p] = [node_p_past, node_p_present]

            except ValueError:
                node_p_present = str(name_p) + '_t'
                node_p_past = str(name_p) + '_t-'
                nodes_present.append(node_p_present)
                nodes_past.append(node_p_past)
                map_names_nodes[name_p] = [node_p_past, node_p_present]
        return nodes_present, nodes_past, map_names_nodes

    def add_sep(self, node_p, node_q, node_r):
        """
        :param p: index of a time series
        :param q: index of a time series
        :param r: index of seperation set
        """
        if node_p in self.ghat.predecessors(node_q):
            self.sep[(node_p, node_q)].append(node_r)
        if node_q in self.ghat.predecessors(node_p):
            self.sep[(node_q, node_p)].append(node_r)

    def number_par_all(self):
        """
        :return: dict of number of adjacencies per node
        """
        dict_num_adj = dict()
        for node_p in self.ghat.nodes:
            dict_num_adj[node_p] = len(list(self.ghat.predecessors(node_p)))
        return dict_num_adj

    def to_summary(self):
        """
        :return: summary graph
        """
        map_names_nodes_inv = dict()
        nodes = list(self.map_names_nodes.keys())
        for node in nodes:
            for node_t in self.map_names_nodes[node]:
                map_names_nodes_inv[node_t] = node

        ghat_summary = nx.DiGraph()
        ghat_summary.add_nodes_from(nodes)

        # unique_edges = []
        for (node_p_t, node_q_t) in self.ghat.edges:
            node_p, node_q = map_names_nodes_inv[node_p_t], map_names_nodes_inv[node_q_t]
            # if (node_p, node_q) not in unique_edges:
            if (node_p, node_q) not in ghat_summary.edges:
                # unique_edges.append((node_p, node_q))
                ghat_summary.add_edges_from([(node_p, node_q)])
        print(ghat_summary.edges)
        return ghat_summary


class DataframeGraph:
    """
    Graph structure
    0: no edge
    1: a tail -
    2: arrow head ->
    """
    def __init__(self):
        """
        :param d: number of nodes
        """
        # self.edges = np.subtract(np.ones([n, n]), np.eye(n))
        # self.edges = np.ones([d, d])
        self.df = pd.DataFrame
        self.sep = dict()
        self.d = 0

    def construct(self, pseudo_g):
        self.df = nx.to_pandas_adjacency(pseudo_g.ghat)
        # self.sep = np.zeros([d, d, d])
        self.d = len(self.df.columns)

        # self.df[self.df == 1] = 2
        for col_i in self.df.columns:
            for col_j in self.df.columns:
                if self.df[col_j].loc[col_i] == 1:
                    if (col_i in pseudo_g.nodes_past) and (col_j in pseudo_g.nodes_present):
                        self.df[col_j].loc[col_i] = 2
                        self.df[col_i].loc[col_j] = 1
                    else:
                        self.df[col_i].loc[col_j] = 1

    def del_edge(self, node_p, node_q):
        """
        :param p: index of a time series
        :param q: index of a time series
        """
        self.df[node_q].loc[node_p] = 0
        self.df[node_p].loc[node_q] = 0

    def to_undirected_graph(self):
        df = self.df.copy()
        for col_i in self.df.columns:
            for col_j in self.df.columns:
                if df[col_j].loc[col_i] > 0:
                    df[col_j].loc[col_i] = 1

        g = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
        return g

    def to_directed_graph(self):
        df = self.df.copy()
        for node_i in self.df.columns:
            for node_j in self.df.columns:
                if df[node_j].loc[node_i] == 2:
                    df[node_j].loc[node_i] = 1
                else:
                    df[node_j].loc[node_i] = 0

        # todo transform df to 0 1 df
        g = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
        return g

    def add_sep(self, node_p, node_q, node_r):
        """
        :param p: index of a time series
        :param q: index of a time series
        :param r: index of seperation set
        """
        g = self.to_directed_graph()
        if node_p in g.predecessors(node_q):
            self.sep[(node_p, node_q)].append(node_r)
        if node_q in g.predecessors(node_p):
            self.sep[(node_q, node_p)].append(node_r)


class RankingList:
    def __init__(self):
        self.val = np.array([])
        self.elem_p = np.array([])
        self.elem_q = np.array([])
        self.elem_r = []

    def add(self, node_p, node_q, val, nodes_r):
        """
        :param p: index of a time series
        :param q: index of a time series
        :param val: value of mutual information
        :param r: index of set of conditionals
        """
        self.val = np.append(self.val, val)
        self.elem_p = np.append(self.elem_p, node_p)
        self.elem_q = np.append(self.elem_q, node_q)
        self.elem_r.append(nodes_r)

    def sort(self, descending=True):
        """
        :param descending: (bool) sort ascending vs. descending. By default True
        """
        idx = np.argsort(self.val)
        if descending:
            idx = np.flip(idx)
        self.val = np.take_along_axis(self.val, idx, axis=0)
        self.elem_p = np.take_along_axis(self.elem_p, idx, axis=0)
        self.elem_q = np.take_along_axis(self.elem_q, idx, axis=0)
        sorted_elem_r = []
        for i in idx:
            sorted_elem_r.append(self.elem_r[i])
        self.elem_r = sorted_elem_r


def pmime(x , y , y_past , z=None, nb_neighbors = 10, Lmax = 5, bootstrap = False,verbose = 1,**kwargs):
    '''
    function that adapts PMIME measure to the FCIGCE use
    basic functions are imported form the PMIME_last file
    
    inputs :
        - x = vector or matrix of the interest variables to be tested 
        - y = vector of the target variable
        - y_past = past of the target
        - z = conditioning set
        - k = number of nearest neighbors for estimation of CMI
        
    outputs :
        - value returned by R = I(Y, w^x|w^y,w^z)/I(Y;w) = pval = val
    '''
    
    
       
    # Lmax = kwargs.get('Lmax', 5)
    
    stopping_threshold = kwargs.get('sig_pmime', 0.05)
    sig_CMI = kwargs.get('sig_CMI', 0.05)
    nb_bs = kwargs.get('nb_bs', 100)
    # nnei = kwargs.get('nnei', 5)
    
    dim_x = x.shape[1]
    dim_y = y.shape[1] 
    dim_y_past = y_past.shape[1]
    '''create lagged component of y'''
    if z is not None:
        z_df = pd.DataFrame()
        for k in z.keys():
            if isinstance(z[k], pd.Series):
                z[k] = z[k].to_frame()
            z_df[z[k].columns] = z[k].reset_index(drop=True)

        dim_z = z_df.shape[1]
        X = np.concatenate((x.values, y_past.values, z_df.values), axis=1)
        xyz = np.array([0] * dim_x + [1] * dim_y_past + [2] * dim_z)
        all_lags = dim_x+dim_y_past+dim_z
        # print('matrix X:\n',X[0,:])
    else:
        X = np.concatenate((x.values, y_past.values), axis=1)  ## X contains [ lagged variables of interest, target var, conditioning set ]
        xyz = np.array([0] * dim_x + [1] * dim_y_past)   ## index of each variable in X
        all_lags = dim_x+dim_y_past
    X = pm.normalize(X)
    
    '''
    ### building the embedding vector   ###
    '''
    embedding_vector_indices = []
    embedding_vector_indices.append( pm.first_cycle(X, y, num_neighbors = nb_neighbors))
    
    embedding_matrix = X[:,embedding_vector_indices]
    
    termination = 0
    SC_l = []
    while termination == 0:
        candidate_list = [i for i in ([j for j in range(all_lags)] + embedding_vector_indices) if i not in embedding_vector_indices]  # indices of candidates
        
        cmi_values = -999 * np.ones((all_lags,))
        
        for var in candidate_list:
            ## we compute CMI and MI for each candidate
            cmi_values[var] = pm.ksg_cmi(y, X[:, var], embedding_matrix, k= nb_neighbors)

            if cmi_values[var] == np.max(cmi_values):
                embedding_temp = var

        temp_embedding_matrix = np.column_stack((embedding_matrix, X[:, embedding_temp]))
        
        if bootstrap:
            p_value = pm.bootstrap_sig_cmi(temp_embedding_matrix, y, nb_bs, cmi_values[embedding_temp],nb_neighbors)

            if p_value < sig_CMI:
                embedding_vector_indices += [embedding_temp]
                embedding_matrix = temp_embedding_matrix
            else:
                termination = 1
        
        else:
            SC = cmi_values[embedding_temp] / pm.ksg_mi(y, temp_embedding_matrix, nb_neighbors)

            SC_l += [SC]
            
            ## Compare the ratio with the threshold
            if len(embedding_vector_indices) <= 2:
                # 2nd and 3rd embedding cycle to be tested
                if SC > stopping_threshold:
                    embedding_vector_indices += [embedding_temp]
                    embedding_matrix = temp_embedding_matrix
                else:
                    termination = 1
            else:
                # 4th embedding cycle to be tested --> additional condition to avoid 
                # false acceptance, if the ratio keeps increasing
                if SC > stopping_threshold and (SC < SC_l[-2]) and (SC < SC_l[-3]):
                    embedding_vector_indices += [embedding_temp]
                    embedding_matrix = temp_embedding_matrix
                else:
                    termination = 1
    
    # idxM: matrix of the true variable index in X + lag of each var
    #       implicated in the embedding vector
    idxM = np.nan * np.ones((len(embedding_vector_indices), 2))
    for j in range(len(embedding_vector_indices)):
        idxM[j, 0] = np.int8(np.ceil((embedding_vector_indices[j] + 1) / Lmax) - 1)  # var indices
        idxM[j, 1] = np.int8(np.mod(embedding_vector_indices[j], Lmax) + 1)  # lag indices for each variable
    
    
    ''' COMPUTE THE R MEASURE '''
    R = pm.compute_R(idxM, embedding_matrix, y, nnei=nb_neighbors)
    pval = R
    if verbose :
        if z !=None and z != {} :
            print('\n for x=',x.columns[0],' and y =',y.columns[0], 'conditioning on z=', z.keys(), 'R = ', R)
        else :
            print('\n for x=',x.columns[0],' and y =',y.columns[0], 'R = ', R)
    return pval, R



def get_sampling_rate(ts):
    # index of all non nan values in time series
    idx = np.argwhere(~np.isnan(ts).values)
    if len(idx) == len(ts):
        return True, 1
    # differentiate index, if there's no nan all values should be equal to 1
    diff = np.diff(idx, axis=0)
    udiff = np.unique(diff)
    if (len(udiff) == 1) and (udiff != 1):
        cd_bool = True
        cd_value = int(udiff)
    elif len(udiff) == 2:
        idx = np.argwhere(diff.reshape(-1) > 1)
        diff = diff[idx]
        udiff = np.unique(diff)
        if len(udiff) == 1:
            cd_bool = True
            cd_value = int(udiff)
        else:
            # ???
            cd_bool = False
            cd_value = np.nan
    else:
        # if there is no delay in the time series
        cd_bool = False
        cd_value = np.nan
    return cd_bool, cd_value


# def get_alpha(mts, k=10):
#     mi_list = []
#     for i in range(mts.shape[1]):
#         for t in range(100):
#             ts_i = mts[mts.columns[i]].dropna().to_frame()
#             ts_j = 0.05*ts_i + 0.95*np.random.randn(ts_i.shape[0], ts_i.shape[1])
#             pval, val = gtce(ts_i, ts_j,  k=k, p_value=False)
#             # pval, val = pmime(ts_i, ts_j,  nb_neihbors=k)
#             mi_list.append(val)
#     alpha = abs(max(mi_list))
    
#     return alpha





class CI:
    def __init__(self, series, sig_lev=0.05, lag_max=5, p_value=True, rank_using_p_value=False,
                 verbose=True, num_processor=-1, bs = False, sig_pmime = 0.03,knnei =0.01, nb_bs = 100, sig_CMI = 0.05):
        """
        Causal inference (Wrapper) using TCE (contain functions for skeleton construction)
        :param series: d-time series (with possibility of different sampling rate)
        :param sig_lev: significance level. By default 0.05
        :param p_value: Use p_value for decision making. By default True
        :param verbose: Print results. By default: True
        :param num_processor: number of processors for parallelization. By default -1 (all)
        """
        self.series = series
        self.n = series.shape[0]
        self.d = series.shape[1]
        self.names = self.series.columns
        self.graph = PseudoSummaryGraph(self.names)
        self.num_processor = num_processor
        ### PMIME args ###
        self.p_value = p_value
        self.lag_max = lag_max        
        self.sig_lev = sig_lev
        # self.alpha = get_alpha(series)
        self.alpha = 10**-3
        self.bootstrap = bs
        self.nb_bs = nb_bs
        self.sig_CMI = sig_CMI
        self.sig_pmime = sig_pmime  ## threshold stopping criterion of pmime embedding
        self.nnei = np.int32(knnei * self.n)+1  ## 
        if self.p_value == rank_using_p_value:
            self.rank_using_p_value = rank_using_p_value
        elif not rank_using_p_value:
            self.rank_using_p_value = rank_using_p_value
        else:
            print("Warning: rank_using_p_value can be True iff p_value is True. Using rank_using_p_value=False")
            self.rank_using_p_value = False
        self.verbose = verbose

        self.data_dict = dict()

        self.sampling_rate = dict()
        for name_p in self.names:
            _, s_r = get_sampling_rate(self.series[name_p])
            name_p_past, name_p_present = self.graph.map_names_nodes[name_p]
            self.sampling_rate[name_p_past] = s_r
            self.sampling_rate[name_p_present] = s_r


        
        for name_p in self.names:
            node_p_past, node_p_present = self.graph.map_names_nodes[name_p]
            self.data_dict[node_p_past], self.data_dict[node_p_present] = self.window_representation_past_present(name_p, node_p_past, node_p_present)
        
        self.mi_df = pd.DataFrame(np.ones([self.graph.d, self.graph.d]), columns=self.graph.ghat.nodes, index=self.graph.ghat.nodes)
        self.cmi_df = pd.DataFrame(np.ones([self.graph.d, self.graph.d]), columns=self.graph.ghat.nodes, index=self.graph.ghat.nodes)

        if self.verbose:
            print("n: "+str(self.n))
            print("d: "+str(self.d))
            print("names: "+str(self.names))
            print("sampling_rate: "+str(self.sampling_rate))
            # print("significance level:"+str(self.sig_lev))
            # print("alpha:"+str(self.alpha))
            print("threshold pmime: "+str(self.sig_pmime))

    def window_representation_past_present(self, name_p, node_p_past, node_p_present, overlap=True):
        ts = self.series[name_p].dropna()
        ts_window_past = pd.DataFrame()
        if self.lag_max == 0:
            ts_present = ts.rename(node_p_present).to_frame()
        else:
            for i in range(self.lag_max):
                i_data = ts[i:(ts.shape[0] - self.lag_max + i)].values
                ts_window_past.loc[:, node_p_past + str(self.lag_max - i)] = i_data
            ts_present = ts.rename(node_p_present).to_frame()
            ts_present = ts_present.loc[self.lag_max:]
            if not overlap:
                ts_window_past = ts_window_past.iloc[::ts_window_past, :]
        return ts_window_past, ts_present.reset_index(drop=True)

    def _pmime_pq(self, node_p, node_q):
        """
        estimate pmime between two time series
        :param p: time series with index p
        :param q: time series with index q
        :return: p, q and the estimated value of tmi(p,q)
        """
        x = self.data_dict[node_p]
        y = self.data_dict[node_q]
        
        y_past = self.data_dict[str(node_q)+'-']
        
        # print('node p, node q', node_p, node_q)
        # print('x :',x )
        # print('y :',y, '\n ')
        # mi_pval, mi_val = gtce(x, y, z=None, sampling_rate_tuple=(self.sampling_rate[node_p], self.sampling_rate[node_q]), p_value=self.p_value)
        pm_pval,pm_val = pmime( x,y,y_past, z = None, nb_neihbors = self.nnei,
                               Lmax = self.lag_max, verbose = self.verbose, sig_pmime = self.sig_pmime,
                               bootstrap= self.bootstrap, nb_bs = self.nb_bs, sig_CMI = self.sig_CMI)
        return node_p, node_q, pm_pval

    def skeleton_initialize(self):
        """
        initialize graph, remove all unconditional independencies and rank neighbors
        """
        if self.verbose:
            print("######################################")
            print("Skeleton Initialization")
            print("######################################")

        unique_edges = []
        for (node_p, node_q) in self.graph.ghat.edges:
            if (node_q, node_p) not in unique_edges:
                unique_edges.append((node_p, node_q))

        res = Parallel(n_jobs=self.num_processor,prefer = "threads")(delayed(self._pmime_pq)(node_p, node_q) for node_p, node_q in tqdm(unique_edges))
        # res = []
        # for (node_p, node_q) in tqdm(unique_edges):
        #     res.append(self._pmime_pq(node_p, node_q))

        for pq in range(len(res)):
            node_p, node_q, mi = res[pq][0], res[pq][1], res[pq][2]
            self.mi_df[node_p].loc[node_q] = mi
            self.mi_df[node_q].loc[node_p] = mi
            if self.verbose:
                print("p=" + node_p + "; q=" + node_q + "; I(p,q)=" + "{: 0.5f}".format(self.mi_df[node_p].loc[node_q]), end=" ")
            if self.p_value:
                if (self.data_dict[node_p].shape[1] > 1) or (self.data_dict[node_q].shape[1] > 1):
                    test = self.mi_df[node_p].loc[node_q] > self.sig_lev * 2
                else:
                    test = self.mi_df[node_p].loc[node_q] > self.sig_lev
            else:
                test = self.mi_df[node_p].loc[node_q] < self.alpha
            if test:
                if self.verbose:
                    print("=> Remove link between "+node_p+" and "+node_q)
                self.graph.ghat.remove_edge(node_p, node_q)
                if (node_p in self.graph.nodes_present) and (node_q in self.graph.nodes_present):
                    self.graph.ghat.remove_edge(node_q, node_p)
            else:
                if self.verbose:
                    print()

    def _cmi_sep_set_pq(self, node_p, node_q, set_size):
        """
        estimate ctmi between two time series conditioned on each set of neighbors with cardinality equal to set_size
        :param p: time series with index p
        :param q: time series with index q
        :param set_size: cardinality of the set of neighbors
        :return: p, q, list if estimated value of gtce(p,q,r_set), and list of all r_sets
        """
        print(" _cmi_sep_set_pq")

        v_list = []
        nodes_r = list(set(list(self.graph.ghat.predecessors(node_p)) + list(self.graph.ghat.predecessors(node_q))))
        if node_p in nodes_r:
            nodes_r.remove(node_p)
        if node_q in nodes_r:
            nodes_r.remove(node_q)
        nodes_r = [list(r) for r in itertools.combinations(nodes_r, set_size)]

        x = self.data_dict[node_p]
        y = self.data_dict[node_q]
        y_past = self.data_dict[str(node_q)+'-']
                    
        # print('x :',x)
        # print('y :', y)
        
        multi_dim = False
        for set_nodes_r in nodes_r:
            z = dict()
            for node_r in set_nodes_r:
                z[node_r] = self.data_dict[node_r]
                if node_r in self.graph.nodes_past:
                    multi_dim = True
            # print('z :', z)
            # cmi_pval, cmi_val = gtce(x, y, z, self.sampling_rate,  p_value=self.rank_using_p_value, multi_dim=multi_dim)
            if self.verbose:
                print('node p',node_p,' node q',node_q,' cond z', z.keys())
            if node_p == str(node_q+'-'):
                cmi_pval = 1
                cmi_val = 1
            else: 
                cmi_pval, cmi_val = pmime( x,y,y_past, z, nb_neihbors = self.nnei,
                                       Lmax = self.lag_max, verbose = self.verbose, sig_pmime = self.sig_pmime,
                                       bootstrap= self.bootstrap, nb_bs = self.nb_bs, sig_CMI = self.sig_CMI)
            if self.rank_using_p_value:
                v_list.append(cmi_pval)
            else:
                v_list.append(cmi_val)
        if v_list:
            return node_p, node_q, v_list, nodes_r

    def rank_cmi_sep_set_parallel(self, set_size):
        """
        rank pairs of time series based on the estimation of gtce between each pair of connected nodes
        :param set_size: cardinality of the set of neighbors
        :return: ranking of each pair of connected nodes based gtce
        """
        print("set_size " + str(set_size))

        dict_num_adj = self.graph.number_par_all()
        unique_qualified_edges = []
        for (node_p, node_q) in self.graph.ghat.edges:
            if (node_q, node_p) not in unique_qualified_edges:
                if dict_num_adj[node_p] + dict_num_adj[node_q] > set_size:
                    unique_qualified_edges.append((node_p, node_q))

        res = Parallel(n_jobs=self.num_processor, prefer = "threads")(delayed(self._cmi_sep_set_pq)(node_p, node_q, set_size) for node_p, node_q in
                                                  tqdm(unique_qualified_edges))
        # res = []
        # for (node_p, node_q) in tqdm(unique_qualified_edges):
        #     res.append(self._cmi_sep_set_pq(node_p, node_q, set_size))

        ranks = RankingList()
        for pq in range(len(res)):
            if res[pq] is not None:
                if isinstance(res[pq][2], list):
                    for r in range(len(res[pq][2])):
                        ranks.add(res[pq][0], res[pq][1], res[pq][2][r], res[pq][3][r])
                else:
                    ranks.add(res[pq][0], res[pq][1], res[pq][2], res[pq][3])
        if self.rank_using_p_value:
            ranks.sort(descending=True)
        else:
            ranks.sort(descending=False)
        return ranks

    def find_sep_set(self):
        """
        find the most contributing separation set (if it exists) between each pair of nodes
        """
        if self.verbose:
            print("######################################")
            print("Skeleton Separation")
            print("######################################")

        # print("max set size = " + str(self.graph.d-1))
        for set_size in range(1, self.graph.d-1):
            ranks = self.rank_cmi_sep_set_parallel(set_size)
            # if self.verbose:
            #     print("Ranking:")
            #     print("p: "+str(ranks.elem_p))
            #     print("q: " + str(ranks.elem_q))
            #     print("r: " + str(ranks.elem_r))
            #     print("val: " + str(ranks.val))
            for node_p, node_q, r_set, cmi in zip(ranks.elem_p, ranks.elem_q, ranks.elem_r, ranks.val):
                test = (node_p in self.graph.ghat.predecessors(node_q)) or (node_q in self.graph.ghat.predecessors(node_p))
                for node_r in r_set:
                    if not test:
                        break
                    test = test and ((node_r in self.graph.ghat.predecessors(node_q)) or (node_r in self.graph.ghat.predecessors(node_p)))
                if test:
                    mi = self.mi_df[node_p].loc[node_q]

                    if self.p_value != self.rank_using_p_value:
                        if self.verbose:
                            print('node p, node q', node_p, node_q)
                        x = self.data_dict[node_p]
                        y = self.data_dict[node_q]
                        y_past = self.data_dict[str(node_q)+'-']
                        z = dict()
                        multi_dim = False
                        for node_r in r_set:
                            z[node_r] = self.data_dict[node_r]
                            if node_r in self.graph.nodes_past:
                                multi_dim = True
                        # print('x :', x, '\n y :',y, 'node_q',node_q ,'\n z : ',z)
                        if node_p == str(node_q+'-'):
                            cmi = 1
                        else: 
                            cmi, _ = pmime( x,y,y_past, z, nb_neihbors = self.nnei,
                                                   Lmax = self.lag_max, verbose = self.verbose, sig_pmime = self.sig_pmime,
                                                   bootstrap= self.bootstrap, nb_bs = self.nb_bs,sig_CMI = self.sig_CMI)    
                        # cmi, _ = gtce(x, y, z, self.sampling_rate,p_value=self.p_value, multi_dim=multi_dim)
                        
                    if self.verbose:
                        print("p=" + node_p + "; q=" + node_q + "; r=" + str(r_set) + "; I(p,q|r)=" + "{: 0.5f}".format(
                            cmi) + "; I(p,q)=" + "{: 0.5f}".format(mi), end=" ")

                    if self.p_value:
                        if (self.data_dict[node_p].shape[1] > 1) or (self.data_dict[node_q].shape[1] > 1) or multi_dim:
                            test = mi < self.sig_lev * 2 < cmi
                        else:
                            test = mi < self.sig_lev < cmi
                    else:
                        test = cmi < self.alpha
                    if test:
                        self.cmi_df[node_p].loc[node_q] = cmi
                        self.cmi_df[node_q].loc[node_p] = cmi
                        if self.verbose:
                            print("=> remove link between " + node_p + " and " + node_q)

                        for node_r in r_set:
                            self.graph.add_sep(node_p, node_q, node_r)

                        self.graph.ghat.remove_edge(node_p, node_q)
                        if (node_p in self.graph.nodes_present) and (node_q in self.graph.nodes_present):
                            self.graph.ghat.remove_edge(node_q, node_p)

                    else:
                        if self.verbose:
                            print()


class PC(CI):
    def __init__(self, series, sig_lev=0.05, lag_max=5, p_value=True, rank_using_p_value=False, verbose=True, num_processor=-1,
                 bs = False, sig_pmime = 0.03,knnei =0.01, nb_bs = 100, sig_CMI = 0.05):
        """
        PC for time series using PMIME
        :param series: d-time series (with possibility of different sampling rate)
        :param sig_lev: significance level. By default 0.05
        :param p_value: Use p_value for decision making. By default True
        :param verbose: Print results. By default: True
        :param num_processor: number of processors for parallelization. By default -1 (all)
        """
        CI.__init__(self, series, sig_lev, lag_max, p_value, rank_using_p_value, verbose, num_processor,
                    bs, sig_pmime, knnei, nb_bs, sig_CMI)

    def rule_origin_causality(self):
        """
        rule 0 (origin of causality) from PC
        """
        if self.verbose:
            print("######################################")
            print("Rule Origin of Causality")
            print("######################################")

        edges_t = [(node_p, node_r) for (node_p, node_r) in self.graph.ghat.edges if (
                (node_p, node_r) in self.graph.ghat.edges) and (node_p in self.graph.nodes_present) and (
                node_r in self.graph.nodes_present)]
        for (node_p_t, node_r_t) in edges_t:
            nodes_q = [node_q for node_q in self.graph.ghat.predecessors(node_r_t) if (node_q != node_p_t) and (
                    node_q not in self.graph.ghat.predecessors(node_p_t)) and (
                    node_q not in self.graph.ghat.successors(node_p_t)) and (
                    node_r_t not in self.graph.sep[(node_q, node_p_t)])]
            for node_q in nodes_q:
                print(node_p_t, node_r_t, node_q)
                if ((node_q, node_r_t) in self.graph.ghat.edges) and ((node_p_t, node_r_t) in self.graph.ghat.edges):
                    if (node_r_t, node_p_t) in self.graph.ghat.edges:
                        self.graph.ghat.remove_edge(node_r_t, node_p_t)
                    if node_q in self.graph.nodes_present:
                        if (node_r_t, node_q) in self.graph.ghat.edges:
                            self.graph.ghat.remove_edge(node_r_t, node_q)

    def _find_shortest_directed_paths_util(self, p, q, visited, path, all_path):
        """
        sub function of _find_shortest_directed_paths
        :param p: index of time series
        :param q: index of time series
        :param visited: list of visited nodes
        :param path: current path
        :param all_path: list of all discovered paths
        """
        # Mark the current node as visited and store in path
        visited[p] = True
        path.append(p)

        # If current vertex is same as destination, then print
        # current path[]
        if p == q:
            if len(path) > 2:
                all_path.append(path.copy()[1:-1])
                return path
        else:
            # If current vertex is not destination
            # Recur for all the vertices child of this vertex
            child_p = np.where(self.graph.edges[p, :] == 2)[0]
            for k in child_p:
                if not visited[k]:
                    self._find_shortest_directed_paths_util(k, q, visited, path, all_path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[p] = False

    def _find_shortest_directed_paths(self, p, q):
        """
        find shortest directed path between time series of index p and time series of index q
        :param p: index of time series
        :param q: index of time series
        :return: all directed paths from p to q
        """
        # Mark all the vertices as not visited
        visited = [False] * self.d

        # Create an array to store paths
        path = []
        all_path = []

        # Call the recursive helper function to print all paths
        self._find_shortest_directed_paths_util(p, q, visited, path, all_path)
        return all_path

    def rule_2(self):
        """
        rule 2 from PC
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 3")
            print("######################################")
        test_find_orientation = False

        for i in range(self.graph.d):
            j_list = np.where(self.graph.edges[i, :] == 1)[0].tolist()
            if i in j_list:
                j_list.remove(i)
            for j in j_list:
                if self.graph.edges[j, i] == 1:
                    shortest_directed_path = self._find_shortest_directed_paths(i, j)
                    if len(shortest_directed_path) > 0:
                        self.graph.edges[i, j] = 2
                        test_find_orientation = True
                        if self.verbose:
                            print_path = '->'.join(map(str, shortest_directed_path[0]))
                            print(str(i)+"-"+str(j)+" and "+str(i) + "->" + print_path + "->" + str(j), end=" ")
                            print("=> orient " + str(i) + "->" + str(j))
        return test_find_orientation

    def rule_3(self):
        """
        rule 3 from PC
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 4")
            print("######################################")

        test_find_orientation = False

        for i in range(self.graph.d):
            for j in range(i + 1, self.graph.d):
                if (self.graph.edges[i, j] == 0) and (self.graph.edges[j, i] == 0):
                    colliders = [k for k in range(self.graph.d) if (k != i) and (k != j) and (
                            (self.graph.edges[j, k] == 2) and (self.graph.edges[i, k] == 2))]
                    k_list = [k for k in range(self.graph.d) if (k != i) and (k != j) and (
                            (self.graph.edges[j, k] == 1) and (self.graph.edges[i, k] == 1))
                              and (self.graph.edges[k, j] == 1) and (self.graph.edges[k, i] == 1)]
                    if len(colliders) > 0 and len(k_list) > 0:
                        for c in colliders:
                            for k in k_list:
                                if (self.graph.edges[c, k] == 1) and (self.graph.edges[k, c] == 1):
                                    test_find_orientation = True
                                    self.graph.edges[k, c] = 2
                                    if self.verbose:
                                        print(str(i) + "->" + str(c) + "<-" + str(j) + " and " + str(i) + "-" +
                                              str(k) + "-" + str(j) + " and " + str(k) + "-" + str(c),
                                              end=" ")
                                        print("=> orient " + str(k) + "->" + str(c))
        return test_find_orientation

    def fit(self):
        """
        run PC
        :return: graph (CPDAG)
        """
        if self.verbose:
            now = datetime.now()
            print("#######################################")
            print("########### Starting PC ###########")
            print("########### " + now.strftime("%H:%M:%S" + " ###########"))
            print("#######################################")

        # initialize skeleton
        self.skeleton_initialize()

        # get separation sets
        self.find_sep_set()

        # orientation
        self.rule_origin_causality()

        # test_rp = True
        # test_r2 = True
        # test_r3 = True
        # while test_rp or test_r2 or test_r3:
        #     test_rp = self.rule_propagation_causality()
        #     test_r2 = self.rule_2()
        #     test_r3 = self.rule_3()


        if self.verbose:
            print("######################################")
            print("Final Results (PC)")
            print("######################################")
            print("Extended Summary Graph:")
            print(self.graph.ghat.edges)
        return self.graph.ghat.edges


class FCI(CI):
    def __init__(self, series, sig_lev=0.05, lag_max=5, p_value=True, rank_using_p_value=False, verbose=True, num_processor=-1,
                 bs = False, sig_pmime = 0.03,knnei =0.01, nb_bs = 100, sig_CMI = 0.05):
        """
        FCI for time series using PMIME
        :param series: d-time series (with possibility of different sampling rate)
        :param sig_lev: significance level. By default 0.05
        :param p_value: Use p_value for decision making. By default True
        :param verbose: Print results. By default: True
        :param num_processor: number of processors for parallelization. By default -1 (all)
        """
        CI.__init__(self, series, sig_lev, lag_max, p_value, rank_using_p_value, verbose, num_processor, bs,
                    sig_pmime, knnei, nb_bs, sig_CMI)
        self.df_graph = DataframeGraph()

    def graph_to_df(self):
        if self.verbose:
            print("######################################")
            print("Graph to Dataframe")
            print("######################################")
        self.df_graph.construct(self.graph)
        print(self.df_graph.df)

    def _find_possible_d_sep_ij(self, node_i, node_j):
        """
        :param i: index of time series
        :param j: index of time series
        :return: all possible d-sep if i and j
        """
        possible_d_sep = []

        ug = self.df_graph.to_undirected_graph()
        dg = self.df_graph.to_directed_graph()
        for node in dg.nodes:
            if (node != node_i) and (node != node_j):
                if (node in nx.ancestors(dg, node_i)) and (node in nx.ancestors(dg, node_j)):
                    all_paths = list(nx.all_simple_paths(ug, source=node_i, target=node))
                    for path in all_paths:
                        test_colliders = True
                        triples_in_path = [path[i:i+3] for i in range(0, len(path)-2)]
                        for triple in triples_in_path:
                            if (triple[1] not in list(dg.successors(triple[0]))) or (
                                    triple[1] not in list(dg.successors(triple[2]))):
                                test_colliders = False
                                break
                        if test_colliders:
                            possible_d_sep.append(node)
                            break

        return possible_d_sep

    def _cmi_possible_d_sep_ij(self, node_p, node_q, set_size):
        """
        estimate ctmi between two time series conditioned on each possible-d-set with cardinality equal to set_size
        :param i: time series with index i
        :param j: time series with index j
        :param set_size: cardinality of the set of neighbors
        :return: i, j, list of estimated values of ctmi(p,q,possible-d-set), and list of all possible-d-sets
        """
        v_list = []
        k_list = self._find_possible_d_sep_ij(node_p, node_q)
        k_list = [list(k) for k in itertools.combinations(k_list, set_size)]
        if self.verbose:
            print('node_p, node_q, k_list', node_p, node_q, k_list)

        x = self.data_dict[node_p]
        y = self.data_dict[node_q]
        y_past = self.data_dict[str(node_q)+'-']
        # print('x :',x, '\n y :',y,'\n ')
        
        multi_dim = False

        for set_nodes in k_list:
            z = dict()
            # print("passe ici")
            for node_r in set_nodes:
                
                # print("bug1",k,'ks',ks,'self name k',self.names[k])
                # print('\n data dict',self.data_dict[self.names[k]])
                z[node_r] = self.data_dict[node_r]
                # z[self.names[k]] = self.data_dict[self.names[k]]
                # print("bug 2",z[self.names[k]])
                if node_r in self.graph.nodes_past:
                    multi_dim = True
            # print("bug 3")
            # print("z :",z)
            # cmi, _ = gtce(x, y, z, self.sampling_rate, p_value=self.rank_using_p_value, multi_dim=multi_dim)
            if node_p == str(node_q) +'-':
                cmi = 1
                # print("passe là")
            else :
                # print("passe là")
                cmi, _ = pmime( x,y,y_past, z, nb_neihbors = self.nnei,
                                Lmax = self.lag_max, verbose = self.verbose, sig_pmime = self.sig_pmime,
                                bootstrap= self.bootstrap, nb_bs = self.nb_bs, sig_CMI = self.sig_CMI)
                print(cmi)
            # cmi, _ = pmime(x, y, y_past, z, nb_neihbors =int(0.01*self.n)+1 , Lmax = self.lag_max)
            v_list.append(cmi)
        if v_list:

            return node_p, node_q, v_list, k_list

    def rank_possible_d_sep_parallel(self, set_size):
        """
        rank pairs of connected time series conditioned of their possible-d-sep based on the estimation of ctmi
        :param set_size: cardinality of the possible-d-sep
        :return: ranking of each pair of connected time series based ctmi
        """
        # list_adj, list_num_adj = self.graph.search_adj_all()
        # i_list = [[i]*list_num_adj[i] for i in range(len(list_num_adj)) if list_num_adj[i] > 0]
        # i_list = [i for sublist in i_list for i in sublist]
        # j_list = [list_adj[j] for j in range(len(list_num_adj)) if list_num_adj[j] > 0]
        # j_list = [j for sublist in j_list for j in sublist]

        dict_num_adj = self.graph.number_par_all()
        unique_qualified_edges = []
        for (node_p, node_q) in self.graph.ghat.edges:
            if (node_q, node_p) not in unique_qualified_edges:
                if dict_num_adj[node_p] + dict_num_adj[node_q] > set_size:
                    unique_qualified_edges.append((node_p, node_q))


        # res = Parallel(n_jobs=self.num_processor)(delayed(self._cmi_possible_d_sep_ij)(i, j, set_size) for i, j in
        #                                           zip(i_list, j_list))

        res = Parallel(n_jobs=self.num_processor, prefer = "threads")(delayed(self._cmi_possible_d_sep_ij)(node_p, node_q, set_size) for node_p, node_q in
                                                  tqdm(unique_qualified_edges))
        
        ranks = RankingList()
        for ij in range(len(res)):
            if res[ij] is not None:
                if isinstance(res[ij][2], list):
                    for k in range(len(res[ij][2])):
                        ranks.add(res[ij][0], res[ij][1], res[ij][2][k], res[ij][3][k])
                else:
                    ranks.add(res[ij][0], res[ij][1], res[ij][2], res[ij][3])
        if self.rank_using_p_value:
            ranks.sort(descending=True)
        else:
            ranks.sort(descending=False)
        return ranks

    def find_d_sep(self):
        """
        find the most contributing d sep (if it exists) between each pair of time series
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("d-seperation")
            print("######################################")
        test_remove_links = False

        for set_size in range(1, self.graph.d-1):
            ranks = self.rank_possible_d_sep_parallel(set_size)
            for node_i, node_j, ks, cmi in zip(ranks.elem_p, ranks.elem_q, ranks.elem_r, ranks.val):
                # node_i = self.df_graph.df.columns[i]
                # node_j = self.df_graph.df.columns[j]
                test = (self.df_graph.df[node_j].loc[node_i] != 0)
                for node_k in ks:
                    if not test:
                        break
                    # node_k = self.df_graph.df.columns[k]
                    test = test and ((self.df_graph.df[node_k].loc[node_j] != 0) or (self.df_graph.df[node_k].loc[node_i] != 0))
                if test:
                    mi = self.mi_df[node_j].loc[node_i]
                    if self.verbose:
                        print("i=" + str(node_i) + "; j=" + str(node_j) + "; z=" + str(ks) + "; I(i,j|z)=" + "{: 0.5f}".format(
                            cmi) + "; I(i,j)=" + "{: 0.5f}".format(mi), end=" ")

                    if self.p_value:
                        test = mi < self.sig_lev < cmi
                    else:
                        test = cmi < self.alpha
                    if test:
                        test_remove_links = True
                        self.cmi_df[node_j].loc[node_i] = cmi
                        self.cmi_df[node_i].loc[node_j] = cmi

                        if self.verbose:
                            print("=> remove link between " + str(node_i) + " and " + str(node_j))
                        self.df_graph.df[node_j].loc[node_i] = 0
                        self.df_graph.df[node_i].loc[node_j] = 0
                        for k in ks:
                            self.graph.add_sep(node_i, node_j, k)
                            self.graph.add_sep(node_j, node_i, k)
                    else:
                        if self.verbose:
                            print()
        return test_remove_links

    def remove_orientation(self):
        """
        turn all vertex into undetermined vertex
        """
        if self.verbose:
            print("######################################")
            print("Remove orientation")
            print("######################################")
        for node_i in self.df_graph.df.columns:
            for node_j in self.df_graph.df.columns:
                if node_i != node_j:
                    if self.df_graph.df[node_j].loc[node_i] != 0:
                        if (node_i in self.graph.nodes_past) and (node_j in self.graph.nodes_present):
                            self.df_graph.df[node_j].loc[node_i] = 2
                        else:
                            self.df_graph.df[node_j].loc[node_i] = 1

    def rule_origin_causality(self):
        """
        rule 0 (origin of causality) from FCI
        """
        if self.verbose:
            print("######################################")
            print("Rule Origin of Causality")
            print("######################################")
        for p in range(self.d):
            for q in range(p+1, self.d):
                node_p = self.df_graph.df.columns[p]
                node_q = self.df_graph.df.columns[q]
                if self.df_graph.df[node_q].loc[node_p] == 0:
                    for r in range(self.d):
                        node_r = self.df_graph.df.columns[r]
                        print(node_p, node_q, node_r, )
                        if (node_r != node_p) and (node_r != node_q) and (node_r not in self.graph.sep[(node_q, node_p)]) and (node_r not in self.graph.sep[(node_p, node_q)]):
                            if (self.df_graph.df[node_r].loc[node_q] != 0) and (self.df_graph.df[node_r].loc[node_p] != 0) and (
                                    self.df_graph.df[node_q].loc[node_r] != 0) and (self.df_graph.df[node_p].loc[node_r] != 0):
                                self.df_graph.df[node_r].loc[node_p] = 2
                                self.df_graph.df[node_r].loc[node_q] = 2

    def add_uncertainty(self):
        for node_p in self.df_graph.df.columns:
            for node_q in self.df_graph.df.columns:
                if self.df_graph.df[node_q].loc[node_p] == 1:
                    self.df_graph.df[node_q].loc[node_p] = 3

    def rule_propagation_causality(self):
        """
        rule 1 from FCI
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule Propagation of Causality")
            print("######################################")

        test_find_orientation = False

        for i in range(self.graph.d):
            for j in range(i + 1, self.graph.d):
                if (self.graph.edges[i, j] == 0) and (self.graph.edges[j, i] == 0):
                    k_list = [k for k in range(self.graph.d) if (k != i) and (k != j) and
                              (((self.graph.edges[j, k] == 2) and
                                (self.graph.edges[k, j] != 0) and (self.graph.edges[k, i] != 0) and
                                (self.graph.edges[i, k] == 3)) or ((self.graph.edges[i, k] == 2) and
                                                                   (self.graph.edges[k, i] != 0) and
                                                                   (self.graph.edges[k, j] != 0) and
                                                                   (self.graph.edges[j, k] == 3)))]
                    if len(k_list) > 0:
                        test_find_orientation = True
                        for k in k_list:
                            if self.graph.edges[i, k] == 2:
                                if self.verbose:
                                    print(str(i) + "*->" + str(k) + "°-*" + str(j), end=" ")
                                    print("=> orient " + str(i) + "*-> " + str(k) + " -> " + str(j))
                                self.graph.edges[k, j] = 2
                                self.graph.edges[j, k] = 1
                            else:
                                if self.verbose:
                                    print(str(j) + "*->" + str(k) + "°-*" + str(i), end=" ")
                                    print("=> orient " + str(j) + "*-> " + str(k) + " -> " + str(i))
                                self.graph.edges[k, i] = 2
                                self.graph.edges[i, k] = 1
        return test_find_orientation

    def _find_shortest_directed_paths(self, i, j):
        """
        find shortest directed path between time series of index i and time series of index j
        :param i: index of time series
        :param j: index of time series
        :return: all directed paths from i to j
        """
        g = self.df_graph.to_graph()
        all_path = nx.shortest_path(g, source=self.df_graph.df.columns[i], target=self.df_graph.df.columns[j])
        return all_path

    def rule_2(self):
        """
        rule 2 from FCI
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 3")
            print("######################################")
        test_find_orientation = False

        for i in range(self.graph.d):
            j_list = np.where(self.graph.edges[i, :] == 3)[0].tolist()
            if i in j_list:
                j_list.remove(i)
            for j in j_list:
                shortest_directed_path = self._find_shortest_directed_paths(i, j)
                if len(shortest_directed_path) > 0:
                    self.graph.edges[i, j] = 2
                    test_find_orientation = True
                    if self.verbose:
                        print_path = '*->'.join(map(str, shortest_directed_path[0]))
                        print(str(i)+"*-0"+str(j)+" and "+str(i) + "*->" + print_path + "*->" + str(j), end=" ")
                        print("=> orient " + str(i) + "*->" + str(j))
        return test_find_orientation

    def rule_3(self):
        """
        rule 3 from FCI
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 3")
            print("######################################")

        test_find_orientation = False

        for i in range(self.graph.d):
            for j in range(i + 1, self.graph.d):
                if (self.graph.edges[i, j] == 0) and (self.graph.edges[j, i] == 0):
                    colliders = [k for k in range(self.graph.d) if (k != i) and (k != j) and (
                            (self.graph.edges[j, k] == 2) and (self.graph.edges[i, k] == 2))]
                    k_list = [k for k in range(self.graph.d) if (k != i) and (k != j) and (
                            (self.graph.edges[j, k] == 3) and (self.graph.edges[i, k] == 3))]
                    if len(colliders) > 0 and len(k_list) > 0:
                        for c in colliders:
                            for k in k_list:
                                if self.graph.edges[k, c] == 3:
                                    test_find_orientation = True
                                    self.graph.edges[k, c] = 2
                                    if self.verbose:
                                        print(str(i) + "*->" + str(c) + "<-*" + str(j) + " and " + str(i) + "*-0" +
                                              str(k) + "0-*" + str(j) + " and " + str(k) + "*-0" + str(c),
                                              end=" ")
                                        print("=> orient " + str(k) + "*->" + str(c))
        return test_find_orientation


    # todo
    def _find_discriminating_paths_util(self, i, j, visited, path, all_path):
        """
        sub function of _find_shortest_directed_paths
        :param i: index of time series
        :param j: index of time series
        :param visited: list of visited nodes
        :param path: current path
        :param all_path: list of all discovered paths
        """
        # Mark the current node as visited and store in path
        visited[i] = True
        path.append(i)
        path.append(j)
        all_path.append(path.copy())
        path.pop()

        i_child = (self.graph.edges[i, :] == 2)
        i_parent = (self.graph.edges[:, i] == 2)
        j_adj1 = (self.graph.edges[:, j] != 0)
        j_adj2 = (self.graph.edges[j, :] != 0)
        next_i = np.where([a and b and c and d for a, b, c, d in zip(i_child, i_parent, j_adj1, j_adj2)])[0]

        for k in next_i:
            if not visited[k]:
                if (self.graph.edges[k, j] == 2) and (self.graph.edges[j, k] == 1):
                    self._find_shortest_directed_paths_util(k, j, visited, path, all_path)
                else:
                    visited[k] = True
                    path.append(k)
                    path.append(j)
                    all_path.append(path.copy())
                    path.pop()

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[i] = False

    def _find_discriminating_paths(self, i, j):
        """
        find discriminating  path between time series of index i and time series of index j
        :param i: index of time series
        :param j: index of time series
        :return: all discriminating paths from i to j
        """
        # Mark all the vertices as not visited
        visited = [False] * self.graph.d

        # Create an array to store paths
        path = [i]
        all_path = []

        i_child = (self.graph.edges[i, :] == 2)
        i_non_parent = (self.graph.edges[:, i] != 2)
        j_adj1 = (self.graph.edges[:, j] != 0)
        j_adj2 = (self.graph.edges[j, :] != 0)
        first_next_i = np.where([a and b and c and d for a, b, c, d in zip(i_child, i_non_parent, j_adj1, j_adj2)])[0]
        print(first_next_i)
        for f in first_next_i:
            # Call the recursive helper function to print all paths
            self._find_discriminating_paths_util(f, j, visited, path, all_path)

        return all_path

    def rule_4(self):
        """
        rule 4 from FCI
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 4")
            print("######################################")

        test_find_orientation = False

        for i in range(self.graph.d):
            for j in range(self.graph.d):
                if (i != j and self.graph.edges[i, j] == 0) and (self.graph.edges[j, i] == 0):
                    discriminating_paths = self._find_discriminating_paths(i, j)
                    for dp in discriminating_paths:
                        k = dp[-2]
                        if self.graph.edges[j, k] == 3:
                            self.graph.edges[j, k] = 1
                            self.graph.edges[k, j] = 2
                        else:
                            self.graph.edges[j, k] = 2
                            self.graph.edges[k, j] = 2
                            s = dp[-3]
                            self.graph.edges[s, k] = 2
                            self.graph.edges[k, s] = 2

        return test_find_orientation

    def _find_uncovered_path_util(self, i, j, i_2, i_1, visited, path, all_path):
        """
        sub function of _find_uncovered_path
        :param i: index of time series
        :param j: index of time series
        :param i_2: index of time series at before the previous iteration
        :param i_1: index of time series at the previous iteration
        :param visited: list of visited nodes
        :param path: current path
        :param all_path: list of all discovered paths
        :return:
        """
        # Mark the current node as visited and store in path
        visited[i] = True
        path.append(i)

        # If current vertex is same as destination, then print
        # current path[]
        if i == j:
            if len(path) > 2:
                if len(path) == 3:
                    print(i, i_2)
                    print(path)
                    print(self.graph.edges[i, i_2])
                    if self.graph.edges[i, i_2] == 0:
                        all_path.append(path.copy())
                else:
                    all_path.append(path.copy())
        else:
            # If current vertex is not destination
            # Recur for all the vertices child of this vertex
            child_i = np.where(self.graph.edges[i, :] != 0)[0]
            for k in child_i:
                if not visited[k]:
                    if len(path) > 2:
                        if self.graph.edges[i, i_2] == 0:
                            self._find_uncovered_path_util(k, j, i_1, i, visited, path, all_path)
                    elif len(path) == 2:
                        self._find_uncovered_path_util(k, j, i_2, i, visited, path, all_path)
                    else:
                        self._find_uncovered_path_util(k, j, i_2, i_1, visited, path, all_path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[i] = False

    def _find_uncovered_path(self, i, j):
        """
        find uncovered path between time series of index i and time series of index j
        :param i: index of time series
        :param j: index of time series
        :return: all uncovered paths from i to j
        """
        # Mark all the vertices as not visited
        visited = [False] * self.graph.d

        # Create an array to store paths
        path = []
        all_uncovered_path = []

        # Call the recursive helper function to print all paths
        self._find_uncovered_path_util(i, j, i, i, visited, path, all_uncovered_path)
        return all_uncovered_path

    def _is_potentially_directed(self, path):
        """
        check if path is a potentially directed path
        :param path: any path in the graph
        :return: bool
        """
        test_list1 = []
        for p in range(len(path)-1):
            test_list1.append((self.graph.edges[path[p+1], path[p]] != 2))
        return all(test_list1)

    def rule_8(self):
        """
        rule 8 from FCI
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 8")
            print("######################################")

        test_find_orientation = False

        for i in range(self.graph.d):
            for j in range(self.graph.d):
                if (self.graph.edges[i, j] == 2) and (self.graph.edges[j, i] == 3):
                    k_list = [k for k in range(self.graph.d) if (k != i) and (k != j) and
                              (((self.graph.edges[i, k] == 2) and (self.graph.edges[k, i] == 1) and
                                (self.graph.edges[k, j] == 2) and (self.graph.edges[j, k] == 1)) or
                               ((self.graph.edges[i, k] == 3) and (self.graph.edges[k, i] == 1) and
                                (self.graph.edges[k, j] == 2) and (self.graph.edges[j, k] == 1)))]
                    if len(k_list) > 0:
                        test_find_orientation = True
                        for k in k_list:
                            if self.verbose:
                                if self.graph.edges[i, k] == 3:
                                    print(str(i) + "-0" + str(k) + "->" + str(j) + " and "+str(i) + "0->" + str(j),
                                          end=" ")
                                    print("=> orient " + str(i) + " -> " + str(j))
                                else:
                                    print(str(i) + "->" + str(k) + "->" + str(j) + " and "+str(i) + "0->" + str(j),
                                          end=" ")
                                    print("=> orient " + str(i) + " -> " + str(j))
                            self.graph.edges[j, i] = 1
        return test_find_orientation

    def rule_9(self):
        """
        rule 9 from FCI
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 9")
            print("######################################")
        test_find_orientation = False
        for i in range(self.graph.d):
            for j in range(self.graph.d):
                if (self.graph.edges[i, j] == 2) and (self.graph.edges[j, i] == 3):
                    uncovered_path_list = self._find_uncovered_path(i, j)
                    if len(uncovered_path_list) > 0:
                        for p_d in uncovered_path_list:
                            if self._is_potentially_directed(p_d):
                                if self.graph.edges[p_d[-1], p_d[1]] == 0:
                                    test_find_orientation = True
                                    if self.verbose:
                                        print(str(i) + "0->" + str(j) + " and found a potential directed path", end=" ")
                                        print("=> orient " + str(i) + "->" + str(j))
                                    self.graph.edges[j, i] = 1
        return test_find_orientation

    def rule_10(self):
        """
        rule 10 from FCI
        :return: (bool) True if the rule made a change in the graph and False otherwise
        """
        if self.verbose:
            print("######################################")
            print("Rule 10")
            print("######################################")
        test_find_orientation = False
        for i in range(self.graph.d):
            for j in range(self.graph.d):
                if (self.graph.edges[i, j] == 2) and (self.graph.edges[j, i] == 3):
                    colliders_tails = [k for k in range(self.graph.d) if (k != j) and (
                            (self.graph.edges[k, j] == 2) and (self.graph.edges[j, k] == 1))]
                    colliders_tails = [list(k) for k in itertools.combinations(colliders_tails, 2)]
                    for ks in colliders_tails:
                        beta = ks[0]
                        theta = ks[1]
                        uncovered_path_list1 = self._find_uncovered_path(i, beta)
                        uncovered_path_list2 = self._find_uncovered_path(i, theta)
                        if (len(uncovered_path_list1) > 0) and (len(uncovered_path_list2) > 0):
                            for p1 in uncovered_path_list1:
                                if self._is_potentially_directed(p1):
                                    for p2 in uncovered_path_list2:
                                        if self._is_potentially_directed(p2):
                                            mu = p1[1]
                                            w = p2[1]
                                            if (mu != w) and (self.graph.edges[mu, w] == 0):
                                                test_find_orientation = True
                                                if self.verbose:
                                                    print(str(i) + "0->" + str(j) + " and ...", end=" ")
                                                    print("=> orient " + str(i) + "->" + str(j))
                                                self.graph.edges[j, i] = 1
        return test_find_orientation

    def fit(self):
        """
        run FCI
        :return: graph (PAG)
        """
        if self.verbose:
            print("#######################################")
            print("########### Starting FCI ###########")
            print("#######################################")

        # initialize skeleton
        self.skeleton_initialize()

        # get separation sets
        self.find_sep_set()

        # include circle in the skeleton
        self.graph_to_df()

        print(self.graph.ghat.edges)
        print(self.df_graph.df)
        # orientation
        self.rule_origin_causality()

        # find possible d sep
        test_remove_links = self.find_d_sep()

        # remove orientation
        if test_remove_links:
            self.remove_orientation()
            # orientation
            self.rule_origin_causality()
        # test_rp = True
        # test_r2 = True
        # test_r3 = True
        # test_r4 = True
        # while test_rp or test_r2 or test_r3 or test_r4:
        #     test_rp = self.rule_propagation_causality()
        #     test_r2 = self.rule_2()
        #     test_r3 = self.rule_3()
        #     test_r4 = self.rule_4()
        #
        # test_r8 = True
        # test_r9 = True
        # test_r10 = True
        # while test_r8 or test_r9 or test_r10:
        #     test_r8 = self.rule_8()
        #     test_r9 = self.rule_9()
        #     test_r10 = self.rule_10()
        self.graph.ghat = self.df_graph.to_directed_graph()


if __name__ == "__main__":
    

    # import sys 
    path_data = 'C:/Users/aa270673/Documents/Code/run_algo/data/simulated_ts_data'

    
    Df = pd.read_csv(path_data +'/7ts2h/data_0.csv')
    Df = Df[:2000]
    Df = Df.set_index('time_index')

    

    # ci_pmime = PC(Df, num_processor= 10, lag_max = 4, p_value = False , verbose = 0)
    fci_pmime = FCI(Df, num_processor= 10, lag_max = 3,p_value = False )
    # print(ci_pmime.graph.ghat.edges)
    # ci_pmime.fit()
    fci_pmime.fit()
    # print(ci_pmime.df_graph.df)
    # res_fci = ci_1.graph
    # nx.draw_circular(res_fci.to_summary(),with_labels = True)

