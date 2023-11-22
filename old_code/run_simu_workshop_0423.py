# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:19:51 2023

@author: Antonin


simulation paper workshop april 2023
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PCPMIME as tpc
from PCPMIME import generate_varnames
import networkx as nx
# import PMIME_for_PC as pm
# from PMIME import *
import OO_PMIME as pmime_tig

import pickle

import causal_discovery_class as cd
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb


def array_to_graph(array, nodes):
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

def generate_varnames(nb_nodes):
    '''
    Parameters
    ----------
    nb_nodes : number of nodes

    Returns
    -------
    l : list of variables labels $X_{0}$...$X_{nb_nodes}$

    '''
    l = []
    for j in range(nb_nodes):
        l.append('$X_{'+str(j)+'}$')
    return l

def get_ground_truth(structure, nodes):
    '''
    function from https://github.com/ckassaad/causal_discovery_for_time_series
    
    to get the true graphs of the several datasets
    '''
    gtrue = nx.DiGraph()
    gtrue.add_nodes_from(nodes)
    ogtrue = nx.DiGraph()
    ogtrue.add_nodes_from(nodes)
    sgtrue = nx.DiGraph()
    tgtrue = nx.DiGraph()
    tgtrue.add_nodes_from(nodes)

    sgtrue.add_nodes_from(nodes)
    for i in range(len(nodes)):
        sgtrue.add_edges_from([(nodes[i], nodes[i])])

    if (structure == "fork") or (structure == "fork_big_lag"):
        ogtrue.add_edges_from([(nodes[0], nodes[1]), (nodes[0], nodes[2])])
        tgtrue.add_edges_from([(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2]), (nodes[0], nodes[1]),
                               (nodes[0], nodes[2])])
        tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
        tgtrue.edges[nodes[0], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[0], nodes[2]]['time'] = [2]
    elif structure == ("v_structure") or (structure == "v_structure_big_lag"):
        ogtrue.add_edges_from([(nodes[0], nodes[2]), (nodes[1], nodes[2])])
        tgtrue.add_edges_from([(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2]), (nodes[0], nodes[2]),
                               (nodes[1], nodes[2])])
        tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
        tgtrue.edges[nodes[0], nodes[2]]['time'] = [2]
        tgtrue.edges[nodes[1], nodes[2]]['time'] = [1]
    elif structure == ("mediator") or (structure == "mediator_big_lag"):
        ogtrue.add_edges_from([(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[0], nodes[2])])
        tgtrue.add_edges_from([(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2]), (nodes[0], nodes[1]),
                               (nodes[0], nodes[2]), (nodes[1], nodes[2])])
        tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
        tgtrue.edges[nodes[0], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[0], nodes[2]]['time'] = [2]
        tgtrue.edges[nodes[1], nodes[2]]['time'] = [1]
    elif structure == ("diamond") or (structure == "diamond_big_lag"):
        ogtrue.add_edges_from([(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], nodes[3]), (nodes[2], nodes[3])])
        tgtrue.add_edges_from([(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2]), (nodes[3], nodes[3]),
                               (nodes[0], nodes[1]), (nodes[0], nodes[2]),
                              (nodes[1], nodes[3]), (nodes[2], nodes[3])])
        tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
        tgtrue.edges[nodes[3], nodes[3]]['time'] = [1]
        tgtrue.edges[nodes[0], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[0], nodes[2]]['time'] = [2]
        tgtrue.edges[nodes[1], nodes[3]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[3]]['time'] = [1]
    elif structure == "7ts0h":
        ogtrue.add_edges_from([(nodes[1], nodes[0]), (nodes[2], nodes[1]), (nodes[3], nodes[2]), (nodes[3], nodes[4]),
                              (nodes[4], nodes[5]), (nodes[5], nodes[6])])
        tgtrue.add_edges_from([(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2]), (nodes[3], nodes[3]),
                               (nodes[4], nodes[4]), (nodes[5], nodes[5]), (nodes[6], nodes[6]),
                               (nodes[1], nodes[0]), (nodes[2], nodes[1]), (nodes[3], nodes[2]), (nodes[3], nodes[4]),
                               (nodes[4], nodes[5]), (nodes[5], nodes[6])])
        tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
        tgtrue.edges[nodes[3], nodes[3]]['time'] = [1]
        tgtrue.edges[nodes[4], nodes[4]]['time'] = [1]
        tgtrue.edges[nodes[5], nodes[5]]['time'] = [1]
        tgtrue.edges[nodes[6], nodes[6]]['time'] = [1]

        tgtrue.edges[nodes[1], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[3], nodes[2]]['time'] = [1]
        tgtrue.edges[nodes[3], nodes[4]]['time'] = [1]
        tgtrue.edges[nodes[4], nodes[5]]['time'] = [1]
        tgtrue.edges[nodes[5], nodes[6]]['time'] = [1]
    elif structure == "7ts2h":
        ogtrue.add_edges_from([(nodes[1], nodes[0]), (nodes[2], nodes[1]), (nodes[3], nodes[2]), (nodes[3], nodes[4]),
                              (nodes[4], nodes[5]), (nodes[5], nodes[6]),
                               (nodes[0], nodes[5]), (nodes[5], nodes[0]), (nodes[1], nodes[6]), (nodes[6], nodes[1])])
        tgtrue.add_edges_from([(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[2], nodes[2]), (nodes[3], nodes[3]),
                               (nodes[4], nodes[4]), (nodes[5], nodes[5]), (nodes[6], nodes[6]),
                               (nodes[1], nodes[0]), (nodes[2], nodes[1]), (nodes[3], nodes[2]), (nodes[3], nodes[4]),
                               (nodes[4], nodes[5]), (nodes[5], nodes[6]), (nodes[0], nodes[5]), (nodes[5], nodes[0]),
                               (nodes[6], nodes[2]), (nodes[2], nodes[6])])
        tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[2]]['time'] = [1]
        tgtrue.edges[nodes[3], nodes[3]]['time'] = [1]
        tgtrue.edges[nodes[4], nodes[4]]['time'] = [1]
        tgtrue.edges[nodes[5], nodes[5]]['time'] = [1]
        tgtrue.edges[nodes[6], nodes[6]]['time'] = [1]

        tgtrue.edges[nodes[1], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[2], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[3], nodes[2]]['time'] = [1]
        tgtrue.edges[nodes[3], nodes[4]]['time'] = [1]
        tgtrue.edges[nodes[4], nodes[5]]['time'] = [1]
        tgtrue.edges[nodes[5], nodes[6]]['time'] = [1]

        tgtrue.edges[nodes[0], nodes[5]]['time'] = [0]
        tgtrue.edges[nodes[5], nodes[0]]['time'] = [0]
        tgtrue.edges[nodes[6], nodes[2]]['time'] = [0]
        tgtrue.edges[nodes[2], nodes[6]]['time'] = [0]
    elif structure == "pair":
        ogtrue.add_edges_from([(nodes[0], nodes[1])])
        tgtrue.add_edges_from([(nodes[0], nodes[0]), (nodes[1], nodes[1]), (nodes[0], nodes[1])])
        tgtrue.edges[nodes[0], nodes[0]]['time'] = [1]
        tgtrue.edges[nodes[1], nodes[1]]['time'] = [1]
        tgtrue.edges[nodes[0], nodes[1]]['time'] = [1]

    elif structure == "indep_pair":
        ogtrue.add_edges_from([(nodes[0], nodes[1])])
        tgtrue.add_edges_from([(nodes[0], nodes[1])])
        tgtrue.edges[nodes[0], nodes[1]]['time'] = [1]

        sgtrue = nx.DiGraph()
        sgtrue.add_nodes_from(nodes)

    gtrue.add_edges_from(ogtrue.edges)
    gtrue.add_edges_from(sgtrue.edges)
    return gtrue, ogtrue, sgtrue, tgtrue

def TP(g_hat,gtrue):
    inter =  nx.intersection(gtrue, g_hat)
    return len(inter.edges)

def FP(g_hat,gtrue):
    dif =  nx.difference(g_hat,gtrue)
    return len(dif.edges)

def FN(g_hat,gtrue):
    dif =  nx.difference(gtrue,g_hat)
    return len(dif.edges)


def precision(g_hat,gtrue):
    tp = TP(g_hat,gtrue)
    fp = FP(g_hat,gtrue)
    if tp+fp == 0:
        return 10**(-9)
    return tp/(tp+fp)

def recall(g_hat,gtrue):
    tp = TP(g_hat,gtrue)
    fn = FN(g_hat,gtrue)
    if tp+fn == 0:
        return 10**(-9)
    return tp/(tp+fn)

def F1_score(g_hat,gtrue):
    p = precision(g_hat,gtrue)
    r = recall(g_hat,gtrue)
    if p==0 and r==0:
        return 0
    return (2*p*r)/(p+r)

def pcmci_dict(data, tau_max=5, cond_ind_test="CMIknn", alpha=0.05):
    if cond_ind_test == "CMIknn":
        cond_ind_test = CMIknn()
    elif cond_ind_test == "ParCorr":
        cond_ind_test = ParCorr()

    data_tigramite = pp.DataFrame(data.values, var_names=data.columns)

    pcmci = PCMCI(
        dataframe=data_tigramite,
        cond_ind_test=cond_ind_test,
        verbosity=2)
    pcmci.run_pcmci(tau_min=0, tau_max=tau_max, pc_alpha=alpha)

    res_dict = dict()
    for effect in pcmci.all_parents.keys():
        res_dict[pcmci.var_names[effect]] = []
        for cause, t in pcmci.all_parents[effect]:
            res_dict[pcmci.var_names[effect]].append((pcmci.var_names[cause], t))


    return res_dict


def tgraph_to_graph(tg):
    g = nx.DiGraph()
    og = nx.DiGraph()
    sg = nx.DiGraph()
    g.add_nodes_from(tg.nodes)
    og.add_nodes_from(tg.nodes)
    sg.add_nodes_from(tg.nodes)
    for cause, effects in tg.adj.items():
        for effect, _ in effects.items():
            if cause != effect:
                og.add_edges_from([(cause, effect)])
                g.add_edges_from([(cause, effect)])
            else:
                sg.add_edges_from([(cause, effect)])
                g.add_edges_from([(cause, effect)])
    return g, og, sg

class TemporalGraphicalModel():
    def __init__(self, nodes):
        nodes = nodes
        self.tghat = nx.DiGraph()
        self.tghat.add_nodes_from(nodes)

    def infer_from_data(self, data):
        raise NotImplementedError

    def _dict_to_tgraph(self, temporal_dict):
        for name_y in temporal_dict.keys():
            for name_x, t_xy in temporal_dict[name_y]:
                if (name_x, name_y) in self.tghat.edges:
                    self.tghat.edges[name_x, name_y]['time'].append(-t_xy)
                else:
                    self.tghat.add_edges_from([(name_x, name_y)])
                    self.tghat.edges[name_x, name_y]['time'] = [-t_xy]
                # self.TGhat.add_weighted_edges_from([(name_x, name_y, t_xy)])
    


    def _tgraph_to_graph(self):
        self.ghat, self.oghat, self.sghat = tgraph_to_graph(self.tghat)

class cPCMCI(TemporalGraphicalModel):
    def __init__(self, nodes, sig_level=0.05, nlags=5, cond_ind_test="CMIknn"):
        TemporalGraphicalModel.__init__(self, nodes)
        self.sig_level = sig_level
        self.nlags = nlags
        self.cond_ind_test = cond_ind_test

    def infer_from_data(self, data):
        data.columns = list(self.tghat.nodes)
        tg_dict = pcmci_dict(data, tau_max=self.nlags, cond_ind_test=self.cond_ind_test, alpha=self.sig_level)
        self._dict_to_tgraph(tg_dict)
        self._tgraph_to_graph()




#%%

def run_on_all(struct, path, save_path = None,save = False):
    len_ts = [125,250,500,1000,2000,4000]
    df_pcpmime = pd.DataFrame(columns = len_ts, index = range(10))
    df_gpw = pd.DataFrame(columns = len_ts, index = range(10))
    df_varl = pd.DataFrame(columns = len_ts, index = range(10))
    df_dyno = pd.DataFrame(columns = len_ts, index = range(10))
    df_pcmci = pd.DataFrame(columns = len_ts, index = range(10))
    
    for N in len_ts :
        F1pcpmime_tmp = []
        F1gpw_tmp = []
        F1varl_tmp = []
        F1dyno_tmp = []
        F1pcmci_tmp = []
        for j in range(10):
            data = pd.read_csv(path +'/'+str(struct)+'/data_'+str(j)+'.csv')
            data = data[:N]
            data = data.set_index('time_index')
            var_names = data.columns
            gtrue, ogtrue, sgtrue, tgtrue = get_ground_truth(str(struct), var_names)           
            
            ## run PCPMIME
            G_p,sepset = tpc.PCPMIME_all_out(data, Lmax = 3, nnei = int(0.01*N)+1, verbose = 0)

            # g = len(data.columns)
            ## model = PWGC
            model_gc = cd.GrangerPW(data.columns, sig_level=0.03, nlags=3)
            model_gc.infer_from_data(data)
            G_g = model_gc.ghat
            
            # ## model = Multivariate Granger causality
            # model_gc = cd.GrangerMV2(data.columns, sig_level=0.03, nlags=3)
            # model_gc.infer_from_data(data)
            # G_g = model_gc.ghat
            
            ## model = varlingam
            model_v = cd.VarLiNGAM(data.columns, sig_level=0.05, nlags=3)
            model_v.infer_from_data(data)
            G_v = model_v.ghat
            
            ## model = DYNOTEARS
            model_d = cd.Dynotears(data.columns, sig_level = 0.05, nlags = 3)
            model_d.infer_from_data(data)
            G_d = model_d.ghat
            
            ## model = PCMCI ParCorr
            model_gc = cPCMCI(data.columns, sig_level=0.03, nlags=3, cond_ind_test = 'ParCorr')
            model_gc.infer_from_data(data)
            G_pc = model_gc.ghat
            
            F1pcpmime_tmp +=[F1_score(G_p,ogtrue)]
            F1gpw_tmp +=[F1_score(G_g,ogtrue)]
            F1varl_tmp +=[F1_score(G_v,ogtrue)]
            F1dyno_tmp +=[F1_score(G_d,ogtrue)]
            F1pcmci_tmp +=[F1_score(G_pc,ogtrue)]
            
        df_pcpmime[N] = F1pcpmime_tmp
        df_gpw[N] = F1gpw_tmp
        df_varl[N] = F1varl_tmp
        df_dyno[N] = F1dyno_tmp
        df_pcmci[N] = F1pcmci_tmp
        
    if save :
        df_pcpmime.to_pickle(save_path+'/df_pcpmime_k001_'+str(struct)+'.pkl')
        df_gpw.to_pickle(save_path+'/df_pwgc_'+str(struct)+'.pkl')
        df_varl.to_pickle(save_path+'/df_varl_'+str(struct)+'.pkl')
        df_dyno.to_pickle(save_path+'/df_dyno_'+str(struct)+'.pkl')
        df_pcmci.to_pickle(save_path+'/df_pcmciparcor_'+str(struct)+'.pkl')
    return df_pcpmime,df_gpw,df_varl,df_dyno,df_pcmci

# df_pcpmime,df_gpw,df_varl,df_dyno = run_on_all('fork', path = path_data,save = False)

# df_pcpmime,df_gpw,df_varl,df_dyno,df_pcmci = run_on_all('mediator', path = path_data,save = True, save_path = save_path)
# df_pcpmime,df_gpw,df_varl,df_dyno, df_pcmci = run_on_all('v_structure', path = path_data,save = True, save_path = save_path)
# df_pcpmime,df_gpw,df_varl,df_dyno,df_pcmci = run_on_all('diamond', path = path_data,save = True, save_path = save_path)
# df_pcpmime,df_gpw,df_varl,df_dyno,df_pcmci = run_on_all('fork', path = path_data,save = True, save_path = save_path)
df_pcpmime,df_gpw,df_varl,df_dyno,df_pcmci = run_on_all('7ts2h', path = path_data,save = True, save_path = save_path)




#%%

def plot_res(df, struct):
    x = df.columns
    y_mean = df.mean().values
    
    var = [df.var().values,df.var().values]
    print(var)
    fig,ax = plt.subplots(nrows = 1)
    ax.set_ylim([0,1])
    # ax.set_xscale('log')
    ax.errorbar(x,y_mean, yerr = var, fmt = ':o', color = 'orange', elinewidth = 0.7,capsize = 5, ecolor = 'orange')
    ax.set_ylabel('F1 score')
    ax.set_xlabel('length of the time series')
    plt.title('method:'+str( struct))
    #plt.title('structure : '+str(struct))
    plt.show()
    
    
plot_res(df_gpw,'struct')
# plot_res

#%%


# save_path = 
def plot_all(structure = 'None'):
    
    unpickle = open(save_path+'/df_pcpmime_k001_'+str(structure)+'.pkl', 'rb')
    df_pcpmime = pickle.load(unpickle)
    
    unpickle = open(save_path+'/df_varl_'+str(structure)+'.pkl', 'rb')
    df_varl = pickle.load(unpickle)
    
    unpickle = open(save_path+'/df_dyno_'+str(structure)+'.pkl', 'rb')
    df_dyno = pickle.load(unpickle)
    
    unpickle = open(save_path+'/df_pwgc_'+str(structure)+'.pkl', 'rb')
    df_gpw = pickle.load(unpickle)
    
    unpickle = open(save_path+'/df_pcmciparcor_'+str(structure)+'.pkl', 'rb')
    df_pcmci = pickle.load(unpickle)
    
    F1_pm = df_pcpmime
    x1 = F1_pm.columns
    y1_mean = F1_pm.mean().values
    
    var1 = [F1_pm.var().values,F1_pm.var().values]
    
    F1_var = df_varl
    x2 = F1_var.columns
    y2_mean = F1_var.mean().values
    
    var2 = [F1_var.var().values,F1_var.var().values]
    F1_gc = df_gpw
    x3 = F1_gc.columns
    y3_mean = F1_gc.mean().values
    
    var3 = [F1_gc.var().values,F1_gc.var().values]
    
    
    x4 = df_dyno.columns
    y4_mean = df_dyno.mean().values
    
    var4 = [df_dyno.var().values,df_dyno.var().values]
    
    x5 = df_pcmci.columns
    y5_mean = df_pcmci.mean().values
    
    var5 = [df_pcmci.var().values,df_pcmci.var().values]
    
    fig,ax = plt.subplots(nrows = 1)
    ax.set_ylim([0,1.001])
    # ax.set_xscale('log')
    ax.errorbar(x1,y1_mean, yerr = var1,label = 'PC-PMIME', fmt = ':o', color = 'orange', elinewidth = 0.7,capsize = 5, ecolor = 'orange')
    ax.errorbar(x2,y2_mean, yerr = var2, label = 'VarLiNGAM',fmt = ':o', color = 'lightsteelblue', elinewidth = 0.7,capsize = 5, ecolor = 'lightsteelblue')
    ax.errorbar(x3,y3_mean, yerr = var3, label = 'PWGC', fmt = ':o', color = 'salmon', elinewidth = 0.7,capsize = 5, ecolor = 'salmon')
    ax.errorbar(x4,y4_mean, yerr = var4, label = 'DYNOTEARS', fmt = ':o', color = 'blue', elinewidth = 0.7,capsize = 5, ecolor = 'blue')
    ax.errorbar(x5,y5_mean, yerr = var5, label = 'PCMCI ParCorr', fmt = ':o', color = 'green', elinewidth = 0.7,capsize = 5, ecolor = 'green')
    ax.set_ylabel('F1-score')
    ax.set_xlabel('length of the time series')
    plt.title('structure: '+str(structure))
    plt.legend(loc = 'best')
    plt.savefig(save_path+'/'+str(structure)+'.pdf',format = 'pdf')
    # plt.legend()
    #plt.title('structure : '+str(struct))
    plt.show()

plot_all(structure = '7ts2h')


# df_pcpmime.to_pickle(save_path+'/pcpmime_fork.pkl')
# df_gpw.to_pickle(save_path+'/gpw_fork.pkl')
# df_varl.to_pickle(save_path+'/varl_fork.pkl')
# df_dyno.to_pickle(save_path+'/dyno_fork.pkl')
