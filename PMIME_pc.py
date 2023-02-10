# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:27:55 2022
author : Antonin Arsac

This file is the for the measure of PMIME in the special PC case. 
MIME --> bivariate case
PMIME --> when a conditioning set is involved
"""


#### Libraries ####

import numpy as np
import pandas as pd
import networkx as nx


#bootstrap method from https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html
from arch.bootstrap import StationaryBootstrap, optimal_block_length
'''
block length optimisation :
    - Dimitris N. Politis & Halbert White (2004) Automatic Block-Length Selection for the Dependent Bootstrap

    - Andrew Patton , Dimitris N. Politis & Halbert White (2009) Correction to “Automatic Block-Length Selection for the Dependent Bootstrap” by D. Politis and H. White
'''


### knncmi is a package from https://github.com/omesner/knncmi for estimating CMI using knn
# import knncmi 
from scipy import spatial, special

from scipy.special import digamma
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

# knn CMI estimators from https://github.com/syanga/pycit/tree/master/pycit/estimators


path = 'C:/Users/AA270673/Documents/Code/PMIME/simulated_ts_data'



# =============================================================================

# # define the functions to estimate MI/CMI

# =============================================================================

def ksg_cmi(x_data, y_data, z_data, k=5):
    """
        KSG Conditional Mutual Information Estimator: I(X;Y|Z)
        See e.g. http://proceedings.mlr.press/v84/runge18a.html
        x_data: data with shape (num_samples, x_dim) or (num_samples,)
        y_data: data with shape (num_samples, y_dim) or (num_samples,)
        z_data: conditioning data with shape (num_samples, z_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """
    xzy_data = np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                               z_data.reshape(-1, 1) if z_data.ndim == 1 else z_data,
                               y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1)

    lookup = NearestNeighbors(metric='chebyshev')   ## chebyshev = max(|x-y|)
    lookup.fit(xzy_data)

    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    x_dim = x_data.shape[1] if x_data.ndim > 1 else 1
    z_dim = z_data.shape[1] if z_data.ndim > 1 else 1

    # compute entropies
    lookup.fit(xzy_data[:, :x_dim+z_dim])
    n_xz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy_data[:, x_dim:])
    n_yz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy_data[:, x_dim:x_dim+z_dim])
    n_z = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    return special.digamma(k) + np.mean(special.digamma(n_z+1.) - special.digamma(n_xz+1.) - special.digamma(n_yz+1.))

def ksg_mi(x_data, y_data, k=5):
    """
        KSG Mutual Information Estimator
        Based on: https://arxiv.org/abs/cond-mat/0305641
        x_data: data with shape (num_samples, x_dim) or (num_samples,)
        y_data: data with shape (num_samples, y_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """
    assert x_data.shape[0] == y_data.shape[0]
    num_samples = x_data.shape[0]

    lookup = NearestNeighbors(metric='chebyshev')
    lookup.fit(np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                               y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1))

    # estimate entropies
    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]

    radius = np.nextafter(radius[:, -1], 0)

    lookup.fit(x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data)
    n_x = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data)
    n_y = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    return special.digamma(num_samples) + special.digamma(k) - np.mean(special.digamma(n_x+1.) + special.digamma(n_y+1.))

# =============================================================================

############################################################################### 

# =============================================================================

def normalize(Mat):
    '''
    Function to normalise each value between 0 and 1 in an array Mat
    '''
    minallM = np.min(Mat, axis=0)
    maxallM = np.max(Mat, axis=0)
    rangeM = maxallM - minallM
    Mat = (Mat - minallM) / rangeM
    return Mat


def build_lagged_matrix(Mat, Lmax, T):
    '''
    Build the lagged matrix of all the variables as follows :
        Mat (N x K) --> all_lagM (N x (K*Lmax)) 
   [X_0,X_1,...,X_K]-->[X_t^0, X_{t-1}^0,..., X_{t-Lmax}^0, X_t^1,...,X_{t-Lmax}^K]
    
    '''
    N,K = Mat.shape
    wV = Lmax * np.ones(K).astype(int)
    all_lagM = np.zeros((N,np.sum(wV)))  ##lag matrix of all variables
    
    indlagM = np.zeros((K,2)).astype(int)    # start and end of columns of each variable in lag matrix
    
    cpt = 0
    
    ##  lag starts at 0 !!
    for var in range(K):

        indlagM[var,0] = cpt; indlagM[var,1] = cpt+wV[var]
    
        all_lagM[:, indlagM[var,0] ] = Mat[:,var] ## we fill the columns whose lag =0
    
        
        for lag in range(1,wV[var]):    #for lag = 1...Lmax-1
            all_lagM[lag:, indlagM[var,0]+lag ] = Mat[: - lag  ,var]
        cpt += wV[var]
    
    all_lagM = all_lagM[Lmax-1:-T , :] ## to start and end at the same time
    return all_lagM, indlagM


def first_cycle(XM,Y,Lmax,nnei):
    '''
    
    _____________________________
    ### first embedding cycle ###
    _____________________________

    As we use PMIME in a PC alg, we already know that X has a relation with Y
    Then we know that X is part of the embedding vector, we just have to
    know which lag is involved in the relation
    So the first embedding cycle in this case is just on the lags of X and those of Y
    --> Indeed, taking into account the past of Y in the measure of its relations with
    its environment might be relevant
    
    inputs: 
    XM: matrix of all the lagged variables
    Y: response vector
    Lmax : maximal lag considered (just for the loop actually)
    nnei: number of nearest neighbors for the MI estimation
    
    output:
    Index of the selected candidate in XM that maximises the mutual information
'''
    all_lagM = XM
    target = Y
    
    MI_esti = []
    for l_var in range(2*Lmax):    #2*Lmax --> the lags of X and Y in all_lagM
        ## compute the MI of future response and each one of
        ## the candidate lags using NN estimate

        MI_esti += [ksg_mi(target, all_lagM[:,l_var], nnei)]
        # MI_esti += [compute_mi(target, all_lagM[:,l_var], nnei)]
        if MI_esti[l_var]==np.max(MI_esti):
            pos_Max= l_var
    
    #print('MI max:', MI_esti[pos_Max])
     ##variable index of the one which maximises the MI
    return pos_Max

def bootstrap_sig_CMI(xtemp,Y,nb_bs,Max_cmi,nnei):
    '''
    Compute significance test for the value of the CMI, based on stationary block 
    bootstrap as in Politis and Romano (1994) with an optimised block length.
    p_value is computed as 
    inputs :
        -xtemp: matrix formed by the embedding vector at cycle j-1 and the candidate vector
            selected = [ xembM all_lagM[:,iemb_temp] ]
        -Y : response vector (Y_t+1)
        -nb_bs :  number of bootstrap replications (= nb of CMI computations)
        -Max_cmi : Value of the maximal CMI between the candidate vector and the response
            variable conditional on the embedding vector

        -nnei: number of nearest neighbors for the CMI estimation
    
    Output:
        p_value for the significance test
    
        
        
    '''
    target = Y
    block_l = optimal_block_length(xtemp[:,-1])['stationary']
    # print('block length :', block_l)
    bs = StationaryBootstrap(block_l, xtemp) 
    CMI_bsf = [] ## conditional mutual info for vector candidate
    #MIfuture = []   ## mutual info for past embedding vector
    #return xtemp,bs
    ## compute CMI on bootstrap data
    
    for data in bs.bootstrap(nb_bs):
        cand_bs = data[0][0][:,-1]  ##candidate vector = last column of xtemp 
        Z_bs = data[0][0][:,:-1]    ##
        CMI_bsf += [ksg_cmi(cand_bs, target, Z_bs, k=nnei)] ##CMI(candidate,Y|embedding vector)
        #MI_bsp += [ksg_cmi(target,xembM, k = nnei)]
    
    ## find indices where the CMI of the data are lower than the CMI over bootstrap distribution
    id_cand_bs = np.where(Max_cmi < CMI_bsf)[0]
    
    p_value = 1/nb_bs*len(id_cand_bs)
    #print('iembv:', iembV,'id_cand :', iemb_temp, 'p_value :', p_value)
    return p_value


def compute_R(id_l_M,xembM,target,nnei):
    ''' This function compute the R measure:
        R_{X->Y|Z } = I(xF;wX | wY, wZ)/I(xF,w) = I_sup/I_down
        w : the embedding vector ,
        wX(resp. wY,wZ) : elements from X (resp. Y,Z) in the embedding vector 
        
        inputs: 
            -id_l_M: matrix of the variables (col 0) and their lag (col 1) involved in the 
                embedding matrix
            -xembM: embedding matrix
            -target: response variable
            -nnei: number of nearest neighbors for the CMI estimation
        output:
            R measure
    
    '''     
    ## find indices of X,Y,Z in id_l_M respectively
    id_X = np.where(id_l_M[:,0]==0)[0]
    id_Y = np.where(id_l_M[:,0]==1)[0]
    id_Z = np.where(id_l_M[:,0]>1)[0]    
    
    
    ''' 
    build w_x,w_y,w_z elements from X (resp. Y,Z) in the embedding vector,
    as w_y and w_z are actually only used in the conditioning of the CMI,
    they are defined as w_cond = [w_y,w_z]
    '''
    w_X = xembM[:,id_X] ## values from X in the embedding vector
    
    if len(id_Y)>0 and len(id_Z)>0: 
        # conditioning set {wY,wZ}
        w_cond = xembM[:,np.concatenate((id_Y,id_Z))]
    elif len(id_Y)>0:
        w_cond = xembM[:,id_Y]
    elif len(id_Z)>0:
        w_cond = xembM[:,id_Z]
    else :
        w_cond = []
        print('no conditioning set')
    if  len(w_cond)==0 :
        ## compute the numerator if there is no conditioning set
        I_sup = ksg_mi(target, xembM, k = nnei)
        # I_sup1 = compute_mi(target, xembM, n_neighbors = nnei)

    else :
        I_sup = ksg_cmi(w_X, target, w_cond, k=nnei)
        # I_sup1 = compute_cmi(w_X, target, w_cond, n_neighbors=nnei)
    I_down = ksg_mi(target,xembM, k = nnei)
    # I_down1 = compute_mi(target,xembM, n_neighbors = nnei)
    
    R = I_sup/I_down
    return R



# =============================================================================
#  PMIME
# =============================================================================


def PMIME_measure(data, idX, idY, Z,verbose = 1,bootstrap = False,window = False,**kwargs):
    '''
    PMIME is a measure of direct relation from X to Y, conditionnal on a set Z
    of variables. 
    PMIME comes from :
    Direct coupling information measure from non-uniform embedding, D. Kugiumtzis
    
    inputs :
        - data : a dataframe of size (length of time series x number of variables)
        - idX, idY : name of the variables X and Y in data.columns
        - Z : names of the variables of the conditioning set in data
        - bootstrap: bool, if True, then the threshold for the stopping criterion of the 
        embedding vector process is not fixed
        
        - sig_pmime < 1: the stopping threshold for the building of the embedding vector 

        - sig_CMI: significance level for the bootstrap test
        - nb_bs : numbers of bootstrap iteration for CMI significance test
        - Lmax : the maximum lag to consider
        - nnei : numbers of neighbors for the estmation of MI/CMI
        - window : bool, if window = True, then the PMIME also returns the matrix idxM
            containing the variables and lags involved in the embedding vector
    '''
    
    T = 1
    if 'Lmax' in kwargs :
        Lmax = kwargs['Lmax']
    else : Lmax = 5
    if 'sig_pmime' in kwargs:
        A = kwargs['sig_pmime']
    else : A = 0.03
    if 'sig_CMI' in kwargs:
        sig_CMI = kwargs['sig_CMI']
    else : sig_CMI = 0.05
    if 'nb_bs' in kwargs :
        nb_bs = kwargs['nb_bs']
    else: nb_bs = 100
    
    if 'nnei' in kwargs :
        nnei = kwargs['nnei']

    else : nnei = 5
    
    
    ''' #### preprocessing #### '''
    varnames = [idX,idY]
    if len(Z)>1:
        for idz in Z:
            varnames.append(idz)
    elif len(Z)==1 :
        varnames.append(Z[0])
    
    ## allM : matrix containing X, Y and Z columns in data
    # allM[:,0] = X, allM[:,1] = Y , allM[:,2:] = Z_0,...
    allM = data[varnames].values
    
    
    #  standardisation of each columns of allM into [0,1]

    allM = normalize(allM)
    
    target = allM[:,1]

    ##building the lagged matrix

    all_lagM, indlagM = build_lagged_matrix(allM[:-1,:], Lmax, T)
    
    N_lag, alllags = all_lagM.shape     ##alllags = all the laggued variables 
    
    ## target = the future response vector : (Y_t+1)
    target = target[Lmax:-1] 
    
    ''' ### Build the embedding vector ###   '''

    iembV = []
    ## First embedding cycle:
    selected_index = first_cycle(all_lagM, target, Lmax, nnei)
    ## add the index in the list of the selected variables
    iembV += [selected_index]
    ## add the column of the selected variable in the embedding matrix
    xembM = all_lagM[:,iembV]
    
    terminaison = 0
    SC_l = []
    while terminaison ==0 :
        
        
        candidate_list  = [i for i in ([j for j in range(alllags)] + iembV) if i not in iembV] ## indices of candidates
        
        cmiV = -999*np.ones((alllags,))
        
        
        for var in candidate_list :
            ## we compute here CMI and MI for each candidate
            
            cmiV[var] = ksg_cmi(target,all_lagM[:,var], xembM,k=nnei)

            if cmiV[var] == np.max(cmiV):
                iemb_temp = var

        xtemp = np.column_stack((xembM,all_lagM[:,iemb_temp]))
        
        if bootstrap:
            p_value = bootstrap_sig_CMI(xtemp, target, nb_bs, cmiV[iemb_temp], nnei)

            if p_value < sig_CMI:
                iembV += [iemb_temp]
                xembM = xtemp
                
            else : terminaison = 1
        
        else:
            SC = cmiV[iemb_temp]/ksg_mi(target,xtemp,nnei)

            SC_l +=[SC]
            
            ## compare the ratio with the threshold
            if len(iembV) <= 2:
                # 2nd and 3rd embedding cycle to be tested
                if SC > A :
                    iembV += [iemb_temp]
                    xembM = xtemp
                else : terminaison = 1
            else  :
                # 4th embedding cycle to be tested --> additional condition to avoid 
                # false acceptation, if the ratio keeps increasing
                if SC > A and (SC<SC_l[-2]) and (SC < SC_l[-3]):
                    iembV += [iemb_temp]
                    xembM = xtemp
                else : terminaison = 1
    # idxM : matrix of the true variable index in allM + lag of each var
    #       implicated in the embedding vector
    idxM = np.nan*np.ones((len(iembV),2));
    for j in range(len(iembV)):
        idxM[j,0] = np.int8(np.ceil((iembV[j]+1)/Lmax)-1) ##  var indices
        idxM[j,1] = np.int8(np.mod(iembV[j],Lmax) +1)  #lag indices for each variables
    ## display infos :
    if verbose :
        print('\nComponents of the embedding vector :' )
        varname = []
        for j in range(len(iembV)):
            if idxM[j,0] ==0 :
                varname+=[idX]
            elif idxM[j,0] == 1 :
                varname+=[idY]
            else :
                varname+=[Z[int(idxM[j,0]-2)]]
        for lag in range(len(iembV)):
            print(varname[lag],'with lag : ', idxM[lag,1])
    
    ''' COMPUTE THE R MEASURE : '''
    R = compute_R(idxM, xembM, target, nnei)
    if window == True:
        return R, idxM,allM, all_lagM,target
    return R







if __name__ == '__main__':
    import timeit
    # n = 2000
    n = 2000
    X=1/10*np.ones((n,4))
    X[1,1] = np.random.randn()
    for j in range(1,n):
        X[j,0] = np.random.random()*X[j-1,0] + 0.01*np.random.randn()
   
    for k in range(1,n):
        X[k,2] = np.random.uniform()*X[k-1,0] + np.random.uniform() *X[k-1,2] + 0.01*np.random.randn()
    for k in range(1,n):
        X[k,3] = np.random.uniform()*X[k-1,3] + 0.01*np.random.randn()
    for t in range(2,n):
        X[t,1] = np.random.uniform()*np.tanh(X[t-3,0])**2  + np.random.uniform()*np.log(X[t-2,2]**2) +  np.random.uniform()*X[t-1,1] + 0.01*np.random.randn()
        #X[t,1] =  np.random.uniform()*X[t-2,2]**2 +  np.random.uniform()*X[t-1,1] + np.random.randn()
    df = pd.DataFrame(data = X, columns = ['X_0','X_1','X_2','X_3'])

    
    # all_lagM, indlagM = build_lagged_matrix(df.values[:-1,:], 5, 1)
    
    #print('PMIME :')
    #r1_ksg = PMIME(df,'X_0','X_1', Z = [], sig_pmime = 0.05)
    #print('measure :',r1_ksg)
    # print('\n PMIME bootstrap :')
    # for j in range(10):
    k = np.int64(0.01*n)    
    # start1 = timeit.default_timer()
    # rbs= PMIME(df,'X_2','X_1', Z= ['X_0'],bootstrap = True, A = 0.03, nnei = k)
    # stop1 = timeit.default_timer()
    start2 = timeit.default_timer()
    r= PMIME(df,'X_2','X_1', Z= ['X_0'],bootstrap = False, A = 0.01, nnei = k)
    stop2 = timeit.default_timer()
    # print('bootstrapped R:',rbs, 'time:',stop1-start1)
    print('not bootstrapped R:',r,'time:',stop2-start2)
    # print('\n PMIME bootstrap, x_solo :')
    # r2_ksg= PMIMEbs(df,'X_0','X_1', Z= [],sig_CMI = 0.05,bs_all = False)
    # print('with ksg : measure r1:', r1_ksg,'measure r :', r_ksg,'measure r_bs1 :', r2_ksg,'\n' )





