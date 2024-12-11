# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:07:56 2020

@author: lvkexin
"""
import argparse
import numpy as np
import sklearn.datasets as ds
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import pandas as pd
import scipy.io as sco
import pickle
from data_class import data_generate
from metric import * 
from numpy import linalg as LA
import time
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from origin_gcca import gcca
from wgcca import WeightedGCCA 
from dgcca_format import dgcca_ 

import setproctitle
setproctitle.setproctitle("lvkexin:SGCCA")

from plot_tsne import plot_tsne

def generate_orthogonal_matrix(m, n):
    if m != 1 or n <= 1:
        raise ValueError("Invalid dimensions for generating an orthogonal matrix.")

    Q = np.eye(n)

    # 应用 Givens 变换生成正交矩阵
    for j in range(n - 1):
        G = np.eye(n)
        c, s = np.random.randn(2)
        G[j, j] = c
        G[j, j + 1] = -s
        G[j + 1, j] = s
        G[j + 1, j + 1] = c
        Q = np.dot(Q, G)

    Q_resized = np.expand_dims(Q[0, :], axis=0)  # 扩展维度为 (1, n)
    
    return Q_resized
def cal_AROC(list_projection):
    '''
    list_projection: [(N, D), (N, D) ... ]
    '''
    v=[]
    for i in range(len(list_projection)):
        v.append(list_projection[i])

    N = v[i].shape[0]     
    precision=[]
    for i in range(len(list_projection)):
        for j in range(i+1,len(list_projection)):
                    precision_ = 0
                    for ii in range(N):
                        temp = []
                        for jj in range(N):
                            dist = np.sum((v[i][ii] - v[j][jj]) ** 2)
                            temp.append((dist, jj))
                            temp = sorted(temp, key=lambda x: x[0], reverse=True)  # descent, least distance is the best
                            index = None
                            for it, t in enumerate(temp):
                                if t[1] == ii:
                                    index = it
                                    break
                        precision_ += float(index + 1) / N #same as cal_average_precision
                    precision_ =precision_/ N
                    precision.append(precision_)

    return precision
class gcca_admm(metric):
    def __init__(self, ds, m_rank=0, mu_x = None):
        '''
        Constructor for GeneralizedCCA.

        Args:
            list_view (list<ndarray>): Training data for each view
            m_rank (int): How many principal components to keep. A value of 0
                indicates that it should be full-rank. (Default 0)
        '''

        super().__init__()
        self.list_view = [dd.T for dd in ds.train_data]  # [(D, N), (D, N) ... ]
        self.testdata = ds.test_data
        self.ds = ds
        self.m_rank = m_rank  # top_r
        self.G = None  # subspace
        self.list_U = []  # save U for each view [(D, r), (D, r) ... ]
        self.list_projection = []  # save project data through U for each view [(N, r), (N, r) ... ]
        self.left_history = []
        self.right_history = []
        self.z_g_history = []
        self.z_history = []
        
        if mu_x == None:
            self.mu_x = [10 for i in range(len(self.list_view))]
        else:
            self.mu_x =  mu_x # [10 for i in
    def solve_g(self):
        '''
        Solves MAX-VAR GCCA optimization problem and returns the matrix G

        Returns:
            numpy.ndarray, the matrix 'G' that solves GCCA optimization problem
        '''

        reg = 0.00000001  # regularization parameter

        M = []  # matrix corresponding to M^tilde

        for i in range(len(self.list_view)):
            X = self.list_view[i].transpose()  # (N, D) (100, 17)

            # Perform rank-m SVD of X_j which yields X_j = A_j*S_j*B_j^T
            A, S, B = np.linalg.svd(X, full_matrices=False)
            # A:(N, m) (100, 17)
            # S:(17,)
            # B:(m, D) (17, 17)

            S = np.diag(S)

            N = np.shape(A)[0]
            m = np.shape(S)[0]

            # Compute and store A_J*T_J where T_j*T_j^T = S_j^T(r_jI+S_jS_j^T)^(-1)S_j
            # T = np.sqrt(np.mat(S.transpose()) * np.linalg.inv(reg * np.identity(m) + np.mat(S) * np.mat(S.transpose())) * np.mat(S))
            # (17, 17) diagonal matrix

            # Create an N by mJ matrix 'M^tilde' which is given by [A_1*T_1 ... A_J*T_J]
            if i == 0:
                M = np.array([], dtype=np.double).reshape(N, 0)
            
            # Append to existing M^tilde
            # M = np.hstack((M, np.mat(A) * np.mat(T)))  # (100, 54) (N, D1 + D2 + D3)
            M = np.hstack((M, np.mat(A)))

        # Perform SVD on M^tilde which yields G*S*V^T
        G, S, V = np.linalg.svd(M, full_matrices=False)

        if self.m_rank != 0:
            G = G[:, 0:self.m_rank]

        # Finally, return matrix G which has been computed from above
        G=np.array(G)
        return G
    def cal_A_B(self):
        '''
        Calculate common space of G and some necessary variable
        :param list_view: [view1, view2 ...] view shape:(D, N)
        :return: matrix G, list A , list B
        '''

        A = []
        B = []
        S = []

        for i, view in enumerate(self.list_view):
            # print('lkx:', view.shape)#feature \times sample
            ## padding
            if view.shape[0] < view.shape[1]: 
                p, s, q = np.linalg.svd(view, full_matrices=False)
                p = np.outer(p[:, 0], np.ones(view.shape[1]))# p_resized
                q = np.outer(np.ones(view.shape[1]), q[0, :])# q_resized
                s = s.repeat(view.shape[1])
                # s = np.zeros((view.shape[1], view.shape[1])) #S_resized 
                # s[:len(s), :len(s)] = np.diag(s)
                
            else:
                p, s, q = np.linalg.svd(view, full_matrices=False)
           
            # print('ok?', p.shape, s.shape, q.shape)
           
            A.append(p)
            B.append(q)
            S.append(s)

            # cal A and B
            n = S[i].shape[0]#1
            # print(n, np.arange(n))
            sigama = np.zeros((n, n))
            sigama[np.arange(n), np.arange(n)] = S[i]
            
            A[i] = A[i].T
            B[i] = -np.linalg.pinv(sigama).dot(B[i])
            
        return A, B
    def solve(self):
        # self.solve_u()
        self.admm()
        features = {'all_data': self.ds.train_data, 'list_projection':self.list_projection, 'W_list': self.list_U}
        convg = {'left_history':self.left_history, 'right_history':self.right_history,
                'z_g_history':self.z_g_history,'z_history':self.z_history}
        with open('convg.pkl', 'wb') as file:
            pickle.dump(convg, file)
        with open('features.pkl', 'wb') as file:
            pickle.dump(features, file)
    def admm(self):
        print()
        print("solve:admm")
        muta =0.1
        p=1.0001
        # 计算出G的数值方便后面的比较
        G_perfect = self.solve_g()
        aa=np.size(G_perfect,0)
        bb=np.size(G_perfect,1) 
        # initialize
        # muta = 0.01 #[0.1,1.5]  #1e-7 对应论文代码的delta
        beta_max = 1e4  # 最大beta值
        # p = 100 #(1,10]#10 #对应论文代码的rho
        tor1 = 1e-3
        tor2 = 1e-5
        res_tor=1e-5
        print('#delta:',muta)
        print("#rho:",p)
        iter = 400 #1680## # 迭代次数
        Z_new_old = 0  # 用来存放更新后的Z
        A, B = self.cal_A_B()
                
        num_view = len(A)  # 返回view的个数
        print("info: numview is ", num_view)
        
        # print("info: A's shape", A[0].shape, A[1].shape)#(49, 4026) (1, 1)  e.g., (50, 5500) (50, 10500) in synthetic data
        

        iter_k = 0  # 迭代步数初始化为0

        # 初始化 beta
        beta_list = []
        for i in range(num_view):
            #temp = 1 / np.max(np.max(np.abs(A[i].T.dot(B[i]))))  # 1 / LA.norm(A[i].T.dot(B[i]), ord=np.inf)  # beta初始化，无穷范数
            temp =  1 / LA.norm(A[i].T.dot(B[i]), ord=np.inf) 
            beta_list.append(temp)
        beta = np.max(beta_list)  # beta 是标量
        #beta=beta_max
        print("#beta_init:",beta)
        # 初始化 每个view的权重(W_list)和tri_list(倒三角)
        W_list = []
        tri_list = []
        tri_list_w = []
        for a in A: #views
            #W = np.ones(shape=(a.shape[1], self.m_rank))
            if  a.shape[1] == 1:
                if self.m_rank ==1:
                    # W = np.random.randn(1, 1)
                    W = 0.001*np.ones((1, 1))
                else:
                    W = generate_orthogonal_matrix(a.shape[1], self.m_rank)
                
            else:
                [W,temp] = LA.qr(np.random.randn(a.shape[1], self.m_rank))
            # print(W.shape)#2-view:(4026, 2) (1, 2)
            #W=np.random.randn(a.shape[1], self.m_rank)
            W_list.append(W)
            tri_list.append(0)
            tri_list_w.append(0)
        
        smax= 60
        train_AROC_list = []
        test_AROC_list = []
        while True:
            iter_k +=1  # 迭代次数加一                
            # update Z: initialize C and D
            for j in range(num_view):
                if j == 0: # initialization 
                    C = tri_list[j]/beta - A[j].dot(W_list[j])#(49, 2)
                    D = B[j]#(49, 49)
                else:
                    C = np.concatenate((C, tri_list[j]/beta - A[j].dot(W_list[j])), axis=0)
                    D = np.concatenate((D, B[j]), axis=0)
            # C: m \times l  D: m \times m  m: samples l:rank
            C_col=C.shape[1] #l
            C_row=C.shape[0]
            D_col=D.shape[1] #m >l
            gap_col=D_col-C_col
            # 方法一初始化
            # C_tilde=np.zeros((C_row,gap_col))
            #方法二初始化
            [e_vals,e_vecs]=LA.eig(D.T.dot(D))
            sorted_indices=np.argsort(e_vals) #ascend
            E=e_vecs[:,sorted_indices[:bb:1]]
            C_tilde=D.dot(E[:,0:(C_row-C_col)])
            
            CC=np.concatenate((C,C_tilde),axis=1)
            [U,S,Vt]=LA.svd(D.T.dot(CC),full_matrices=False)
            Gs=np.dot(U,Vt.T)
            Hs=Gs[:, C_col:D_col]
            Zs=Gs[:,0:C_col]
            res_old=LA.norm(D.dot(Zs)-C, ord="fro")
            #update Z          
            for i in range(smax): 
                [U,S,Vt]=LA.svd(D.T.dot(np.concatenate((C,D.dot(Hs)),axis=1)),full_matrices=False)                     
                Gs=np.dot(U,Vt.T)
                Hs=Gs[:, C_col:D_col]
                Zs=Gs[:,0:C_col]
                res_new=LA.norm(D.dot(Zs)-C, ord="fro")
                if (res_old-res_new)< res_tor*res_old:
                    break
                res_old=res_new  
            Z_new=Zs


           
            # 把旧的值存起来，作为下面的判断条件
            W_list_old= [w.copy() for w in W_list]
            tri_list_old = [t for t in tri_list]

            for i in range(num_view):
                # update W
                temp_x1 = A[i].dot(W_list[i]) + B[i].dot(Z_new)
                temp_x2 =  tri_list_w[i]/beta
                temp_x3 = muta*A[i].T.dot(temp_x1 - temp_x2)
                x = W_list[i] - temp_x3

                mu = muta/beta
                temp_W_new1 = np.sign(x)
                temp_W_new2 = np.abs(x) - mu
                W_list[i] = temp_W_new1 * np.maximum(temp_W_new2, 0)

                # update tri_list
                tri_list[i] = tri_list[i] - beta*(A[i].dot(W_list[i]) + B[i].dot(Z_new))#cal Z
                tri_list_w[i] = tri_list_w[i] - beta*(A[i].dot(W_list[i]) + B[i].dot(Z_new)) #real
                
                
            #print("#:",W_list[i].shape,A[i].shape)#: (491, 20) (575, 491)
            # 求平均全部加起来更新W_mean和tri_mean，这是K+1的更新
            
            W_mean = np.mean([A[i].dot(W_list[i]) for i in range(len(W_list))], axis=0)
            tri_mean = np.mean(tri_list, axis=0)

            # 判断条件
            left_list = [(beta * LA.norm(W_list[i] - W_list_old[i], ord="fro") /
                         max(1, LA.norm(W_list_old[i], ord="fro") ))
                         for i in range(num_view)]
            right_list = [(LA.norm(tri_list[i] -
                          tri_list_old[i], ord="fro") / beta)
                          for i in range(num_view)]
            # save the history
            self.left_history.append(left_list)
            self.right_history.append(right_list)
            self.z_g_history.append(LA.norm(Z_new - G_perfect, ord="fro"))
            self.z_history.append(LA.norm(Z_new - Z_new_old, ord="fro"))

            left_judge = [i < tor1 for i in left_list]
            right_judge = [i < tor2 for i in right_list]
            if ((False not in left_judge) and (False not in right_judge)) or iter_k == iter:  # 判断如果都小于容忍值或达到迭代次数
                break  # 结果已经更新到对应的list上面
            # update beta
            beta = min(beta_max, p * beta)
            # save old Z
            Z_new_old = Z_new
            
            # print("#iter::",iter_k)
            train_AROC_list.append(np.mean(cal_AROC([self.list_view[i].transpose().dot(W_list[i]) for i in range(num_view)])))
            test_AROC_list.append(np.mean(cal_AROC([self.testdata[i].dot(W_list[i]) for i in range(num_view)])))
            if ((False not in left_judge) and (False not in right_judge)) or iter_k == iter:  # 判断如果都小于容忍值或达到迭代次数
                break  # 结果已经更新到对应的list上面
        print()
        print("$$$$$$$$$$$$$$$$$$")
        AROC_dict = {'train_AROC_list':np.array(train_AROC_list), 'test_AROC_list':np.array(test_AROC_list)}
        with open('AROC_dict.pkl', 'wb') as file:
            pickle.dump(AROC_dict, file)
        # stack and show easily
        self.left_history = np.stack(self.left_history)
        self.right_history = np.stack(self.right_history)
        
        self.G = Z_new
        self.list_U = W_list
        self.list_projection = [self.list_view[i].transpose().dot(W_list[i]) for i in range(num_view)]
        plot_tsne(self.list_projection[0], np.abs(self.list_projection[1][:, 0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--method', default = 'gcca_admm', type=str, help='gcca_admm, gcca, wgcca, dgcca')
    parser.add_argument("--dataset", default = 'synthetic', type=str, help='synthetic, genedata, documentdata')
    parser.add_argument("--gene_cls", default = '1', type=int, help='[0,1,2,3,4,5]')
    
    args = parser.parse_args()
    print('programming start!')
    data = data_generate()


    # dataset
    if args.dataset == 'synthetic':# three-view
        l = 1
        data.generate_synthetic_dataset()
    elif args.dataset == 'genedata': # two-view: feature and label
        # genedata information:
        # gene_cls = [0,1,2,3,4,5]
        name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']
        target_l = [1, 1, 2, 1, 4, 1]
        
        if args.gene_cls in range(len(name)):  
            l = target_l[args.gene_cls]
            print('#rank:', l)    
        else:
            print('genedata error!')       
        data.generate_genes_data(num=args.gene_cls, normalize=True)
                    
    elif args.dataset == 'documentdata':
        l = 1
        data.generate_three_view_tfidf_dataset()
    else:
        print('Data load error!')
    

    ## cross validation
    train_err = []
    test_err = []
    train_cor = []
    test_cor = []
    train_AROC = []
    test_AROC =[]
    spare = []
    # for p in np.arange(5.00,5.21,0.1):
    # for muta in np.arange(0.05, 0.065, 0.001):
        # model->clf: class function
    if args.method == 'gcca_admm':
        clf = gcca_admm
    elif args.method == 'gcca':
        clf = gcca
    elif args.method == 'dgcca':
        clf = dgcca_
    elif args.method == 'wgcca':
        clf = WeightedGCCA
    model = clf(ds = data, m_rank = 2)#l
    # training
    s = time.time()
    model.solve()
    e1 = time.time()
    print('------------training in {} finishied-----------------'.format(args.dataset))
    print ("!! total running time of {} is {} ".format(args.method, e1 - s) )

    # calculate all kind of metric
    if args.dataset == 'synthetic':# three-view
        print('-----------------------------------------')
        # G_error/reconstruction error of G: error between optimized G and each optimized projection
        print("!! reconstruction error of G in training of %s is: %f"%(args.method,  model.cal_G_error(data.train_data, test=False)))
        print("!! reconstruction error of G in testing of %s is: %f"%(args.method,  model.cal_G_error(data.test_data, test=True)))
        train_err.append(model.cal_G_error(data.train_data, test=False))
        test_err.append(model.cal_G_error(data.test_data, test=True))
        print('-----------------------------------------')
        # sum of correlation: Pearson product-moment correlation coefficients, [-1,1], abs=1 is better
        print("!! total correlation in training data of %s is: %f "%(args.method, np.sum(model.cal_correlation(model.list_projection))))
        print("!! total correlation in testing data of %s is: %f "%(args.method, np.sum(model.cal_correlation(model.transform(data.test_data))))) # projection for test data
        train_cor.append(np.sum(model.cal_correlation(model.list_projection)))
        test_cor.append(np.sum(model.cal_correlation(model.transform(data.test_data))))
        print('-----------------------------------------')
        res = model.cal_spare() # save info
        print("!! each view's sparsity of {} is: {}".format(args.method, res))
        print("!! averaged sqarsity of %s is: %f"%(args.method, np.mean(res)))
        spare.append(np.mean(res))
    elif args.dataset == 'genedata': # two-view: feature and label
        print('-----------------------------------------')
        # G_error/reconstruction error of G:
        print("!! reconstruction error of G in training of %s is: %f"%(args.method,  model.cal_G_error(data.train_data, test=False)))
        print('-----------------------------------------')
        # sum of correlation: Pearson product-moment correlation coefficients, [-1,1], abs=1 is better
        print("!! total correlation in training data of %s is: %f "%(args.method, np.sum(model.cal_correlation(model.list_projection))))
        print("!! total correlation in testing data of %s is: %f "%(args.method, np.sum(model.cal_correlation(model.transform(data.test_data))))) # projection for test data
        print('-----------------------------------------')
        res = model.cal_spare() # save info
        print("!! each view's sparsity of {} is: {}".format(args.method, res))
        print("!! averaged sqarsity of %s is: %f"%(args.method, np.mean(res)))
        print('-----------------------------------------')
        print("!! classification accuracy in training data of %s is: %f "%(args.method, model.cal_acc(model.list_projection)))
        print("!! classification accuracy in testing data of %s is: %f "%(args.method, model.cal_acc(model.transform(data.test_data))))  
        # print("!! AROC in training data of {} is: {} ".format(args.method, model.cal_AROC(model.list_projection)))
        # print("!! AROC in testing data of {} is: {} ".format(args.method, model.cal_AROC(model.transform(data.test_data)))) 
        # print("!! retrieval precision in training data of %s is: %f "%(args.method, model.cal_average_precision(model.list_projection)))
        # print("!! retrieval precision in testing data of %s is: %f "%(args.method, model.cal_average_precision(model.transform(data.test_data))))  
    elif args.dataset == 'documentdata':
        print('-----------------------------------------')
        # G_error/reconstruction error of G:
        print("!! reconstruction error of G in training of %s is: %f"%(args.method,  model.cal_G_error(data.train_data, test=False)))
        print('-----------------------------------------')
        # sum of correlation: Pearson product-moment correlation coefficients, [-1,1], abs=1 is better
        print("!! total correlation in training data of %s is: %f "%(args.method, np.sum(model.cal_correlation(model.list_projection))))
        print("!! total correlation in testing data of %s is: %f "%(args.method, np.sum(model.cal_correlation(model.transform(data.test_data))))) # projection for test data
        train_cor.append(np.sum(model.cal_correlation(model.list_projection)))
        test_cor.append(np.sum(model.cal_correlation(model.transform(data.test_data))))
        print('-----------------------------------------')
        res = model.cal_spare() # save info
        print("!! each view's sparsity of {} is: {}".format(args.method, res))
        print("!! averaged sqarsity of %s is: %f"%(args.method, np.mean(res)))
        spare.append(np.mean(res))
        print('-----------------------------------------')
        print("!! retrieval precision/AROC in training data of {} is: {} ".format(args.method, model.cal_AROC(model.list_projection)))
        print(np.mean(model.cal_AROC(model.list_projection)))
        print("!! retrieval precision/AROC in testing data of {} is: {} ".format(args.method, model.cal_AROC(model.transform(data.test_data)))) 
        print(np.mean(model.cal_AROC(model.transform(data.test_data))))
        train_AROC.append(np.mean(model.cal_AROC(model.list_projection)))
        test_AROC.append(np.mean(model.cal_AROC(model.transform(data.test_data))))
    print('------------evaluation finishied-----------------')
    # break
    
   
  
    # check_list = {'train_AROC': train_AROC, 'test_AROC': test_AROC, 'train_cor': train_cor,'test_cor':test_cor, 'spare':spare}
    # # 保存数据到文件
    # with open('check_list.pkl', 'wb') as file:
    #     pickle.dump(check_list, file)


    # canonical variables
    # print("# W_list of %s: %r, and norm is %r"%(args.method, clf.list_U, np.linalg.norm(clf.list_U[0],ord=1)+np.linalg.norm(clf.list_U[1],ord=1)))
    # print("# object fuction of %s:"%(args.method, clf.cal_target()))


    # print("b_training data ACC is: ",clf.cal_acc_lkx(clf.list_projection), clf2.cal_acc_lkx(clf2.list_projection))
    # print("b_testing data ACC is: ",clf.cal_acc_lkx([v1_test, v2_test]), clf2.cal_acc_lkx([v1_test, v2_test]))
    # print("training data Err is: ", clf.cal_err_lkx(clf.list_projection),clf2.cal_err_lkx(clf2.list_projection))
    # print("testing data Err is: ", clf.cal_err_lkx([v1_test, v2_test,v3_test]),clf2.cal_err_lkx([v1_test, v2_test,v3_test]))
    # print("testing data G_error is: ", clf.cal_G_error([v1_1_test, v2_1_test, v3_1_test], test=True),clf2.cal_G_error([v1_2_test, v2_2_test, v3_2_test], test=True),clf3.cal_G_error([v1_3_test, v2_3_test, v3_3_test], test=True),clf4.cal_G_error([v1_4_test, v2_4_test, v3_4_test], test=True))
    # print("total spasity is: ", clf.cal_spare()[0],clf2.cal_spare()[0])
    
    ### Europarl data sets
   
    # data.generate_three_view_tfidf_dataset()
    # data.generate_mnist()
    # data.generate_multi_view_tfidf_dataset()

    # L=1

    # mu_x = (10, 10, 10)
    # clf = clf_(ds=data, m_rank=1, mu_x = mu_x)#admm_lkx_2
    # print("#W_list of admm-gcca:",clf.list_U,np.linalg.norm(clf.list_U[0],ord=1)+np.linalg.norm(clf.list_U[1],ord=1))

    # print("#gcca_admm_t",clf.cal_target())
    # print("##gcca_t",clf2.cal_target())

    # print("a_training data ACC is: ",clf.cal_acc(clf.list_projection), clf2.cal_acc(clf2.list_projection))
    # print("a_testing data ACC is: ",clf.cal_acc([v1_a_test, v2_a_test]), clf2.cal_acc([ v1_2_test, v2_2_test]))
    # print("b_training data ACC is: ",clf.cal_acc_lkx(clf.list_projection), clf2.cal_acc_lkx(clf2.list_projection))
    # print("b_testing data ACC is: ",clf.cal_acc_lkx([v1_a_test, v2_a_test]), clf2.cal_acc_lkx([ v1_2_test, v2_2_test]))

    # print("training data Err is: ", clf.cal_err_lkx(clf.list_projection),clf2.cal_err_lkx(clf2.list_projection))
    # print("testing data Err is: ", clf.cal_err_lkx([v1_a_test, v2_a_test]),clf2.cal_err_lkx([ v1_2_test, v2_2_test]))
    # print("training data averaged AROC  is :", np.mean(clf.cal_AROC(clf.list_projection)), np.mean(clf2.cal_AROC(clf2.list_projection)))
    # print("testing data averaged AROC  is :", np.mean(clf.cal_AROC([v1_a_test, v2_a_test])), np.mean(clf2.cal_AROC([v1_2_test, v2_2_test])))#, v3_a_test
    # print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    


    
    
