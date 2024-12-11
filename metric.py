import numpy as np
import sklearn.datasets as ds
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# from utils import *
import pandas as pd
import scipy.io as sco
import pickle
from data_class import *
#from wgcca import *

class metric:
    def __init__(self):
        self.list_projection = []
        self.list_U = []
        self.ds = None
        self.G = None
        self.p=None
        self.list_view = []
    def cal_err_lkx(self, list_projection, test=False):
        # v1 = list_projection[0]
        # v2 = list_projection[1]
        # N = v1.shape[0]
        err=0
        #print("#len:",len(list_projection[0]),len(list_projection[1]),len(self.G))
        if not test:
            
            for i in range(len(self.list_view)):
                err += np.sum(np.linalg.norm(list_projection[i]-self.G,axis=0))
             
        return err/len(list_projection[0]) 
    def cal_acc_lkx(self, list_projection):
        #print("~~@@@@",list_projection,list_projection[0][0],list_projection[1][0])
        acc=0
        e=0.01
        print("#len:",len(list_projection[0]),len(list_projection[1]))
        #y_pred = list_projection[0].dot(np.linalg.pinv(self.list_U[1]))
        for i in range(len(list_projection[0])):
            if (np.linalg.norm(list_projection[0][i]-list_projection[1][i])) < e:
                acc += 1     
        print("#acc:",acc)        
        #print("!~!~!~",y_pred)
        return acc/len(list_projection[0])

            
    def cal_correlation(self, list_projection):
        def rank_corr(corr_array):
            D = int(corr_array.shape[0] / 2)#1=int(3/2)
            res = []
            for i in range(D):
                if not np.isnan(corr_array[i][i + D]) :
                    res.append(abs(corr_array[i][i + D]))
            return res
        # print('debug:',np.concatenate(list_projection, axis=1).shape)
        corr_array = np.corrcoef(np.concatenate(list_projection, axis=1), rowvar=False)
        # print(corr_array.shape)#(8, 8)
        return abs(corr_array[np.triu_indices(corr_array.shape[0], k = 1)])#rank_corr(corr_array)

    def cal_r2_score(self):
        return r2_score(self.list_projection[0], self.list_projection[1]), r2_score(self.list_projection[1], self.list_projection[0])

    def cal_spare(self):
        res = []
        for i in range(len(self.list_U)):
            print("info of sparsity: L1 norm of each view:", np.linalg.norm(self.list_U[i],ord=1)) # L1 norm of each view's U
        for u in self.list_U:
            print("shape of list_U", u.shape[0],u.shape[1]) # U's shape: number \times feature
            res.append(np.sum(np.abs(u)<=1e-5) / (u.shape[0] * u.shape[1]))
            print("info of sparsity: zero number:", np.sum(np.abs(u)<=1e-5))   # sparsity: the number of zero in list of view U
        return res
    
    def cal_average_precision(self, list_projection):# 第几个it找到的
        '''
        list_projection: [(N, D), (N, D) ... ]
        '''
        #print("where?",list_projection[2].shape)
        v1 = list_projection[0]
        v2 = list_projection[1]
        
        N = v1.shape[0]
        # print("#v1.shape[0]",N,v1.shape[1])
        # print("#v2.shape[0]",v2.shape[0],v2.shape[1])
        #acc=0
        precision = 0
        for i in range(N):
            temp = []
            for j in range(N):
                dist = np.sum((v1[i] - v2[j]) ** 2)
                temp.append((dist, j))
            temp = sorted(temp, key=lambda x: x[0], reverse=True)  #descent, least distance is the best
            index = None
            for it, t in enumerate(temp):
                if t[1] == i: #if j == i, index
                    index = it
                    break
            #print ("#index:",index)
            precision += float(index + 1) / N
        precision =precision/ N
        return precision
    
    def cal_AROC(self, list_projection):
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

    def cal_acc(self, list_projection):
        v1 = list_projection[0]
        v2 = list_projection[1]
        

        

        # sco.savemat('list_pro.mat', {'v1': np.array(v1), 'v2': np.array(v2)})
        N = v1.shape[0]

        # some pre for y
        label = set()
        for arr in v2:
            label.add(tuple(arr))


        label = list(label)
        res = []
        for arr in v2:
            for i, t in enumerate(label):
                if tuple(arr) == t:
                    res.append(i)
                    break
        
        c = 0
        for i in range(N):
            temp = []
            for j in range(N):
                dist = np.sum((v1[i] - v2[j]) ** 2)
                temp.append((dist, j))
            temp = sorted(temp, key=lambda x: x[0], reverse = False)#ascent
            # print('debug:', v2)
            # print('temp:', temp, temp[0][1])
            for iz, z in enumerate(label):
                # print(iz,z)#0 (-0.10648042524107533,); 1 (0.19360077316559163,)
                tt = tuple(v2[temp[0][1]])# temp[0][1]: approximate of label; temp: {dist, label_found}
                if tt == z:
                    if iz == res[i]:
                        c += 1
        return float(c) / N

    def solve_g(self):
        pass
    ## objection function is G_error
    def cal_G_error(self, list_view, test = True):
        res = 0
        list_projection = self.transform(list_view)
        if test:
            #self.list_view = [dd.T for dd in list_view]
            self.solve_g()

        
        for v in list_projection[1:]:
            res += np.linalg.norm((self.G - v)**2)
            # res += np.sum(np.mean(np.abs(v - self.G), axis=0))
        # p = np.array(np.concatenate([v[None, :, :] for v in list_projection], axis=0))
        # print('debug1::',res)
        # res = 0
        # self.G = np.array(self.G)
        # self.p=p
        # # print('debug2::', p.shape, list_projection[0].shape, self.G.shape )#(3, 50, 1) (50, 1) (50, 1)
        # res = np.linalg.norm((self.G - p)**2)
        # print('debug2::',res)
        return res

    def transform(self,list_view,lable=None):
        '''
        :param v1: (N, D)
        :param v2:
        :return:
        '''
        res = []
        # print("sdsd:",list_view[0].shape,self.list_U[0].shape)
        if lable=='fista':
            for i in range(len(self.list_U)):
                res.append(list_view[i].dot(self.list_U[i].transpose()))
        else:
            for i in range(len(self.list_U)):
                res.append(list_view[i].dot(self.list_U[i])) 
        return res

    def predict(self, X):
        '''
        X: (N, D)
        '''
        X = X.copy()
        X -= self.ds.x_mean
        X /= self.ds.x_std
        X_proj = X.dot(self.list_U[0])
        y_pred = X_proj.dot(np.linalg.pinv(self.list_U[1])) * self.ds.y_std + self.ds.y_mean
        return y_pred

    def save_U(self, name):
        with open("../gcca_data/weight/" + name + ".pickle", "wb") as f:
            pickle.dump(self.list_U, f)

                
#if __name__ == '__main__':
#    data = data_generate()
#    clf_ = WeightedGCCA
#    data.generate_multi_view_tfidf_dataset()
#    clf = clf_(ds=data, m_rank=20)
#    v1_test, v2_test, v3_test, v4_test = clf.transform(data.test_data)