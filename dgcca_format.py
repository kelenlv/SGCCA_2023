import os

from dgcca_tool.dgcca import DGCCAArchitecture, LearningParams, DGCCA
from metric import *
import theano
import theano.tensor as T
import time
from plot_tsne import plot_tsne

class dgcca_(metric):
    def __init__(self, ds, m_rank, batchSize=40, epochs = 100):
        super().__init__()
        self.list_view = ds.train_data  # [np.float32(d) for d in ds.train_data]   # [(N, D), (N, D) ... ]
        self.ds = ds
        self.list_U = []  # save U for each view [(D, r), (D, r) ... ]
        self.list_projection = []  # save project data through U for each view [(N, r), (N, r) ... ]
        self.m_rank = m_rank

        self.model = None

        # parameter you can tune
        self.batchSize = batchSize
        self.epochs = epochs

    def solve(self):
        viewMlpStruct = [[v.shape[1], 10, 10, 10, v.shape[1]] for v in self.list_view]  # Each view has single-hidden-layer MLP with slightly wider hidden layers

        # Actual data used in paper plot...

        
        cnt = len(self.list_view)
        arch = DGCCAArchitecture(viewMlpStruct, self.m_rank, 2, activation=T.nnet.relu)

        #Little bit of L2 regularization -- learning params from matlab synthetic experiments
        lparams = LearningParams(rcov=[0.01] * cnt, viewWts=[1.0] * cnt, l1=[0.0] * cnt, l2=[5.e-4] * cnt,
                                 optStr='{"type":"adam","params":{"adam_b1":0.1,"adam_b2":0.001}}',
                                 batchSize=self.batchSize,
                                 epochs=self.epochs)
        # vnames = ['View1', 'View2', 'View3']
        vnames = None
        model = DGCCA(arch, lparams, vnames)
        model.build()

        history = []
        history.extend(model.learn(self.list_view, tuneViews=None, trainMissingData=None,
                                               tuneMissingData=None, embeddingPath=None,
                                               modelPath=None, logPath=None, calcGMinibatch=False))

        # self.train_G = model.apply(self.list_view, isTrain=True)
        self.model = model
        self.list_U = self.model.getU()
        np.save("W_list_dgcca.npy",self.list_U)
        self.list_projection = [self.list_view[i].transpose().T.dot(self.list_U[i]) for i in range(len(self.list_view))]
        print(np.abs(self.list_projection[1][:, 0]))
        plot_tsne(self.list_projection[0], np.abs(self.list_projection[1][:, 0]))

    def cal_G_error(self, list_view, test = True):
#        K = np.ones((list_view[0].shape[0], 3), dtype=np.float32)
        K = np.ones((list_view[0].shape[0], 3), dtype=np.float32)
        return self.model.reconstructionErr(list_view, missingData=K)

if __name__ == "__main__":
    data = data_generate()
    clf_ = dgcca_

    # # three views data for tfidf language data
    #
    # data.generate_three_view_tfidf_dataset()
#    data.generate_multi_view_tfidf_dataset()
    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']
    i = 0
    data.generate_genes_data(num=i,normalize=True)
    # data.generate_synthetic_dataset()
    clf = clf_(ds=data, m_rank=2, epochs = 200)#, batchSize=40, epochs = 200)
    
    s = time.time()
    clf.solve()
    e3 = time.time()
    print ("total time_3 is ", e3 - s)
#    v1_test, v2_test, v3_test, v4_test = clf.transform(data.test_data)
    v1_test, v2_test,v3_test = clf.transform(data.test_data)
#    print(len(clf.list_U)) #4
#    print(clf.list_U[0].shape,clf.list_U[1].shape,clf.list_U[2].shape,clf.list_U[3].shape)#(575, 8) (449, 8) (478, 8) (491, 8)

    #
    # # calculate all kind of metric
#    print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
#    print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
#    print("total correlation in training data is: ",np.sum(clf.cal_correlation(clf.list_projection)))
#    print("total correlation in testing data is: ",np.sum(clf.cal_correlation([v1_test, v2_test])))#data.test_data
#    print("training data Err is: ", clf.cal_G_error(data.train_data, test=False))#different cal_G_error
#    print("testing data Err is: ", clf.cal_G_error(data.test_data, test=True))
    print("a_training data ACC is: ",clf.cal_acc(clf.list_projection))
    print("a_testing data ACC is: ",clf.cal_acc([v1_test, v2_test,v3_test]))
#    print("b_training data ACC is: ",clf.cal_acc_lkx(clf.list_projection))
#    print("b_testing data ACC is: ",clf.cal_acc_lkx([v1_test, v2_test,v3_test]))
    print("c_training data ACC is: ",clf.cal_average_precision(clf.list_projection))
#    print("c_testing data ACC is: ",clf.cal_average_precision(data.test_data))
    print("c_testing data ACC is: ",clf.cal_average_precision([v1_test, v2_test,v3_test]))
  
#    print("each view's spare of U is ", clf.cal_spare())
#    print("total sqare is: ", np.mean(clf.cal_spare()))
    #
    # print()
    # print()
    
    # for synthetic data
#    data.generate_synthetic_dataset()
    
#    clf = clf_(ds=data, m_rank=2, batchSize=40, epochs = 10)
#    clf.solve()

    # print(clf.list_U)

    # calculate all kind of metric
    print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
#    print("total correlation in training data is: ",np.sum(clf.cal_correlation(clf.list_projection)))
#    print("total correlation in testing data is: ",np.sum(clf.cal_correlation(data.test_data)))
    print("each view's spare of U is ", clf.cal_spare())
    print("total sqare is: ", np.mean(clf.cal_spare()))

    print()
    print()

#    clf.save_U("deepgcca_synthetic")