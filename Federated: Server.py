


# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:45:06 2021

@author: ali.raza
"""


from sklearn.metrics import classification_report
import socket
import threading
import time
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
import pygad
import pygad.nn
import pygad.gann
import numpy
from sklearn.decomposition import KernelPCA
from sklearn.neighbors._kde import KernelDensity
from sklearn.model_selection import GridSearchCV
import pygad
import pygad.nn


import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

from cvxopt import matrix, solvers
import sklearn.metrics.pairwise as smp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.metrics import auc

#-----------------------------------------Datan Preparation--------------------------------------------------------------------------
def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    #return MAE
    return np.mean(abs(v1 - v2), axis=1) 
class Visualization():
    
    def testResult(svdd, distance):
    
        """ 
        DESCRIPTION
        
        Plot the test results
        
        testResult(model, distance)
        
        --------------------------------------------------------------- 
        
        INPUT
        svdd             SVDD hypersphere
        distance         distance from test data to SVDD hypersphere 
        
        --------------------------------------------------------------- 
        
        """
        plt.rcParams['font.size'] = 15
        n = distance.shape[0]
        
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_subplot(1, 1, 1)
        radius = np.ones((n, 1))*svdd.model["radius"]
        ax.plot(radius, 
                color ='r',
                linestyle = '-', 
                marker = 'None',
                linewidth = 2, 
                markeredgecolor ='k',
                markerfacecolor = 'w', 
                markersize = 6)
        
        ax.plot(distance,
                color = 'k',
                linestyle = ':',
                marker='o',
                linewidth=1,
                markeredgecolor = 'k',
                markerfacecolor = 'C4',
                markersize = 6)
        
        ax.set_xlabel('Samples')
        ax.set_ylabel('Distance')
        
        ax.legend(["Radius","Distance"], 
                  ncol = 1, loc = 0, 
                  edgecolor = 'black', 
                  markerscale = 2, fancybox = True)
    
        plt.show() 
        
    def testROC(label, distance):
        """ 
        DESCRIPTION
        
        Plot the test ROC
        
        testROC(label, distance)
        
        --------------------------------------------------------------- 
        
        INPUT
        label            test label
        distance         distance from test data to SVDD hypersphere 
        
        --------------------------------------------------------------- 
        
        """
        if np.abs(np.sum(label)) == label.shape[0]:
            raise SyntaxError('Both positive and negative labels must be entered for plotting ROC curve.')
       
        # number of positive samples
        plt.rcParams['font.size'] = 15
        n_p = np.sum(label == 1)
        n_n = np.sum(label == -1)
        
        #  sort
        index = np.argsort(distance)
        label_sort = label[index]
        FP = np.cumsum(label_sort == -1)
        TP = np.cumsum(label_sort == 1)
        FPR = FP/n_n
        TPR = TP/n_p
        
        roc_auc = auc(FPR.T, TPR.T) 
                  
        fig = plt.figure(figsize = (6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(FPR.T, TPR.T,
                color ='C3',
                linestyle = '-', 
                marker = 'None',
                linewidth = 5, 
                markeredgecolor ='k',
                markerfacecolor = 'w', 
                markersize = 6)
        
        ax.set_xlabel('False positive rate (FPR)')
        ax.set_ylabel('True positive rate (TPR)')
        ax.set_title('Area under the curve (AUC) = %.4f' %roc_auc)
        
        plt.grid()
        plt.show()
        
        
    def boundary(svdd, data, label, r=0.3, nn=2):
        """ 
        DESCRIPTION
        
        Plot the boundary
        
        boundary(svdd, data, label, r=0.3, nn=2)
        
        --------------------------------------------------------------- 
        
        INPUT
        svdd             SVDD hypersphere
        data             training data 
        label            training label
        r                radio of expansion (0<r<1)
        nn               number of grids
        
        --------------------------------------------------------------- 
        
        """ 
        
        dim = data.shape[1]
        if dim!=2:
            raise SyntaxError('Visualization of decision boundary only supports for 2D data')
    
        # compute the range of grid 
        numGrids = np.rint(data.shape[0]/nn).astype(int)  # number of grids
        x_range = np.zeros(shape=(numGrids, 2))
        for i in range(2):  
            _tmp_ = (np.max(data[:, i])-np.min(data[:, i]))*r
            xlim_1 = np.min(data[:, i])-_tmp_
            xlim_2 = np.max(data[:, i])+_tmp_
            x_range[:, i] = np.linspace(xlim_1, xlim_2, numGrids)
        
        # grid
        xv, yv = np.meshgrid(x_range[:, 0], x_range[:, 1])
        
        num1 = xv.shape[0]
        num2 = yv.shape[0]
        distance = np.zeros(shape=(num1, num1))
        
        # calculate the grid scores
        print("Calculating the grid (%04d*%04d) scores...\n" %(num1, num2))
        
        display_ = svdd.parameters["option"]["display"]
        svdd.parameters["option"]["display"] = 'off'
        start_time = time.time()       
        for i in range(num1):
            for j in range(num2):
                tmp = np.mat([xv[i, j], yv[i, j]])   
                distance[i, j], _ = svdd.test(tmp, 1)
                # print('[feature 1: %06d]  [feature 2: %06d] \n' % (i+1,j+1))
        end_time = time.time()
        print('Grid scores completed. Time cost %.4f s\n' % (end_time-start_time))
        svdd.parameters["option"]["display"] = display_
        
        # plot the contour (3D)
        fig = plt.figure(figsize = (20, 6))
        
        ax3 = fig.add_subplot(1, 3, 1, projection='3d') 
        # ax3 = ax3.axes(projection='3d')
        ada = ax3.plot_surface(xv, yv, distance, cmap=plt.cm.jet)
        ax3.contourf(xv, yv, distance, zdir='z', offset=np.min(distance)*0.9, cmap=plt.cm.coolwarm)
        ax3.set_zlim(np.min(distance)*0.9, np.max(distance)*1.05)
        # plt.colorbar(ada)
            
    
    
        # plot the contour (2D)
        # fig = plt.figure(figsize = (10, 8))
        ax1 = fig.add_subplot(1, 3, 2)    
          
        ctf1 = ax1.contourf(xv, yv, distance, alpha = 0.8, cmap=plt.cm.jet)
        ctf2 = ax1.contour(xv, yv, distance, colors='black', linewidths=1)
        plt.clabel(ctf2, inline=True)
        # plt.colorbar(ctf1)
        
        # plot the boundary
        # fig = plt.figure(figsize = (10, 8))
        ax2 = fig.add_subplot(1, 3, 3)    
        
        if svdd.labeltype == 'single':

            ax2.scatter(data[:,0], data[:,1],
                        color='green',marker='o',
                        edgecolor='black',alpha=0.5, zorder = 2)
            ax2.scatter(data[svdd.model["sv_index"],0], data[svdd.model["sv_index"],1],
                    facecolor='C2',marker='o',s = 144,linewidths = 2,
                    edgecolor='black', zorder = 2)
        
            ax2.contour(xv, yv, distance,[svdd.model["radius"]],
                              colors='C3', linewidths=5, zorder = 1)
            
            ax2.legend(["Training data", "Support vectors"], 
                      ncol = 1, loc = 0, 
                      edgecolor = 'black',markerscale = 1.2, fancybox = True) 
                
        else:
            ax2.scatter(data[svdd.model["pIndex"],0], data[svdd.model["pIndex"],1],
                    facecolor='C0',marker='o', s = 100,linewidths = 2,
                    edgecolor='black', zorder = 2)
            ax2.scatter(data[svdd.model["nIndex"],0], data[svdd.model["nIndex"],1],
                    facecolor='C4',marker='s',s = 100,linewidths = 2,
                    edgecolor='black', zorder = 2)
        
            ax2.scatter(data[svdd.model["sv_index"],0], data[svdd.model["sv_index"],1],
                    facecolor='C2',marker='o',s = 144,linewidths = 2,
                    edgecolor='black', zorder = 2)
            
            ax2.contour(xv, yv, distance,[svdd.model["radius"]],
                              colors='C3', linewidths=5, zorder = 1)
            
            ax2.legend(["Training data (+)","Training data (-)", "Support vectors"], 
                      ncol = 1, loc = 0, 
                      edgecolor = 'black',markerscale = 1.2, fancybox = True) 
        
        plt.show()
class SVDD():
    
    def __init__(self, parameters):
        
        """ 
        DESCRIPTION
        
        --------------------------------------------------        
        INPUT
          parameters   
             "positive penalty": positive penalty factor
             "negative penalty": negative penalty factor
             "kernel"          : kernel function         
             "option"          : some options 
             
        
        """                
        self.parameters = parameters



    def train(self, data, label):
        
        """ 
        DESCRIPTION
        
        Train SVDD model
        
        -------------------------------------------------- 
        Reference
        Tax, David MJ, and Robert PW Duin.
        "Support vector data description." 
        Machine learning 54.1 (2004): 45-66.
        
        -------------------------------------------------- 
        model = train(data, label)
        
        --------------------------------------------------        
        INPUT
        data        Training data (n*d) 
                        n: number of samples
                        d: number of features
        label       Training label (n*1)
                        positive: 1
                        negative: -1
                        
        OUTPUT
        model       SVDD hypersphere
        --------------------------------------------------
        
        """
        start_time = time.time()
        
        label = np.array(label, dtype = 'int')      
        if np.abs(np.sum(label)) == data.shape[0]:
            self.labeltype = 'single'
        else:
            self.labeltype = 'hybrid'
        
        # index of positive and negative samples
        pIndex = label[:,0] == 1
        nIndex = label[:,0] == -1
        
        # threshold for support vectors
        threshold = 1e-7
        
        # compute the kernel matrix
        K = self.getMatrix(data, data)

        # solve the Lagrange dual problem
        alf, obj, iteration = self.quadprog(K, label)
        
        # find the index of support vectors
        sv_index = np.where(alf > threshold)[0][:]

        # support vectors and alf
        sv_value = data[sv_index, :]
        sv_alf = alf[sv_index]
        
        # compute the center of initial feature space
        center = np.dot(alf.T, data)
        
        ''''
        compute the radius: eq(15)
        
        The distance from any support vector to the center of 
        the sphere is the hypersphere radius. 
        Here take the 1st support vector as an example.
        
        '''
        # the 1st term in eq(15)
        used = 0
        term_1 = K[sv_index[used], sv_index[used]]
        
        # the 2nd term in eq(15)
        term_2 = -2*np.dot(K[sv_index[used], :], alf)
        
        # the 3rd term in eq(15)
        term_3 = np.dot(np.dot(alf.T, K), alf)

        # R
        radius = np.sqrt(term_1+term_2+term_3)
        
        end_time = time.time()
        timecost = end_time - start_time
        
        # numbers of positive and negative samples
        pData = np.sum(pIndex)/data.shape[0]
        nData = np.sum(nIndex)/data.shape[0]
        
        # number of support vectors
        nSVs = sv_index.shape[0]
        
        # radio of  support vectors
        rSVs = nSVs/data.shape[0]
        
        # store the results
        self.model = {"data"      : data        ,
                      "sv_alf"    : sv_alf      ,
                      "radius"    : radius      ,
                      "sv_value"  : sv_value    ,
                      "sv_index"  : sv_index    ,
                      "nSVs"      : nSVs        ,
                      "center"    : center      ,
                      "term_3"    : term_3      ,
                      "alf"       : alf         ,  
                      "K"         : K           ,
                      "pIndex"    : pIndex      ,
                      "nIndex"    : nIndex      ,
                      "obj"       : obj         ,
                      "iteration" : iteration   ,
                      "timecost"  : timecost    ,
                      "pData"     : pData       ,
                      "nData"     : nData       ,
                      "rSVs"      : rSVs        ,
                      }
        
        # compute the training accuracy
        display_ = self.parameters["option"]["display"]
        self.parameters["option"]["display"] = 'off'
        _, accuracy = self.test(data, label)
        self.parameters["option"]["display"] = display_      
        self.model["accuracy"] = accuracy
        
        # display training results       
        if self.parameters["option"]["display"] == 'on':
            print('\n')
            print('*** SVDD model training finished ***\n')
            print('iter             = %d'       % self.model["iteration"])
            print('time cost        = %.4f s'   % self.model["timecost"])
            print('obj              = %.4f'     % self.model["obj"])
            print('pData            = %.4f %%'  % (100*self.model["pData"]))
            print('nData            = %.4f %%'  % (100*self.model["nData"]))
            print('nSVs             = %d'       % self.model["nSVs"])
            print('radio of nSVs    = %.4f %%'  % (100*self.model["rSVs"]))
            print('accuracy         = %.4f %%'  % (100*self.model["accuracy"]))
            print('\n')
  
    def test(self, data, label):
    
        """ 
        DESCRIPTION
        
        Test the testing data using the SVDD model
    
        distance = test(model, Y)
        
        --------------------------------------------------        
        INPUT
        data        Test data (n*d) 
                        n: number of samples
                        d: number of features
        label       Test label (n*1)
                        positive: 1
                        negative: -1
            
        OUTPUT
        distance    Distance between the test data and hypersphere
        --------------------------------------------------
        
        """    
        
        start_time = time.time()
        n = data.shape[0]
        
        # compute the kernel matrix
        K = self.getMatrix(data, self.model["data"])
        
        # the 1st term
        term_1 = self.getMatrix(data, data)
        
        # the 2nd term
        tmp_1 = -2*np.dot(K, self.model["alf"])
        term_2 = np.tile(tmp_1, (1, n))
        
        # the 3rd term
        term_3 =  self.model["term_3"]
        
        # distance
        distance = np.sqrt(np.diagonal(term_1+term_2+term_3))
        
        # predicted label
        predictedlabel = np.mat(np.ones(n)).T
        fault_index = np.where(distance > self.model["radius"])[1][:]
        predictedlabel[fault_index] = -1
            
        # compute prediction accuracy
        accuracy = np.sum(predictedlabel == label)/n
        
        end_time = time.time()
        timecost = end_time - start_time
        if self.parameters["option"]["display"] == 'on':
        # display test results
            print('\n')
            print('*** SVDD model test finished ***\n')
            print('time cost        = %.4f s'   % timecost)
            print('accuracy         = %.4f %%'  % (100*accuracy))
            print('\n')
        
        
        return distance, accuracy 

    def quadprog(self, K, label):
    
        """ 
        DESCRIPTION
        
        Solve the Lagrange dual problem
        
        quadprog(self, K, label)
        
        --------------------------------------------------
        INPUT
        K         Kernel matrix
        label     training label
        
                        
        OUTPUT
        alf       Lagrange multipliers
        
        --------------------------------------------------
        
        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b                    
        --------------------------------------------------
        
        """ 
        solvers.options['show_progress'] = False
        
        label = np.mat(label)
        K = np.multiply(label*label.T, K)
        
        # P
        n = K.shape[0]
        P = K+K.T
        
        # q
        q = -np.multiply(label, np.mat(np.diagonal(K)).T)

        # G
        G1 = -np.eye(n)
        G2 = np.eye(n)
        G = np.append(G1, G2, axis=0)
        
        # h
        h1 = np.mat(np.zeros(n)).T # lb
        h2 = np.mat(np.ones(n)).T
        if self.labeltype == 'single':
            h2[label == 1] = self.parameters["positive penalty"]
        
        if self.labeltype == 'hybrid':
            h2[label == 1] = self.parameters["positive penalty"]
            h2[label == -1] = self.parameters["negative penalty"]

            
        h = np.append(h1, h2, axis=0)
        
        # A, b
        A = np.mat(np.ones(n))
        b = 1.
        
        #
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        
        #
        sol =solvers.qp(P, q, G, h, A, b)
        alf = np.array(sol['x'])
        obj = np.array(sol['dual objective'])
        iteration = np.array(sol['iterations'])

        return alf, obj, iteration

    def getMatrix(self, X, Y):
    
        """ 
        DESCRIPTION
        
        Compute kernel matrix 
        
        K = getMatrix(X, Y)
        
        -------------------------------------------------- 
        INPUT
        X         data (n*d)
        Y         data (m*d)
        OUTPUT
        K         kernel matrix 
        -------------------------------------------------- 
                        
                            
        type   -  
        
        linear :  k(x,y) = x'*y+c
        poly   :  k(x,y) = (x'*y+c)^d
        gauss  :  k(x,y) = exp(-s*||x-y||^2)
        tanh   :  k(x,y) = tanh(g*x'*y+c)
        lapl   :  k(x,y) = exp(-s*||x-y||)
           
        degree -  d
        offset -  c
        width  -  s
        gamma  -  g
        
        --------------------------------------------------      
        ker    - 
        
        ker = {"type": 'gauss', "width": s}
        ker = {"type": 'linear', "offset": c}
        ker = {"type": 'ploy', "degree": d, "offset": c}
        ker = {"type": 'tanh', "gamma": g, "offset": c}
        ker = {"type": 'lapl', "width": s}
    
        """
        def gaussFunc():
            
            if self.parameters["kernel"].__contains__("width"):
                s =  self.parameters["kernel"]["width"]
            else:
                s = 2
            K = smp.rbf_kernel(X, Y, gamma=s)

                
            return K
            
        def linearFunc():
            
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 0

            K = smp.linear_kernel(X, Y)+c
            
            return K
        
        def ployFunc():
            if self.parameters["kernel"].__contains__("degree"):
                d =  self.parameters["kernel"]["degree"]
            else:
                d = 2
                
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 0
                
            K = smp.polynomial_kernel(X, Y, degree=d, gamma=None, coef0=c)
            
            return K
             
        def laplFunc():
            
            if self.parameters["kernel"].__contains__("width"):
                s =  self.parameters["kernel"]["width"]
            else:
                s = 2
            K = smp.laplacian_kernel(X, Y, gamma=s)

            return K
    
        def tanhFunc():
            if self.parameters["kernel"].__contains__("gamma"):
                g =  self.parameters["kernel"]["gamma"]
            else:
                g = 0.01
                
            if self.parameters["kernel"].__contains__("offset"):
                c =  self.parameters["kernel"]["offset"]
            else:
                c = 1
            
            K = smp.sigmoid_kernel(X, Y, gamma=g, coef0=c)

            return K

        kernelType = self.parameters["kernel"]["type"]
        switcher = {    
                        "gauss"   : gaussFunc  ,        
                        "linear"  : linearFunc ,
                        "ploy"    : ployFunc   ,
                        "lapl"    : laplFunc   ,
                        "tanh"    : tanhFunc   ,
                     }
        
        return switcher[kernelType]()
df=pd.read_csv('ecg_final.txt',sep='  ',header=None)
df=df.add_prefix('c')
df['c0']=df['c0'].apply(lambda x: 1 if(x>1) else 0)

train_data,test_data,train_labels,test_labels=train_test_split(df.values,df.values[:,0:1],test_size=0.2, random_state=111)
scaler=MinMaxScaler()
data_scaled=scaler.fit(train_data)
train_data_scaled=data_scaled.transform(train_data)
test_data_scaled=data_scaled.transform(test_data)

normal_train_data=train_data_scaled[train_data_scaled[:,0]==0]
anomaly_train_data=train_data_scaled[train_data_scaled[:,0]==1]

normal_test_data=test_data_scaled[test_data_scaled[:,0]==0]
anomaly_test_data=test_data_scaled[test_data_scaled[:,0]==1]
normal_train_data=normal_train_data[:,1:]
anomaly_train_data=anomaly_train_data[:,1:]

normal_test_data=normal_test_data[:,1:]
anomaly_test_data=anomaly_test_data[:,1:]

train_data_scaled=train_data_scaled[:,1:]
test_data_scaled=test_data_scaled[:,1:]
normal_train_data= normal_train_data.reshape(len(normal_train_data), normal_train_data.shape[1],1)
anomaly_train_data= anomaly_train_data.reshape(len(anomaly_train_data), anomaly_train_data.shape[1],1)
train_data_scaled = train_data_scaled.reshape(len(train_data_scaled), train_data_scaled.shape[1],1)
test_data_scaled= test_data_scaled.reshape(len(test_data_scaled), test_data_scaled.shape[1],1)
normal_test_data= normal_test_data.reshape(len(normal_test_data), normal_test_data.shape[1],1)
anomaly_test_data= anomaly_test_data.reshape(len(anomaly_test_data), anomaly_test_data.shape[1],1)
test_data_scaled.shape
plt.plot(normal_test_data[0])
plt.plot(normal_test_data[1])
plt.plot(normal_test_data[2])
plt.plot(anomaly_test_data[0])
plt.plot(anomaly_test_data[1])
plt.plot(anomaly_test_data[2])

input_data = keras.Input(shape=(normal_train_data.shape[1]))

num_classes = 6
input_shape =input_data
learning_rate = 0.01
weight_decay = 0.001
batch_size = 56
num_epochs = 500
image_size = 90  # We'll resize input images to this size
patch_size = 188  # Size of the patches to be extract from the input images
num_patches = 1
projection_dim = 10
num_heads =2
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 2
mlp_head_units = [10]  # Size of the dense layers of the final classifier

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        #layers.experimental.preprocessing.Resizing(x_train.shape[1],1),
        #layers.experimental.preprocessing.RandomFlip("horizontal"),
        #layers.experimental.preprocessing.RandomRotation(factor=0.02),
        #layers.experimental.preprocessing.RandomZoom(
         #   height_factor=0.2, width_factor=0.2
        #),
    ],
    name="data_augmentation",
)



# Compute the mean and the variance of the training data for normalization.
#data_augmentation.layers[0].adapt(x_train)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim=6,name=None,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
    def build(self,input_shape):
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim )
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({"num_patches": self.num_patches,
                })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def sample(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
latent_dim = 10

encoder_inputs= layers.Input(shape=(normal_train_data.shape[1],1))
# Augment data.
augmented = data_augmentation(encoder_inputs)
# Create patches.
#patches = Patches(patch_size)(augmented)
# Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(augmented)

# Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])

# Create a [batch_size, projection_dim] tensor.
trans_shape = K.int_shape(encoded_patches) #Shape of conv to be provided to decoder
#Flatten
x = Flatten()(encoded_patches)
x = Dense(100, activation='relu')(x)

#z_mean = Dense(latent_dim, name='z_mean')(x)
#z_log_var = Dense(latent_dim, name='z_log_var')(x)
# use the reparameterization trick and get the output from the sample() function
#z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(encoder_inputs, x, name='encoder')
encoder.summary()
# decoder model
latent_inputs = Input(shape=(x.shape[1],), name='z_sampling')
x = Dense(trans_shape[1]*trans_shape[2], activation='relu')(latent_inputs)
x = Reshape((trans_shape[1], trans_shape[2]))(x)

# upscale (conv2D transpose) back to original shape
# use Conv2DTranspose to reverse the conv layers defined in the encoder
x = Dense(100,  activation='relu')(x)
#Can add more conv2DTranspose layers, if desired. 
#Using sigmoid activation
outputs = Dense(1,  activation='sigmoid', name='decoder_output')(x)
# Instantiate the decoder model:
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
outputs = decoder(encoder(encoder_inputs))
vae_model = Model(encoder_inputs, outputs, name='vae_mlp')

#def vae_loss(x, x_decoded_mean):
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
 #   reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
    # compute the KL loss
  #  kl_loss = - 0.6 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
    # return the average loss over all 
   # total_loss = K.mean(reconstruction_loss + kl_loss)    
    #total_loss = reconstruction_loss + kl_loss
    #return total_loss

opt = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
#opt = optimizers.RMSprop(learning_rate=0.0001)




vae_model.compile(optimizer=opt, loss='mse')
vae_model.summary()
tf.keras.models.save_model(vae_model, 'model.h5')
#-----------------------------------------Assign weights------------------------------------------------

from mlsocket import MLSocket
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
import numpy as np

ip = '172.23.32.182'
#ip='192.168.0.193'
port = 65432

number_clients=3

global count
count=0
global counter1
counter1=0
global counter2
counter2=0
global ave_weights
global ave_weights_auto
import socket 
from threading import Thread 
import threading 
from _thread import *
import time

s=MLSocket()
try:
    s.bind((ip, port))
except s.error as e:
    print(e)
print("Listening to clients")
s.listen()  
# initial weights of model make to zer0.  
ave_weights=vae_model.get_weights()
ave_weights=[i * 0 for i in ave_weights]

ave_weights_auto=vae_model.get_weights()
ave_weights_auto=[i * 0 for i in ave_weights_auto]

weight_scalling_factor=1


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Multithreaded Python server : TCP Server Socket Thread Pool
def ClientThread_send(conn,address,weights):
    global count
    global counter1
    global counter2
    if counter1<number_clients:
        conn.send(vae_model)
        counter1=counter1+1
        weight_scalling_factor=1
        print("Autoencoder Sent to :",address)
        print("waiting for weights")
        model_recv=conn.recv(1024)
        print("weights received from autoencoder:",address)
        global count
        global counter
        global ave_weights
        global ave_weights_auto
        scaled_local_weight_list = list()
        scaled_weights = scale_model_weights(model_recv.get_weights(), weight_scalling_factor)
        scaled_local_weight_list.append(scaled_weights)
        ave_weights_auto = sum_scaled_weights(scaled_local_weight_list)
        counter2=counter2+1
        print("counter 2 value:",counter2)
    
    if (counter2==number_clients):
        #while counter2<number_clients:
        time.sleep(3)
        print("Waiting for the classifier to complete the aggregation...")
            
        vae_model.set_weights(ave_weights_auto) 
       
        X_pred = vae_model.predict(normal_test_data)


        
        
        reconstructions=vae_model.predict(normal_test_data)
        
        mae_vector = get_error_term(reconstructions,normal_test_data, _rmse=False)
        train_loss=tf.keras.losses.mae(reconstructions, normal_test_data)
        plt.hist(mae_vector , bins=50)
        threshold=np.mean(mae_vector)+2.3*np.std(mae_vector)
        
        reconstructions_an=vae_model.predict(anomaly_test_data)
        mae_vector_an = get_error_term(reconstructions_an,anomaly_test_data, _rmse=False)
        plt.hist(mae_vector_an , bins=50)
        
        plt.hist(mae_vector_an , bins=50,label='abnormal')
        plt.hist(mae_vector , bins=50, label='normal')
        plt.axvline(threshold,color='r', linewidth=3, linestyle='dashed', label='{:0.3f}'.format(threshold))
        plt.show()
        
        
        reconstructions_asc=vae_model.predict(test_data_scaled)
        mae_vector_an = get_error_term(reconstructions_asc,test_data_scaled, _rmse=False)
        anomalies = (mae_vector_an > threshold)
        
        np.count_nonzero(anomalies) / len(anomalies)
        
        
        
        
        
        print(classification_report(test_labels, anomalies))
        X_pred = vae_model.predict(normal_test_data)
        
        reconstructions=vae_model.predict(normal_test_data)
        
        
        mae_vector = get_error_term(reconstructions,normal_test_data, _rmse=False)
        train_loss=tf.keras.losses.mae(reconstructions, normal_test_data)
        plt.hist(mae_vector , bins=50)
        threshold=np.mean(mae_vector)+.5*np.std(mae_vector)
        
        reconstructions_an=vae_model.predict(anomaly_test_data)
        mae_vector_an = get_error_term(reconstructions_an,anomaly_test_data, _rmse=False)
        plt.hist(mae_vector_an , bins=50)
        
        plt.hist(mae_vector_an , bins=50,label='abnormal')
        plt.hist(mae_vector , bins=50, label='normal')
        plt.axvline(threshold,color='r', linewidth=3, linestyle='dashed', label='{:0.3f}'.format(threshold))
        plt.legend(loc='upper right')
        plt.show()
        X_pred = vae_model.predict(normal_test_data)
        
        reconstructions=vae_model.predict(normal_test_data)
        
        
        mae_vector = get_error_term(reconstructions,normal_test_data, _rmse=False)
        train_loss=tf.keras.losses.mae(reconstructions, normal_test_data)
        plt.hist(mae_vector , bins=50)
        threshold=np.mean(mae_vector)+2.3*np.std(mae_vector)
        
        reconstructions_an=vae_model.predict(anomaly_test_data)
        mae_vector_an = get_error_term(reconstructions_an,anomaly_test_data, _rmse=False)
        plt.hist(mae_vector_an , bins=50)
        
        plt.hist(mae_vector_an , bins=50,label='abnormal')
        plt.hist(mae_vector , bins=50, label='normal')
        plt.axvline(threshold,color='r', linewidth=3, linestyle='dashed', label='{:0.3f}'.format(threshold))
        plt.show()
        
        
        reconstructions_asc=vae_model.predict(test_data_scaled)
        mae_vector_an = get_error_term(reconstructions_asc,test_data_scaled, _rmse=False)
        anomalies = (mae_vector_an > threshold)
        
        np.count_nonzero(anomalies) / len(anomalies)
        
        
        
        
        
        print(classification_report(test_labels, anomalies))
        train_lanel=np.where(train_labels==1, -1,train_labels)
        train_lanel2=np.where(train_lanel==0, 1,train_lanel)
        test_lanel=np.where(test_labels==1, -1,test_labels)
        test_lanel2=np.where(test_lanel==0, 1,test_lanel)
        
        predicted_train = vae_model.predict(train_data_scaled)
        predicted = vae_model.predict(test_data_scaled)
        
        
        
        
        
        train_lanel=np.where(train_labels==1, -1,train_labels)
        train_lanel2=np.where(train_lanel==0, 1,train_lanel)
        test_lanel=np.where(test_labels==1, -1,test_labels)
        test_lanel2=np.where(test_lanel==0, 1,test_lanel)
        
        predicted_train = vae_model.predict(train_data_scaled)
        predicted = vae_model.predict(test_data_scaled)
        
        params = {'bandwidth': np.linspace(3, 0.2, 10)}
        grid = GridSearchCV(KernelDensity(), params, cv = 30)
        
        e=(train_data_scaled- predicted_train)/np.max(train_data_scaled- predicted_train)
        nsamples, nx, ny = e.shape
        d2_e = e.reshape((nsamples,nx*ny))
        grid.fit(d2_e)
        h=grid.best_estimator_.bandwidth
        e1=(test_data_scaled - predicted)/np.max(test_data_scaled - predicted)
        nsamples1, nx1, ny1 = e1.shape
        d2_e1 = e1.reshape((nsamples1,nx1*ny1))
        
        kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1, fit_inverse_transform=True)
        X_train = kpca.fit_transform(d2_e)
        X_test=kpca.transform(d2_e)
        
        # kernel list
        kernelList = {#"1": {"type": 'gauss', "width": 0.05},
                      #"2": {"type": 'linear', "offset": 0},
                      #"3": {"type": 'ploy', "degree": 2, "offset": 0},
                      #"4": {"type": 'tanh', "gamma": 1e-4, "offset": 0},
                      "1": {"type": 'lapl', "width": h}
                      }
        
        
        for i in range(len(kernelList)):
        
            # set SVDD parameters
            parameters = {"positive penalty": 0.9,
                          "negative penalty": 0.7,
                          "kernel": kernelList.get(str(i+1)),
                          "option": {"display": 'on'}}
            
            # construct an SVDD model
            svdd = SVDD(parameters)
            
            # train SVDD model
            svdd.train(d2_e, train_lanel2)
              
            # test SVDD model
            distance, accuracy = svdd.test(d2_e1, test_lanel2)
            
            # visualize the results
            Visualization.testResult(svdd, distance)
            Visualization.testROC(test_lanel2, distance)
        
            conn.close()
            


    
while count<number_clients:
    conn, address = s.accept()
    start_new_thread(ClientThread_send,(conn,address,ave_weights))   
    print("count value", count)



   

    
