# coding: utf-8
'''
Created on 2018. 3. 27.

@author: Insup Jung
'''

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from BackPropa.TwoLayerNet import TwoLayerNet

if __name__ == '__main__':
    
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) 
    #normalize는 입력이미지의 픽셀값을 0.0~1.0 사이의 값으로 정규화할지를 정한다.
#     if normalize:
#         for key in ('train_img', 'test_img'):
#             dataset[key] = dataset[key].astype(np.float32)
#             dataset[key] /= 255.0
#    정답만 1이고 나머지는 0으로 만들어준다.
#     if one_hot_label:
#         dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
#         dataset['test_label'] = _change_one_hot_label(dataset['test_label'])   
#     def _change_one_hot_label(X):
#         T = np.zeros((X.size, 10))
#         for idx, row in enumerate(T):
#             row[X[idx]] = 1
#              
#         return T

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) #입력층, 은닉층, 출력층 개수
    
    iters_num = 10
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size) #train_size에서 batch_size만큼의 샘플을 뽑아낸다.ㅆ
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        print(batch_mask)
        
        grad = network.gradient(x_batch, t_batch)
        
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate*grad[key]
            
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)    
    
    pass