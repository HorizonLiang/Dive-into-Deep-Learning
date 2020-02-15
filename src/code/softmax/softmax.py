#!/usr/bin/env python
# coding: utf-8

# # softmax和分类模型

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython import display
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import time

import sys
data_path = os.path.abspath("../../") + "\\data"
sys.path.append(os.path.abspath("../../.."))
from src.code.utils import *

print(torch.__version__)
print(torchvision.__version__)


# ### 获取数据

# In[7]:


mnist_train = torchvision.datasets.FashionMNIST(root=data_path + "/FashionMNIST2065", train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=data_path + "/FashionMNIST2065", train=False, download=True, transform=transforms.ToTensor())


# In[ ]:


# 通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)


# In[ ]:


mnist_PIL = torchvision.datasets.FashionMNIST(root=data_path + "/FashionMNIST2065", train=True, download=True)
PIL_feature, label = mnist_PIL[0]
print(PIL_feature)


# In[8]:


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# In[9]:


def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# In[10]:


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0]) # 将第i个feature加到X中
    y.append(mnist_train[i][1]) # 将第i个label加到y中
show_fashion_mnist(X, get_fashion_mnist_labels(y))


# In[ ]:


batch_size = 256
num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# In[ ]:


start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))


# # softmax从零开始的实现

# In[2]:


import torch
import torchvision
import numpy as np
import sys
data_path = os.path.abspath("../../") + "\\data"
from src.code.utils import *

print(torch.__version__)
print(torchvision.__version__)


# In[ ]:


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, root=data_path + "/FashionMNIST2065")


# In[ ]:


num_inputs = 784
print(28*28)
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)


# In[ ]:


W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# ## 定义softmax操作
# 
# $$
#  \hat{y}_j = \frac{ \exp(o_j)}{\sum_{i=1}^3 \exp(o_i)} 
# $$
# 
# 

# In[ ]:


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  


# ## softmax回归模型
#  
# $$
#  \begin{aligned} \boldsymbol{o}^{(i)} &= \boldsymbol{x}^{(i)} \boldsymbol{W} + \boldsymbol{b},\\ \boldsymbol{\hat{y}}^{(i)} &= \text{softmax}(\boldsymbol{o}^{(i)}). \end{aligned} 
# $$
# 

# In[ ]:


def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# ## 定义损失函数
# 
# $$
# H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)},
# $$
#   
# 
# $$
# \ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),
# $$
#   
# 
# $$
# \ell(\boldsymbol{\Theta}) = -(1/n) \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}
# $$
#   
# 
# 

# In[ ]:


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# ## 定义准确率

# In[ ]:


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# In[ ]:


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# ## 训练模型

# In[ ]:


num_epochs, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step() 
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)


# ## 模型预测

# In[ ]:


X, y = iter(test_iter).next()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])


# # softmax的简洁实现

# In[ ]:


import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
data_path = os.path.abspath("../../") + "\\data"
from src.code.utils import *

print(torch.__version__)


# In[ ]:


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, root=data_path + '/FashionMNIST2065')


# ## 定义网络模型

# In[ ]:


num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

from collections import OrderedDict
net = nn.Sequential(
        OrderedDict([
           ('flatten', FlattenLayer()),
           ('linear', nn.Linear(num_inputs, num_outputs))])
        )


# ## 初始化模型参数

# In[ ]:


init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)


# ## 定义损失函数

# In[ ]:


loss = nn.CrossEntropyLoss()


# ## 定义优化函数

# In[ ]:


optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# ## 训练

# In[ ]:


num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

