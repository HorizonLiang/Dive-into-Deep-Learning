#!/usr/bin/env python
# coding: utf-8

# # 线性回归

# ## 矢量计算
# 在模型训练或预测时，我们常常会同时处理多个数据样本并用到矢量计算。在介绍线性回归的矢量计算表达式之前，让我们先考虑对两个向量相加的两种方法。
# 
# 
# 1. 向量相加的一种方法是，将这两个向量按元素逐一做标量加法。
# 2. 向量相加的另一种方法是，将这两个向量直接做矢量加法。

# In[2]:
import torch
import time
import os

# In[3]:
n = 1000
a = torch.ones(n)
b = torch.ones(n)


# In[7]:


class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # start the timer
        self.start_time = time.time()

    def stop(self):
        # stop the timer and record time into a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        # calculate the average and return
        return sum(self.times)/len(self.times)

    def sum(self):
        # return the sum of recorded time
        return sum(self.times)


# 将两个向量使用for循环按元素逐一做标量加法:

# In[6]:


timer = Timer()
c = torch.zeros(n)
for i in range(n):
    c[i] = a[i] + b[i]
'%.5f sec' % timer.stop()


# 使用torch来将两个向量直接做矢量加法：

# In[8]:


timer.start()
d = a + b
'%.5f sec' % timer.stop()


# ## 线性回归模型从零开始的实现

# In[10]:


# import packages and modules
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

print(torch.__version__)


# ### 生成数据集
# 使用线性模型来生成数据集，生成一个1000个样本的数据集，下面是用来生成数据的线性关系：
# 
# $$
# \mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b
# $$
# 
# 

# In[13]:


num_inputs = 2
num_examples = 1000

true_w = [2,-3.4]
true_b = 4.2


# In[19]:


features = torch.randn(num_examples,num_inputs,dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)


# In[21]:


plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)


# ### 读取数据集

# In[22]:


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # random read 10 samples
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # the last time may be not enough for a whole batch
        yield  features.index_select(0, j), labels.index_select(0, j)


# In[23]:


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# ### 初始化模型参数

# In[24]:


w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# ### 定义模型
# 定义用来训练参数的训练模型：
# 
# $$
# \mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b
# $$
# 
# 

# In[26]:


def linreg(X, w, b):
    return torch.mm(X, w) + b


# ### 定义损失函数
# 我们使用的是均方误差损失函数：
# $$
# l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2,
# $$
# 

# In[27]:


def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# ### 定义优化函数
# 优化函数使用的是小批量随机梯度下降：
# 
# $$
# (\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)
# $$

# In[28]:


def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track


# ### 训练

# In[29]:


lr = 0.03
num_epochs = 5

net = linreg
loss = squared_loss

for epoch in range(num_epochs):

    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  
        l.backward()  
        sgd([w, b], lr, batch_size)  
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))


# In[30]:


w, true_w, b, true_b


# ## 使用pytorch的简洁实现

# In[31]:


import torch
from torch import nn
import numpy as np
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')


# ### 生成数据集

# In[32]:


num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


# ### 读取数据集

# In[43]:


import torch.utils.data as Data

batch_size = 10

dataset = Data.TensorDataset(features, labels)

data_iter = Data.DataLoader(
    dataset=dataset,            
    batch_size=batch_size,      
    shuffle=True,               
    num_workers=2,           
)


# In[34]:


for X, y in data_iter:
    print(X, '\n', y)
    break


# ### 定义模型

# In[35]:


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()      
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net)


# In[36]:


net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    )

net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
        ]))

print(net)
print(net[0])


# ### 初始化模型参数

# In[37]:


from torch.nn import init

init.normal_(net[0].weight, mean=0.0, std=0.01)
init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly


# In[38]:


for param in net.parameters():
    print(param)


# ### 定义损失函数

# In[39]:


loss = nn.MSELoss()


# In[40]:


import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)


# ### 训练

# In[41]:


num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


# In[42]:


dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)


# In[ ]:




