#!/usr/bin/env python
# coding: utf-8

# # 多层感知机

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("../../.."))
from src.code.utils import *
print(torch.__version__)


# In[3]:


def xyplot(x_vals, y_vals, name):
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')


# In[5]:


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')


# In[6]:


y.sum().backward()
xyplot(x, x.grad, 'grad of relu')


# #### Sigmoid函数
# sigmoid函数可以将元素的值变换到0和1之间：
# 
# 
# $$
# \text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.
# $$

# In[8]:


y = x.sigmoid()
xyplot(x, y, 'sigmoid')


# In[9]:


x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')


# #### tanh函数
# tanh（双曲正切）函数可以将元素的值变换到-1和1之间：
# 
# 
# $$
# \text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.
# $$

# In[10]:


y = x.tanh()
xyplot(x, y, 'tanh')


# In[11]:


x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')


# ## 多层感知机从零开始的实现
# 
# 多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。以单隐藏层为例并沿用本节之前定义的符号，多层感知机按以下方式计算输出：
# 
# 
# $$
#  \begin{aligned} \boldsymbol{H} &= \phi(\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h),\\ \boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o, \end{aligned} 
# $$
# 
# 
# 其中$\phi$表示激活函数。

# In[12]:


import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../../.."))
from src.code.utils import *
print(torch.__version__)


# In[13]:


data_path = os.path.abspath("../../") + "\\data"


# In[15]:


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size,root=data_path + '/FashionMNIST2065')


# ### 定义模型参数

# In[ ]:


num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


# ### 定义激活函数

# In[ ]:


def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


# ### 定义网络

# In[ ]:


def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


# ### 定义损失函数

# In[ ]:


loss = torch.nn.CrossEntropyLoss()


# ### 训练

# In[ ]:


num_epochs, lr = 5, 100.0

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


# ## 多层感知机pytorch实现

# In[ ]:


import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append(os.path.abspath("../../.."))
from src.code.utils import *

print(torch.__version__)


# ### 初始化模型和各个参数

# In[ ]:


num_inputs, num_outputs, num_hiddens = 784, 10, 256
    
net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )
    
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)


# ### 训练

# In[ ]:


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size,root=data_path + '/FashionMNIST2065')
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

