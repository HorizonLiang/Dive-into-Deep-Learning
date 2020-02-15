#!/usr/bin/env python
# coding: utf-8

# ## 读取数据集

# In[3]:


import sys
import os
data_path = os.path.abspath("../../") + "\\data"


# In[6]:


with open(data_path + '/jaychou_lyrics4703/jaychou_lyrics.txt',encoding="utf-8") as f:
    corpus_chars = f.read()
print(len(corpus_chars))
print(corpus_chars[: 40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[: 10000]


# ## 建立字符索引

# In[7]:


idx_to_char = list(set(corpus_chars))
char_to_idx = {char: i for i, char in enumerate(idx_to_char)}
vocab_size = len(char_to_idx)
print(vocab_size)

corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[: 20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)


# In[8]:


def load_data_jay_lyrics():
    with open(data_path + '/jaychou_lyrics4703/jaychou_lyrics.txt',encoding="utf-8") as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# ## 时序数据的采样
# 
# 在训练中我们需要每次随机读取小批量样本和标签。与之前章节的实验数据不同的是，时序数据的一个样本通常包含连续的字符。假设时间步数为5，样本序列为5个字符，即“想”“要”“有”“直”“升”。该样本的标签序列为这些字符分别在训练集中的下一个字符，即“要”“有”“直”“升”“机”，即$X$=“想要有直升”，$Y$=“要有直升机”。
# 
# 现在我们考虑序列“想要有直升机，想要和你飞到宇宙去”，如果时间步数为5，有以下可能的样本和标签：
# * $X$：“想要有直升”，$Y$：“要有直升机”
# * $X$：“要有直升机”，$Y$：“有直升机，”
# * $X$：“有直升机，”，$Y$：“直升机，想”
# * ...
# * $X$：“要和你飞到”，$Y$：“和你飞到宇”
# * $X$：“和你飞到宇”，$Y$：“你飞到宇宙”
# * $X$：“你飞到宇宙”，$Y$：“飞到宇宙去”
# 
# 可以看到，如果序列的长度为$T$，时间步数为$n$，那么一共有$T-n$个合法的样本，但是这些样本有大量的重合，我们通常采用更加高效的采样方式。我们有两种方式对时序数据进行采样，分别是随机采样和相邻采样。
# 
# ### 随机采样
# 
# 下面的代码每次从数据里随机采样一个小批量。其中批量大小`batch_size`是每个小批量的样本数，`num_steps`是每个样本所包含的时间步数。
# 在随机采样中，每个样本是原始序列上任意截取的一段序列，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。

# In[9]:


import torch
import random
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):

    num_examples = (len(corpus_indices) - 1) // num_steps
    example_indices = [i * num_steps for i in range(num_examples)]
    random.shuffle(example_indices)

    def _data(i):
        return corpus_indices[i: i + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0, num_examples, batch_size):
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)


# In[10]:


my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')


# ### 相邻采样
# 
# 在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻。

# In[11]:


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size
    corpus_indices = corpus_indices[: corpus_len]
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# In[12]:


for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')


# In[ ]:




