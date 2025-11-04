---
tags:
  - ai
  - ds
aliases:
  - 循环神经网络
  - RNN
publish: true
---
### 关键公式

#### **前向传播 (Forward Pass)**

1.  **输入编码 (One-hot encoding)**:
    $$
    xs[t] = \text{one-hot}(inputs[t])
    $$
    每个输入字符被转换为一个独热向量（one-hot vector），其中只有一个位置为1，对应于该字符在词汇表中的索引。

2.  **隐藏状态计算 (Hidden State Update)**:
    $$
    h_{t} = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
    $$
    这是最核心的公式。它将当前输入$x_t$和上一个隐藏状态$h_{t-1}$进行加权求和，然后通过`tanh`非线性激活函数得到新的隐藏状态$h_t$。这个`tanh`激活函数将输出值压缩到$[-1, 1]$之间。
    

3.  **输出计算 (Output Score)**:
    $$
    y_t = W_{hy} h_t + b_y
    $$
    隐藏状态$h_t$被用来计算一个未归一化的对数概率向量$y_t$，也称为“scores”。这个向量的每个元素代表词汇表中对应字符的得分。

4.  **Softmax 概率 (Softmax Probability)**:
    $$
    p_t = \frac{\exp(y_t)}{\sum_j \exp(y_{t,j})}
    $$
    将$y_t$的得分通过softmax函数转换成一个概率分布$p_t$。这个向量中的每个元素代表下一个字符是该字符的概率。

5.  **损失函数 (Cross-Entropy Loss)**:
    $$
    L_t = -\log(p_{t, \text{target}})
    $$
    脚本使用交叉熵损失函数来衡量预测概率$p_t$与实际目标字符$targets[t]$之间的差异。总损失是所有时间步损失的累加。
$$L=\sum^T_{t=1}L_{t}$$
	总损失即对每个时间步上的损失求和
#### **反向传播 (Backward Pass)**

这是训练的核心步骤，目的是计算损失函数 $L$ 对所有参数的梯度。

* **从后往前计算：** 反向传播是按时间步从后往前进行的，从 $t=T$ 回溯到 $t=1$。MLP也是从后往前（因为链式法则是从输出层的梯度往前倒推），但再RNN里还有一点重要的原因是，我们需要知道损失对**隐藏状态** $h_t$ 的梯度 。即：
$$\nabla_{h_t} L = \nabla_{h_t} L_{\text{from output}} + \nabla_{h_t} L_{\text{from future}}$$
* **梯度传播：**
    1.  **从输出层开始：** 计算损失对输出层的梯度 $\nabla_y L_t$。
        $$\nabla_{y_t} L_t = p_{t} - \mathbb{I}(\text{target})$$
    2.  **传播到隐藏层：** 计算损失对隐藏状态的梯度 $\nabla_{h_t} L$。这个梯度由两部分组成：从当前时间步的输出层传来的梯度，和从下一个时间步的隐藏层传来的梯度。
        $$\nabla_{h_t} L = W_{hy}^T \cdot \nabla_{y_t} L_t + W_{hh}^T \cdot \nabla_{\text{raw}_{t+1}} L_{t+1}$$
    3.  **通过激活函数：** 计算损失对隐藏层原始输入的梯度 $\nabla_{\text{raw}_t} L$。
        $$\nabla_{\text{raw}_t} L = \nabla_{h_t} L \odot (1 - h_t^2)$$
    4.  **计算参数梯度：** 使用 $\nabla_{\text{raw}_t} L$ 来计算所有权重和偏置的梯度。
        * $\nabla_{W_{xh}} L_t = \nabla_{\text{raw}_t} L \cdot x_t^T$
        * $\nabla_{W_{hh}} L_t = \nabla_{\text{raw}_t} L \cdot h_{t-1}^T$
        * $\nabla_{W_{hy}} L_t = \nabla_{y_t} L_t \cdot h_t^T$
        * $\nabla_{b_h} L_t = \nabla_{\text{raw}_t} L$
        * $\nabla_{b_y} L_t = \nabla_{y_t} L_t$
* **累加梯度：** 将每个时间步计算出的梯度累加起来，得到总梯度，例如：
    $$\nabla_{W_{xh}} L_{total} = \sum_{t=1}^{T} \nabla_{W_{xh}} L_t$$

* **梯度裁剪（可选但常用）：** 为了防止梯度爆炸，对所有累积的梯度进行裁剪，将其值限制在一个预设的范围内。
**注意：Vanilla RNN很难解决[[10 - Areas/20 - Concepts/梯度消失]]问题**（只能通过特殊的初始化/[[10 - Areas/20 - Concepts/层归一化 layer normalization]]/[[10 - Areas/20 - Concepts/循环归一化 Recurrent Normalization]]来缓解）

### **5. 参数更新 (Parameter Update)**

使用优化算法（如随机梯度下降 SGD）来更新模型的参数，以最小化损失函数。

* **更新公式：**
    $$W_{new} = W_{old} - \alpha \cdot \nabla_W L_{total}$$
    其中 $\alpha$ 是**学习率**，控制每一步更新的幅度。

## MLP和RNN的区别

MLP（多层感知机）和RNN（循环神经网络）在数学上的核心区别在于它们的**前向传播公式**，特别是**如何处理时间维度**。

### **1. RNN 前向传播中的“记忆”**

RNN 的前向传播公式包含一个关键的**循环项**，它将上一个时间步的隐藏状态 $h_{t-1}$ 作为当前时间步的输入。这在数学上为模型赋予了处理序列数据的能力，因为它将历史信息编码在隐藏状态中。

$$h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

* $W_{hh}$ 权重矩阵和 $h_{t-1}$ 向量的乘积是**RNN 独有**的，它代表了模型对“记忆”的数学表示。
* $h_{t-1}$ 是前一个时间步的隐藏状态，它携带了之前的序列信息。

### **2. MLP 缺乏“记忆”**

相比之下，MLP 的前向传播是完全**静态**的。每一层的输出只依赖于当前层的输入，没有来自“之前”状态的输入。
$$h_1 = f_1(W_1x + b_1)$$$$h_2 = f_2(W_2h_1 + b_2)$$$$...$$$$y = f_L(W_Lh_{L-1} + b_L)$$
* MLP 的每一层都只接收来自前一层（在层级结构上）的输入，不涉及时间上的循环或反馈。
* **没有任何项**可以将信息从一个数据点（或时间步）传递到另一个，因此它无法处理序列数据。

### **3. 参数共享**

* **RNN**：RNN 在**所有时间步**之间共享相同的权重矩阵 ($W_{xh}$, $W_{hh}$, $W_{hy}$)。这使得它能够处理任意长度的序列，并且参数数量固定。
* **MLP**：MLP 的每一层都有独立的权重矩阵 ($W_1$, $W_2$, ...)。如果要处理序列，需要为每个时间步设计单独的输入和参数，这在数学上是不可扩展的。

总结来说，**RNN 在数学上引入了一个带有循环连接的方程**，通过 $h_{t-1}$ 项来表示**时间上的依赖性**和**参数共享**，而 MLP 的方程组是**纯粹的前馈式**的，缺乏这些特性。

---
### **代码实现详解**

我们现在来看看Karpathy的 [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086)

```python
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
```


#### **1. 数据准备 (`data I/O`)**

```python
# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
```

* `data = open('input.txt', 'r').read()`: 读入文本数据。
* `chars = list(set(data))`: 提取所有不重复的字符，构建词汇表。
* `char_to_ix` 和 `ix_to_char`: 创建字符到索引和索引到字符的映射，方便独热编码和解码。

#### **2. 模型参数和超参数 (`hyperparameters` & `model parameters`)**

```python
# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
```

* **`hidden_size`**: 隐藏层神经元的数量，代表了模型的“记忆”容量。
* **`seq_length`**: 序列长度，即RNN每次处理的字符数量。
* **`learning_rate`**: 学习率，控制每次梯度更新的步长。
* **`Wxh`, `Whh`, `Why`**: 权重矩阵，分别连接输入到隐藏层、隐藏层到隐藏层、隐藏层到输出层。
* **`bh`, `by`**: 偏置向量。

#### **3. `lossFun` 函数**

```python
def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity.
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
```

这个函数是模型的训练引擎，实现了前向传播和反向传播。

* **前向传播**:
    * 循环遍历输入序列的每个字符。
    * 将每个字符转换为独热向量`xs[t]`。$x_t = \text{one-hot}(inputs[t])$
    * 用前一个隐藏状态`hs[t-1]`和当前输入`xs[t]`计算新的隐藏状态`hs[t]`。$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$
    * 计算输出`ys[t]`，通过softmax得到概率`ps[t]`。$y_t = W_{hy}h_t + b_y$ , $p_t = \text{softmax}(y_t) = \frac{\exp(y_t)}{\sum_j \exp(y_{t,j})}$
    * 计算并累加[[10 - Areas/20 - Concepts/交叉熵损失 Cross-Entropy Loss]]`loss`。$L_t = -\log(p_{t, \text{target}})$

* **反向传播**:
	* 在计算总梯度时，我们需要对每个时间步的梯度进行累加，所以首先将累加器清零。
	在 RNN 的反向传播中，`dWxh`, `dWhh`, `dWhy` 分别代表了损失函数 $L$ 对三个权重矩阵的梯度：`Wxh`, `Whh`, 和 `Why`。
	* **`dWxh` (输入到隐藏层的权重梯度)**：
	    $\nabla_{W_{xh}} L = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{xh}}$
	    其中，每个时间步 $t$ 的梯度通过链式法则计算：
	    $\frac{\partial L_t}{\partial W_{xh}} = \frac{\partial L_t}{\partial \text{raw}_{t}} \cdot x_t^T$
	    代码中的 `np.dot(dhraw, xs[t].T)` 正是这一部分，`dhraw` 就是 $\nabla_{\text{raw}_{t}} L_t$。循环中的 `+=` 操作实现了对所有时间步梯度的累加。

	* **`dWhh` (隐藏层到隐藏层的权重梯度)**：
	    $\nabla_{W_{hh}} L = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$
	    其中，每个时间步 $t$ 的梯度通过链式法则计算：
	    $\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial \text{raw}_{t}} \cdot h_{t-1}^T$
	    代码中的 `np.dot(dhraw, hs[t-1].T)` 实现了这一部分，`dhraw` 同样是 $\nabla_{\text{raw}_{t}} L_t$。循环中的 `+=` 操作实现了累加。

	* **`dWhy` (隐藏层到输出层的权重梯度)**：
	    $\nabla_{W_{hy}} L = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hy}}$
	    其中，每个时间步 $t$ 的梯度通过链式法则计算：
	    $\frac{\partial L_t}{\partial W_{hy}} = \frac{\partial L_t}{\partial y_t} \cdot h_t^T$
	    代码中的 `np.dot(dy, hs[t].T)` 实现了这一部分，`dy` 就是 $\nabla_{y_t} L_t$。循环中的 `+=` 操作实现了累加。
	    
    * 使用`for t in reversed(...)`循环从后向前计算梯度。
    * `dy`是输出层的梯度。
    * `dWhy`和`dby`通过`dy`和`hs[t]`计算。
    * `dh`是隐藏层的梯度，它接收来自输出层和下一时间步的梯度。

    * `dhraw`是`tanh`激活函数反向传播的梯度。`dhraw`就是$nabla_{\text{raw}_t} L$
	    * $\nabla_{\text{raw}_t} L = \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial \text{raw}_t}$
	    * $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$
	    * $\nabla_{\text{raw}_t} L = (1 - h_t^2) \odot \nabla_{h_t} L$
    * `dWxh`, `dWhh`, `dbh`通过`dhraw`和相应的输入计算。
    * `np.clip(dparam, -5, 5, out=dparam)`: **梯度裁剪**，用于防止**[[10 - Areas/20 - Concepts/梯度爆炸]]**问题。

#### **4. `sample` 函数**

```python
def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes
```
这个函数用于从训练好的模型中生成文本，是模型能力的可视化。

* 给定一个初始隐藏状态`h`和一个种子字符`seed_ix`。
* 循环`n`次，每次执行一次前向传播：
    * 计算新的隐藏状态`h`。
    * 计算输出`y`和概率`p`。
    * 使用`np.random.choice`根据`p`分布随机采样下一个字符的索引`ix`。
    * 将新采样的字符作为下一个时间步的输入。
* 将所有采样的字符索引转换为文本并返回。

#### **5. 训练循环 (`while True`)**

```python
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter
```
* **[[10 - Areas/20 - Concepts/Adagrad]] 优化器**: 脚本使用一个手搓的[[10 - Areas/20 - Concepts/Adagrad]]来更新模型参数。`mem`变量用于累积梯度的平方，从而实现自适应的学习率。
* **序列切分**: 训练数据被分割成`seq_length`长的序列。`p`是数据指针，每次迭代前进`seq_length`步。
* **模型状态传递**: `hprev`保存了上一个序列块的最终隐藏状态，并在下一个序列块的开始时作为初始隐藏状态，从而保持了序列间的记忆连续性。
* **`smooth_loss`**: 这是一个平滑过的损失值，通过指数加权移动平均来跟踪训练进度，使损失曲线更平滑。
* **采样和打印**: 每100次迭代，脚本会调用`sample`函数生成一段文本，并打印当前的平滑损失，让用户观察模型的学习进度。