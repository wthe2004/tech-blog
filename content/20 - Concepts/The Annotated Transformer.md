---
{"publish":true,"created":"2025-10-14T19:52:27.139-04:00","modified":"2025-10-17T10:40:17.499-04:00","tags":["ai","ml","transformer"],"cssclasses":""}
---

感谢 [The Annotated Transformer]([nlp.seas.harvard.edu/](https://nlp.seas.harvard.edu/ "https://nlp.seas.harvard.edu/")) 和 [The Annotated Transformer中文翻译](https://github.com/mcxiaoxiao/annotated-transformer-Chinese)

### 准备工作

```python
!pip install -r requirements.txt
# #将被安装的包括 👇
# #pandas：处理数据，分析和管理。
# #torch：机器学习库。
# #torchdata：这是数据下载和预处理的工具。
# #torchtext：针对PyTorch框架的文本处理工具箱。
# #spacy：负责管理机器翻译的一些公开数据集
# #altair：互动式的绘图工具，用来制作漂亮的统计图表。
# #jupytext：可以在Markdown和Julia文件之间自由切换。
# #flake8：自动检查Python代码的风格。
# #black：代码格式化工具，可以自动调整代码的格式。
# #GPUtil：它是一个GPU使用的监控工具，可以实时查看GPU的状态和使用情况。
# #wandb：做训练记录的云服务
```

```python
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# warnings.filterwarnings("ignore")
# 这是作者自定义的一个变量 设置为 False 时，可以跳过 notebook 执行（例如，在调试模式下可以快速定位错误）
RUN_EXAMPLES = True
```

```python
# 这个 notebook 中使用的一些辅助工具函数，能方便地完成一些重复性的任务，减轻编程负担

def is_interactive_notebook():
    return __name__ == "__main__"

def show_example(fn, args=[]):
	# 这里就用上了我们之前设定的那个RUN_EXAMPLES这个变量
	# 这里是执行且返回，用于
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)

# 占位符，假优化器，继承了 PyTorch 的基础优化器类但实际上什么工作都不做。
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

# 同上，假学习率调度器
class DummyScheduler:
    def step(self):
        None
```

### 模型架构

```python
class EncoderDecoder(nn.Module):
    """
    一个标准的编码器-解码器架构。是本例和许多其他
    模型的基础。
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 源嵌入，即图中input embedding
        self.tgt_embed = tgt_embed  # 目标嵌入，即图中output embedding
        self.generator = generator  # 生成概率分布，即途图中Linear+Softmax的部分

    def forward(self, src, tgt, src_mask, tgt_mask):
        "接收并处理 mask 的 源（src） 和 目标（target） 序列。"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)  # 对源序列进行编码

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)  # 对目标序列进行解码
```

这段代码实际上就最简单的对应了这张图

![[99 - Attachments/images/vaswani2017attention-fig1.png]]

#### Encoder

##### 基础Encoder大框架

```python
def clones(module, N):
    "复制n个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

这个函数输入一个`Module`，输出一个把它复制了n遍的`Module List`. 你不能像调用普通`Module`那样去调用 `ModuleList`

```python
class Encoder(nn.Module):
    "包含 N 个层的堆叠"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # 创建 N 个 layer 的副本，并存储在 self.layers 中。类型是 nn.ModuleList
        self.norm = LayerNorm(layer.size)  # 创建一个 LayerNorm ，并存储在 self.norm 中
        
    def forward(self, x, mask):
        "逐层传递输入和掩码"
        for layer in self.layers:
            x = layer(x, mask)  # 逐层对输入 x 进行处理
        return self.norm(x)  # 对处理后的结果 x 进行 Layer Normalization（层归一化）
```

如果要对应Transformer的架构图，那这里`self.layers`对应了要$\times N$的哪部分，而`self.norm`没有被画出来。这里是实际实现上在所有 N 个 `EncoderLayer` 执行完毕后，额外增加一个归一化层。

`backward`函数不需要重载，但是`forward`需要。这是pytroch框架的惯例。

##### LayerNorm

```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2`
```
这个代码块实现的是 **层归一化（Layer Normalization）**。

它对应的计算公式如下：

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

* **$x$**: 网络的某一层输入数据。在代码中对应参数 `x`。
* **$\mu$**: 输入数据 $x$ 的 **均值** (mean)。
    * 代码实现: `mean = x.mean(-1, keepdim=True)`
    * 这里 `-1` 表示沿着最后一个维度计算均值。
* **$\sigma^2$**: 输入数据 $x$ 的 **方差** (variance)。$\sigma$ 则是标准差 (standard deviation)。
    * 代码实现: `std = x.std(-1, keepdim=True)`
* **$\epsilon$** (epsilon): 一个非常小的正数（例如 $1 \times 10^{-6}$），用于防止分母为零，增加数值稳定性。
    * 代码实现: `self.eps`
* **$\gamma$** (gamma): **缩放因子** (scale factor)。这是一个可学习的参数，模型在训练过程中会自己学习最优值。
    * 代码实现: `self.a_2` (初始化为全1的向量)
* **$\beta$** (beta): **平移因子** (shift factor)。这也是一个可学习的参数。
    * 代码实现: `self.b_2` (初始化为全0的向量)

##### Sublayer
```python
class SublayerConnection(nn.Module):
    """
    一个残差连接（residual connection）后跟一个层归一化（LayerNorm）
    为了代码的简洁性，将层归一化放在残差连接之前
    """

    def __init__(self, size, dropout):
        # size=d_model=512; dropout=0.1
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size) # (512)，用来定义a_2和b_2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "将残差连接应用于具有相同大小的任何子层。"
        # x (batch.size, sequence.len, 512)
        # sublayer 是一个具体的 MultiHeadAttention
        # 或者 Position-wise FeedForward 对象
        return x + self.dropout(sublayer(self.norm(x)))
        # x (30, 10, 512) -> norm (LayerNorm) -> (30, 10, 512)
        # -> sublayer (MultiHeadAttention or PositionwiseFeedForward)
        # -> (30, 10, 512) -> dropout -> (30, 10, 512)
        
        # 然后输入的x（没有走sublayer) + 上面的结果，
        # 即实现了残差相加的功能
        # 应用残差连接和层归一化，返回最终的输出
```

这种实现称为"Pre-LN" (Pre-Layer Normalization)，即在子层计算 **之前** 进行归一化。

**数据流动步骤分解：**

1.  `self.norm(x)`: **首先**，对输入 `x` 进行层归一化（LayerNorm）。
2.  `sublayer(...)`: **然后**，将归一化后的结果送入真正的子层（`sublayer`）进行计算。这个 `sublayer` 要么是多头注意力模块，要么是前馈网络模块。
3.  `self.dropout(...)`: 对子层的输出应用 Dropout，防止过拟合。
4.  `x + ...`: **最后**，将经过所有处理的子层输出与 **原始的输入 `x`** 相加，完成[[20 - Concepts/残差连接]]。

##### Encoder Layer

```python
class EncoderLayer(nn.Module):
    """
    编码器由自注意力和前馈神经网络组成（如下）。
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # 自注意力机制
        self.feed_forward = feed_forward  # 前馈神经网络
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 这里对应了Add and Norm的部分的功能，Add即残差连结，Norm是层归一化
        self.size = size

    def forward(self, x, mask):
        """
        参考图1（左侧）进行连接。
        """
        # 第一个子层连接：自注意力机制
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个子层连接：前馈神经网络
        return self.sublayer[1](x, self.feed_forward)

```

这是图中灰色Encoder块中内部的实现。

#### Decoder

##### 基础Decoder大框架

```python
class Decoder(nn.Module):
    "通用的 带有掩码（masking）的 N层解码器"

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)  # 克隆N个解码器层
        self.norm = LayerNorm(layer.size)  # 归一化层

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            # 应用每个解码器层，传递输入x、记忆memory、源掩码src_mask和目标掩码tgt_mask
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)  # 对输出进行归一化处理
```

与Encoder非常相似。

```python
class DecoderLayer(nn.Module):
    "解码器由自注意力机制（self-attn）、源注意力机制（src-attn）和前馈神经网络（feed forward）组成。"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size  # 解码器层的大小
        self.self_attn = self_attn  # 自注意力机制
        self.src_attn = src_attn  # 源注意力机制
        self.feed_forward = feed_forward  # 前馈神经网络
        self.sublayer = clones(SublayerConnection(size, dropout), 3)  # 克隆三个子层连接

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        参考图1（右侧）进行连接。
        """
        m = memory  # 记忆
        # 第一个子层连接：自注意力机制
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个子层连接：源注意力机制
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 第三个子层连接：前馈神经网络
        return self.sublayer[2](x, self.feed_forward)
```

这里用的是同一个sublayer，毕竟是同样的add和norm。

```python
def subsequent_mask(size):
    "屏蔽后续位置的注意力"
    # 定义注意力矩阵的形状
    attn_shape = (1, size, size)
    
    # 创建一个上三角矩阵
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    # 反转上三角矩阵
    return subsequent_mask == 0
```


返回的这个布尔矩阵就是最终的**注意力掩码（Attention Mask）**。

  * **`True`**: 表示模型在计算注意力时可以**保留**这个位置的分数。
  * **`False`**: 表示模型必须将这个位置的分数**屏蔽**掉（通常是将其设置为一个非常小的负数，如 `-inf`，这样在经过 softmax 后，其权重会趋近于0）。

我们来看看这个矩阵的含义：

  * **第1行**: `[True, False, False, False]` -\> 第1个单词只能关注它自己。
  * **第2行**: `[True, True, False, False]` -\> 第2个单词可以关注第1和第2个单词。
  * **第3行**: `[True, True, True, False]` -\> 第3个单词可以关注第1、2、3个单词。
  * **第4行**: `[True, True, True, True]` -\> 第4个单词可以关注所有4个单词。

这完美地实现了我们的目标：**在任何一个时间步 `t`，模型只能关注从位置 `0` 到 `t` 的所有输入，而不能关注 `t+1` 及其之后的位置。**

这个掩码最终会被应用到解码器的“自注意力（Self-Attention）”层中，确保了 Transformer 在生成式任务中的正确性。

### Attention

回忆一下公式

![[99 - Attachments/images/Scaled Dot-Product Attention.png]]

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Attention公式实现如下

```python
def attention(query, key, value, mask=None, dropout=None):
    "计算缩放点积 'Scaled Dot Product Attention'"

    d_k = query.size(-1)  # 获取查询向量的最后一个维度大小，即注意力的维度

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力得分

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 对掩码为0的位置进行填充，使得对应位置的注意力得分变为一个很小的负数

    p_attn = scores.softmax(dim=-1)  # 对注意力得分进行softmax归一化，得到注意力权重

    if dropout is not None:
        p_attn = dropout(p_attn)  # 对注意力权重进行dropout操作

    return torch.matmul(p_attn, value), p_attn  # 返回加权后的value和注意力权重
```

#### Multi-Head Attention

![[99 - Attachments/images/Multi-Head Attention.png]]

```python
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "接受模型大小和头部数量作为输入参数。"
        super().__init__()
        assert d_model % h == 0
        # 我们假设 d_v 总是等于 d_k
        self.d_k = d_model // h  # 每个头部的注意力维度
        self.h = h  # 头部的数量
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 线性变换层的集合
        #定义四个Linear networks, 每个的大小是(512, 512)的，
        #每个Linear network里面有两类可训练参数，Weights，
        #其大小为512*512，以及biases，其大小为512=d_model。
        self.attn = None  # 存储注意力权重的变量
        self.dropout = nn.Dropout(p=dropout)  # dropout层用于随机丢弃部分神经元的输出

    def forward(self, query, key, value, mask=None):
        "实现图2中的操作"
        # 注意，输入query的形状类似于(30, 10, 512)，
        # key.size() ~ (30, 11, 512), 
        #以及value.size() ~ (30, 11, 512)
        if mask is not None:
            # 将相同的掩码应用于所有头部
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # 批次大小

        # 1) 在批次中对所有线性投影进行处理，从 d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # 这里是前三个Linear Networks的具体应用，
        #例如query=(30,10, 512) -> Linear network -> (30, 10, 512) 
        #-> view -> (30,10, 8, 64) -> transpose(1,2) -> (30, 8, 10, 64)
        #，其他的key和value也是类似地，
        #从(30, 11, 512) -> (30, 8, 11, 64)。
        
        # 2) 在批次中对所有投影向量应用注意力机制
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        #调用上面定义好的attention函数，输出的x形状为(30, 8, 10, 64)；
        #attn的形状为(30, 8, 10=target.seq.len, 11=src.seq.len)
        
        # 3) 使用视图进行"拼接"，然后应用最后一个线性层
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        # x ~ (30, 8, 10, 64) -> transpose(1,2) -> 
        #(30, 10, 8, 64) -> contiguous() and view -> 
        #(30, 10, 8*64) = (30, 10, 512)
        del query
        del key
        del value
        return self.linears[-1](x)
        #执行第四个Linear network，把(30, 10, 512)经过一次linear network，
        #得到(30, 10, 512).
```

理论上来讲，`d_k`即Key向量的维度（which始终等于向量的维度）与`d_v`即Value向量的维度可以不同，但将其设置为相同可以便于线性层对其维度；矩阵运算又在处理形状规整的张量时效率最高。况且没有充分的理由使他们不同，如果`d_v`与`d_k`不同还会导致压缩/扩张维度时导致的信息丢失或额外不必要计算量。

这里强调一下`mask`。掩码有三种情况：编码器自注意力层 (Encoder Self-Attention)，解码器自注意力层 (Decoder Self-Attention)，编码器-解码器注意力层 (Encoder-Decoder Attention)。和两种分类：用于补齐句子长度的**填充掩码 (Padding Mask)** 和用于训练解码器时掩盖未来位置的**前瞻掩码 (Look-ahead Mask / Subsequent Mask)**

#### 逐位前馈网络

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$
线性变换升维 -> 激活函数 -> 线性变换降维

```python
class PositionwiseFeedForward(nn.Module):
    "实现 FFN（Feed-Forward Network）"
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model = 512
        # d_ff = 2048 = 512*4
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        # 构建第一个全连接层，(512, 2048)，其中有两种可训练参数：
        # weights矩阵，(512, 2048)，以及
        # biases偏移向量, (2048)
        self.w_2 = nn.Linear(d_ff, d_model)
        # 构建第二个全连接层, (2048, 512)，两种可训练参数：
        # weights矩阵，(2048, 512)，以及
        # biases偏移向量, (512)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape = (batch.size, sequence.len, 512)
        # 例如, (30, 10, 512)
        return self.w_2(self.dropout(self.w_1(x).relu()))
        # x (30, 10, 512) -> self.w_1 -> (30, 10, 2048)
        # -> relu -> (30, 10, 2048) 
        # -> dropout -> (30, 10, 2048)
        # -> self.w_2 -> (30, 10, 512)是输出的shape
```

注意到这里代码只有w没有b了吗？那是因为虽然层被起名叫W，但它其实是一个完整的层。pytorch会处理W和b两个分开的参数。

#### Embedding和Softmax

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

* `d_model`: 模型的维度，也就是你希望用来表示一个单词的向量的长度。在 Transformer 论文中，这个值是 512。
* `vocab`: 词汇表的大小 (Vocabulary Size)。
* `self.lut = nn.Embedding(vocab, d_model)`: 查找表（Look-Up Table）。这是整个类的核心。
    * `nn.Embedding` 是 PyTorch 中专门用来处理词嵌入的模块。
    * 这个表本质上是一个形状为 `(vocab, d_model)` 的矩阵。
    * 你可以把它想象成一个有 `vocab` 行、`d_model` 列的表格。每一行代表词汇表中的一个单词，该行的 `d_model` 个数字就是这个单词的向量表示。这些向量在模型训练过程中会被不断学习和优化。
* `self.d_model = d_model`: 将模型维度存储下来，方便在 `forward` 方法中使用。

#### 位置编码


对于序列中的任意位置 `pos` (position)，其位置编码向量 `PE` 的计算方式如下：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        #d_model=512,dropout=0.1,
        #max_len=5000代表事先准备好长度为5000的序列的位置编码
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 在对数空间中计算位置编码
        pe = torch.zeros(max_len, d_model)
        #(5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
        #每个位置用一个512维度向量来表示其位置编码
        position = torch.arange(0, max_len).unsqueeze(1)
		# torch.arange(0, max_len) 生成 [0, 1, 2, ..., 4999] 的一维张量。 # .unsqueeze(1) 将其形状从 (5000,) 变为 (5000, 1) 的列向量。 # 这个形状变换是为了后续能和 div_term 进行广播（broadcast）运算。
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
            # 这是一种算分母的那个指数的代数上实现的一个小技巧。在对数空间计算，数值上更稳定一点。
            # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
        )
        pe[:, 0::2] = torch.sin(position * div_term)# 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term)# 奇数下标的位置
        # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # 接受1.Embeddings的词嵌入结果x，
        #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
        #例如，假设x是(30,10,512)的一个tensor，
        #30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
        #则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
        #在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
        #保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
        return self.dropout(x)
    # 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数。
```

#### 定义完整模型

这是定义的最后一步。一个从超参数到完整模型的函数。

```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Helper: 根据超参数构建一个模型。
    """
    c = copy.deepcopy # 这是一个快捷方式，在这个函数内部可以访问
    attn = MultiHeadedAttention(h, d_model)  # 创建多头注意力机制实例
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 创建位置前馈网络实例
    position = PositionalEncoding(d_model, dropout)  # 创建位置编码实例
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),  # 创建编码器实例
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),  # 创建解码器实例
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),  # 创建源语言嵌入层实例
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),  # 创建目标语言嵌入层实例
        Generator(d_model, tgt_vocab),  # 创建生成器实例
    )

    # 这是他们代码中的重要部分。
    # 使用 Glorot / fan_avg 初始化参数。
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
```

这里来讲一下权重初始化这一块。此处的意思是，遍历整个 Transformer 模型的所有可训练参数。如果这个参数是一个维度大于1的张量（即，它是一个权重矩阵而不是一个偏置向量），那么就使用 [[20 - Concepts/Xavier 均匀初始化]]来重新初始化它的值。
* `model.parameters()` 是一个 PyTorch 方法，它会返回一个迭代器，其中包含了 `model` 这个庞大对象中**所有可训练的参数**。这包括了模型中每一个 `nn.Linear` 层的权重矩阵（weight）和偏置向量（bias），每一个 `nn.Embedding` 层的嵌入矩阵等等。Xavier 初始化的理论基础主要针对的是权重矩阵的乘法效应，而偏置向量通常保持默认的零或小值初始化即可。

### 训练

#### Batch

```python
class Batch:
    """用于在训练过程中保存一个数据批次及其掩码的对象。"""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>，用于指定填充标记的索引
        # src: 源语言序列，(batch.size, src.seq.len)
        # 二维tensor，第一维度是batch.size；第二个维度是源语言句子的长度
        # 例如：[ [2,1,3,4], [2,3,1,4] ]这样的二行四列的，
        # 1-4代表每个单词word的id
        
        # trg: 目标语言序列，默认为空，其shape和src类似
        # (batch.size, trg.seq.len)，
        # 二维tensor，第一维度是batch.size；第二个维度是目标语言句子的长度
        # 例如trg=[ [2,1,3,4], [2,3,1,4] ] for a "copy network"
        # (输出序列和输入序列完全相同）
        
        # pad: 源语言和目标语言统一使用的 位置填充符号，'<blank>'
        # 所对应的id，这里默认为0
        # 例如，如果一个source sequence，长度不到4，则在右边补0
        # [1,2] -> [1,2,0,0]
        self.src = src  # 源序列张量
        self.src_mask = (src != pad).unsqueeze(-2)
        # 源序列的掩码张量，用于遮盖填充位置
        # 会把有内容的位置标注为1，没有的标注为0
        # src = (batch.size, seq.len) -> != pad -> 
        # (batch.size, seq.len) -> usnqueeze ->
        # (batch.size, 1, seq.len) 相当于在倒数第二个维度扩展
        # e.g., src=[ [2,1,3,4], [2,3,1,0] ]对应的是
        # src_mask=[ [[1,1,1,1], [1,1,1,0]] ]
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # 目标序列张量，去除最后一个位置的标记
            # trg 相当于目标序列的前N-1个单词的序列
            #（去掉了最后一个词）
            self.tgt_y = tgt[:, 1:]  # 目标序列张量的下一个位置的标记
            # trg_y 相当于目标序列的后N-1个单词的序列
            # (去掉了第一个词）
            # 目的是(src + trg) 来预测出来(trg_y)，
            self.tgt_mask = self.make_std_mask(self.tgt, pad)  # 目标序列的掩码张量，用于遮盖填充位置和未来位置
            self.ntokens = (self.tgt_y != pad).data.sum()  # 目标序列中非填充标记的数量

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个掩码，用于隐藏填充位置和未来的词。"
        # 这里的tgt类似于：
        #[ [2,1,3], [2,3,1] ] （最初的输入目标序列，分别去掉了最后一个词
        # pad=0, '<blank>'的id编号
        tgt_mask = (tgt != pad).unsqueeze(-2)  # 创建一个掩码张量，用于遮盖填充位置
        # 得到的tgt_mask类似于
        # tgt_mask = tensor([[[1, 1, 1]],[[1, 1, 1]]], dtype=torch.uint8)
        # shape=(2,1,3)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )  # 与一个用于遮盖未来位置的掩码张量相与，以得到最终的目标序列掩码张量
        # 先看subsequent_mask, 其输入的是tgt.size(-1)=3
        # 这个函数的输出为= tensor([[[1, 0, 0],
        # [1, 1, 0],
        # [1, 1, 1]]], dtype=torch.uint8)
        # type_as 把这个tensor转成tgt_mask.data的type(也是torch.uint8)
        
        # 这样的话，&的两边的tensor分别是(2,1,3), (1,3,3);
        #tgt_mask = tensor([[[1, 1, 1]],[[1, 1, 1]]], dtype=torch.uint8)
        #and
        # tensor([[[1, 0, 0], [1, 1, 0], [1, 1, 1]]], dtype=torch.uint8)
        
        # (2,3,3)就是得到的tensor
        # tgt_mask.data = tensor([[[1, 0, 0],
        # [1, 1, 0],
        # [1, 1, 1]],

        #[[1, 0, 0],
        # [1, 1, 0],
        # [1, 1, 1]]], dtype=torch.uint8)
        return tgt_mask
```

