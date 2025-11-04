---
{"publish":true,"created":"2025-10-19T10:59:40.301-04:00","modified":"2025-10-22T12:07:55.160-04:00","tags":["ai"],"cssclasses":""}
---

本文是有关Diffusion model, 尤其是DDPM的简要讲解。

### The corruption process (adding noise to data)

```python
def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount
```

非常简单的添加随机数。如果 amount = 0，返回未经任何修改的输入。如果 amount = 1，返回无关输入的纯噪声。

### What a UNet is, and how to implement an extremely minimal one from scratch

UNet 由一个“收缩路径”和一个“扩展路径”组成，数据通过收缩路径被压缩，再通过扩展路径恢复到原始维度（类似于自编码器），但它还具备跳跃连接，允许在不同层级的信息和梯度流动。

![[99 - Attachments/images/UNet.png]]

```python
class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x
```


这个"U"形结构包含两个主要部分：
1.  **编码器 (Encoder)**：也称为下采样路径（Down Path）。它像一个标准的卷积网络一样，通过一系列卷积和池化操作来提取特征，同时逐渐减小特征图的空间维度（高度和宽度）。
2.  **解码器 (Decoder)**：也称为上采样路径（Up Path）。它将编码器提取的低分辨率特征图逐渐上采样，恢复到原始图像的分辨率。
3.  **跳跃连接 (Skip Connections)**：这是U-Net的核心。它将编码器中较高分辨率的特征图“跳跃”连接到解码器中相应分辨率的层。这允许解码器在重建图像时，同时利用深层的、抽象的特征（来自编码器底部）和浅层的、高分辨率的特征（来自编码器顶部），这对于精确的像素级定位至关重要。

---

#### 1. `__init__` (模型结构定义)

`__init__` 方法用于初始化模型的所有构建块（即神经网络层）。

* `self.down_layers` (下采样层/编码器):
    * 这是一个 `ModuleList`，包含3个 `nn.Conv2d` (2D卷积) 层。
    * `nn.Conv2d(in_channels, 32, ...)`: 第1层。通道数从 `in_channels` (默认为1，如灰度图) 变为 32。
    * `nn.Conv2d(32, 64, ...)`: 第2层。通道数从 32 变为 64。
    * `nn.Conv2d(64, 64, ...)`: 第3层。通道数从 64 保持为 64。
    * **关键参数**: `kernel_size=5, padding=2`。
        * 这是一个“**Same Padding**”设置。当 $padding = (kernel\_size - 1) / 2$ 时（这里 $2 = (5-1)/2$），卷积操作**不会改变**特征图的高度(H)和宽度(W)。

* `self.up_layers` (上采样层/解码器):
    * 同样是包含3个卷积层的 `ModuleList`。
    * `nn.Conv2d(64, 64, ...)`: 第1层。通道数 64 -> 64。
    * `nn.Conv2d(64, 32, ...)`: 第2层。通道数 64 -> 32。
    * `nn.Conv2d(32, out_channels, ...)`: 第3层。通道数 32 -> `out_channels` (默认为1，如二值分割掩码)。
    * 同样使用 `kernel_size=5, padding=2` 来保持 H/W 不变。

* `self.act = nn.SiLU()`:
    * 定义激活函数。`SiLU` (Sigmoid Linear Unit)，也常被称为 **Swish**。它是一个平滑的、非单调的激活函数，通常表现优于 ReLU。

* `self.downscale = nn.MaxPool2d(2)`:
    * 下采样操作。使用步长为2的最大池化，这将使特征图的 H 和 W **减半**。

* `self.upscale = nn.Upsample(scale_factor=2)`:
    * 上采样操作。使用 `scale_factor=2` 将特征图的 H 和 W **加倍**。默认情况下，它使用"nearest"（最近邻）插值，这是一种简单、非学习性的上采样方法。

---

#### 2. `forward` (数据流)

`forward` 方法定义了数据（即你的输入图像 `x`）如何流经你在 `__init__` 中定义的层。

下面我们一步步追踪这个流程，假设输入图像 `x` 的尺寸为 `(B, 1, 128, 128)` (Batch, Channels, Height, Width)。

##### **Part 1: 下采样路径 (Encoder)**

`h = []`：初始化一个空列表，用于存储跳跃连接的特征图。

* **循环 (i=0): `down_layers[0]`**
    1.  `l(x)`: `x` 通过第1个卷积层 (In->32)。尺寸: `(B, 32, 128, 128)` (尺寸不变)。
    2.  `self.act(...)`: 通过 SiLU 激活。
    3.  `if i < 2` (True):
        * `h.append(x)`: **存储**这个激活后的特征图 `(B, 32, 128, 128)` 到列表 `h` 中。`h` 现在是 `[h_layer0]`。
        * `x = self.downscale(x)`: `x` 被最大池化。尺寸变为 `(B, 32, 64, 64)`。

* **循环 (i=1): `down_layers[1]`**
    1.  `l(x)`: `x` (尺寸 64x64) 通过第2个卷积层 (32->64)。尺寸: `(B, 64, 64, 64)`。
    2.  `self.act(...)`: 通过 SiLU 激活。
    3.  `if i < 2` (True):
        * `h.append(x)`: **存储**这个特征图 `(B, 64, 64, 64)`。`h` 现在是 `[h_layer0, h_layer1]`。
        * `x = self.downscale(x)`: `x` 再次被池化。尺寸变为 `(B, 64, 32, 32)`。

* **循环 (i=2): `down_layers[2]`**
    1.  `l(x)`: `x` (尺寸 32x32) 通过第3个卷积层 (64->64)。尺寸: `(B, 64, 32, 32)`。
    2.  `self.act(...)`: 通过 SiLU 激活。
    3.  `if i < 2` (False): 不执行 `h.append` 和 `downscale`。

**编码器结束时**:
* `x` 是**瓶颈层 (Bottleneck)** 的特征图，尺寸为 `(B, 64, 32, 32)`。
* `h` 列表包含两个用于跳跃连接的特征图：`h[0]` (128x128, 32通道) 和 `h[1]` (64x64, 64通道)。

---

##### **Part 2: 上采样路径 (Decoder)**

现在，`x` (瓶颈特征) 进入上采样循环。

* **循环 (i=0): `up_layers[0]`**
    1.  `if i > 0` (False): 不执行上采样或跳跃连接。
    2.  `l(x)`: `x` (尺寸 32x32) 通过第1个上采样卷积 (64->64)。尺寸: `(B, 64, 32, 32)`。
    3.  `self.act(...)`: 通过 SiLU 激活。`x` 现在的尺寸仍为 `(B, 64, 32, 32)`。

* **循环 (i=1): `up_layers[1]`**
    1.  `if i > 0` (True):
        * `x = self.upscale(x)`: `x` 被上采样。尺寸从 `(B, 64, 32, 32)` 变为 `(B, 64, 64, 64)`。
        * `h.pop()`: 从 `h` 列表中**弹出**最后一个元素。`h` 是 `[h_layer0, h_layer1]`，所以 `h.pop()` 返回 `h_layer1` (尺寸 `(B, 64, 64, 64)`）。
        * `x += ...`: **✨ 执行跳跃连接!** ✨ 将上采样后的 `x` 与 `h_layer1` 逐元素相加。
    2.  `l(x)`: 这个融合后的特征图 (64x64) 通过第2个上采样卷积 (64->32)。尺寸: `(B, 32, 64, 64)`。
    3.  `self.act(...)`: 通过 SiLU 激活。

* **循环 (i=2): `up_layers[2]`**
    1.  `if i > 0` (True):
        * `x = self.upscale(x)`: `x` 被上采样。尺寸从 `(B, 32, 64, 64)` 变为 `(B, 32, 128, 128)`。
        * `h.pop()`: 从 `h` 列表中弹出最后一个元素。`h` 现在只剩 `[h_layer0]`，所以 `h.pop()` 返回 `h_layer0` (尺寸 `(B, 32, 128, 128)`）。
        * `x += ...`: **✨ 执行第二次跳跃连接!** ✨ 将上采样后的 `x` 与 `h_layer0` 相加。
    2.  `l(x)`: 融合后的特征图 (128x128) 通过第3个上采样卷积 (32->`out_channels`)。尺寸: `(B, out_channels, 128, 128)`。
    3.  `self.act(...)`: 通过 SiLU 激活。

**解码器结束时**:
* `x` 是最终的输出特征图，尺寸为 `(B, out_channels, 128, 128)`。

---

##### **Part 3: 返回**

* `return x`:
    * 返回最终的输出。这个输出图的空间维度 (H, W) 与输入图相同，但通道数变为了 `out_channels`。对于分割任务，这个输出图的每个像素值可以代表对应类别的“分数”或“概率”（如果最后再加一个 Sigmoid/Softmax）。

---
#### 总结

这个 `BasicUNet` 的结构可以概括为：

1.  **输入 (H, W)**
2.  (Conv -> SiLU) -> **存 `h[0]`** -> Pool (H/2, W/2)
3.  (Conv -> SiLU) -> **存 `h[1]`** -> Pool (H/4, W/4)
4.  (Conv -> SiLU) -> **瓶颈层**
5.  (Conv -> SiLU) -> 第一个上采样块 (H/4, W/4)
6.  Upsample (H/2, W/2) -> **加 `h[1]`** -> (Conv -> SiLU)
7.  Upsample (H, W) -> **加 `h[0]`** -> (Conv -> SiLU)
8.  **输出 (H, W)**
### Diffusion model training

训练的流程简单来讲就是：

- Get a batch of data
- Corrupt it by random amounts
- Feed it through the model
- Compare the model predictions with the clean images to calculate our loss
- Update the model’s parameters accordingly.



```python
# Dataloader (you can mess with batch size)
batch_size = 128
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# How many runs through the data should we do?
n_epochs = 3

# Create the network
net = BasicUNet()
net.to(device)

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
losses = []

# The training loop
for epoch in range(n_epochs):

    for x, y in train_dataloader:

        # Get some data and prepare the corrupted version
        x = x.to(device)  # Data on the GPU
        noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
        noisy_x = corrupt(x, noise_amount)  # Create our noisy x

        # Get the model prediction
        pred = net(noisy_x)

        # Calculate the loss
        loss = loss_fn(pred, x)  # How close is the output to the true 'clean' x?

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
    print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")

# View the loss curve
plt.plot(losses)
plt.ylim(0, 0.1)
```


### Sampling theory

```python
n_steps = 40
x = torch.rand(64, 1, 28, 28).to(device)
for i in range(n_steps):
    # noise_amount = torch.ones((x.shape[0],)).to(device) * (1 - (i / n_steps))  # Starting high going low
    with torch.no_grad():
        pred = net(x)
    mix_factor = 1 / (n_steps - i)
    x = x * (1 - mix_factor) + pred * mix_factor

```

重要的只有更新公式：

`x = x * (1 - mix_factor) + pred * mix_factor`

在数学上，这叫做[[10 - Areas/20 - Concepts/线性插值 Linear Interpolation]]。我们可以把它写成：

$$
x^{(i+1)} = (1 - \alpha_i) \cdot x^{(i)} + \alpha_i \cdot p^{(i)}
$$

其中：
* $x^{(i)}$ 是第 $i$ 步的图像（`x`）。
* $x^{(i+1)}$ 是更新后的图像。
* $p^{(i)}$ 是模型在第 $i$ 步的预测（`pred`）。
* $\alpha_i$ 是第 $i$ 步的混合因子（`mix_factor`）。

这个公式的含义是：“新的图像 $x^{(i+1)}$ 是 $x^{(i)}$ 和 $p^{(i)}$ 之间的点，我们从 $x^{(i)}$ 朝着 $p^{(i)}$ 移动了 $\alpha_i$ 比例的距离。”

#### 关键公式：混合因子

这个算法的“魔法”在于 `mix_factor` 的选择：
`mix_factor = 1 / (n_steps - i)`

$\alpha_i = \frac{1}{n - i}$，其中 $n$ 是总步数 `n_steps`， $i$ 是当前步数（从 $0$ 到 $n-1$）。

#### 为什么是这个公式？（“理想情况”分析）

为了理解为什么这个公式有效，我们先做一个**简化的理想假设**：
假设我们的模型 `net` 是**完美且一致的**。无论给它看多么嘈杂的 $x^{(i)}$，它**总是**能完美地预测出那个唯一的、最终的清晰图像。我们称这个理想图像为 $P$。

在这个理想情况下， $p^{(i)} = P$ （一个常数）。
我们的更新规则变成了：
$$
x^{(i+1)} = (1 - \alpha_i) \cdot x^{(i)} + \alpha_i \cdot P
$$

现在，让我们以 $n=5$ (n_steps=5) 为例，追踪从 $x^{(0)}$（纯噪声）到 $P$（目标）的**距离**。
* 我们定义“总距离”为 $D_{total} = P - x^{(0)}$。

**Step i=0:**
* $\alpha_0 = \frac{1}{5 - 0} = \frac{1}{5}$
* $x^{(1)} = (1 - \frac{1}{5})x^{(0)} + \frac{1}{5}P$
* $x^{(1)} = x^{(0)} + \frac{1}{5}(P - x^{(0)})$
* **含义**: 我们从 $x^{(0)}$ 出发，朝着 $P$ 移动了**总距离的 1/5**。
* **剩余距离**: $P - x^{(1)} = (1 - \frac{1}{5})(P - x^{(0)}) = \frac{4}{5} D_{total}$。

**Step i=1:**
* $\alpha_1 = \frac{1}{5 - 1} = \frac{1}{4}$
* $x^{(2)} = (1 - \frac{1}{4})x^{(1)} + \frac{1}{4}P$
* $x^{(2)} = x^{(1)} + \frac{1}{4}(P - x^{(1)})$
* **含义**: 我们朝着 $P$ 移动了**剩余距离的 1/4**。
* **这一步移动了多远？**
    * $\text{Move} = \frac{1}{4}(P - x^{(1)}) = \frac{1}{4} \left( \frac{4}{5} D_{total} \right) = \frac{1}{5} D_{total}$
* **Aha!** 这一步的移动距离**也是总距离的 1/5**。
* **剩余距离**: $P - x^{(2)} = (1 - \frac{1}{4})(P - x^{(1)}) = \frac{3}{4} \cdot (\frac{4}{5} D_{total}) = \frac{3}{5} D_{total}$。

**Step i=2:**
* $\alpha_2 = \frac{1}{5 - 2} = \frac{1}{3}$
* $x^{(3)} = x^{(2)} + \frac{1}{3}(P - x^{(2)})$
* **含义**: 我们移动了**剩余距离的 1/3**。
* **这一步移动了多远？**
    * $\text{Move} = \frac{1}{3}(P - x^{(2)}) = \frac{1}{3} \left( \frac{3}{5} D_{total} \right) = \frac{1}{5} D_{total}$
* **又是一个 1/5！**

**结论（理想情况）：**
这个 $\alpha_i = \frac{1}{n - i}$ 的公式，**是一种精妙的数学构造，它确保了在每一步中，我们都前进 $\frac{1}{n}$ 的恒定距离**。
#### “现实情况”

在真实的代码中，我们的假设 $p^{(i)} = P$ 并不成立。
* `net` 不是完美的。
* `pred = net(x)` 的预测结果会**随着 $x$ 变得更清晰而变得更准确**。

所以，我们的“目标” $P$ 实际上是一个**移动目标 $p^{(i)}$**。

* **Step i=0:** `x` 是纯噪声。`pred` 可能是个模糊的猜测 $p^{(0)}$。我们朝着这个模糊的猜测移动 $1/n$。
* **Step i=1:** `x` 现在是 "9/10 噪声 + 1/10 猜测"。`net` 看到这个，给出了一个*更好*的猜测 $p^{(1)}$。我们再朝着这个*新的、更好的*目标移动 $1/n$。
* **Step i=2:** `x` 更清晰了。`net` 给出*更更*好的猜测 $p^{(2)}$。我们再朝 $p^{(2)}$ 移动 $1/n$。

所以说，这个公式在数学上是一个**迭代修正的线性插值方案**。它被设计为在 $n$ 步内，每一步都前进 $1/n$ 的距离。在每一步，它都会**重新评估目标**（通过 `pred = net(x)`），并朝着这个**最新、最准的**目标前进一小步。

最后一步 $i = n-1$ 时，$\alpha = 1$，`x = x * 0 + pred * 1`，这意味着我们**完全相信**模型在看了 $n-1$ 步的半成品后给出的最终预测。


### Improvements over our mini UNet

DDPM的UNet相较于这个UNet改进了数处：

- GroupNorm applies group normalization to the inputs of each block
- Dropout layers for smoother training
- Multiple resnet layers per block (if layers_per_block isn’t set to 1)
- Attention (usually used only at lower resolution blocks)
- Conditioning on the timestep.
- Downsampling and upsampling blocks with learnable parameters

```python
model = UNet2DModel(
    sample_size=28,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
print(model)
```

### DDPM Forward Diffusion Process

DDPM 的前向加噪过程 (Forward Process)在数学上被定义为一个**马尔可夫链（Markov Chain）**。其核心思想是从一个原始的、清晰的数据（如一张图片） $\mathbf{x}_0$ 出发，通过 $T$ 个离散的时间步（timesteps），在每一步都添加少量的高斯噪声，最终将 $\mathbf{x}_0$ 变为一个纯粹的、无规律的**标准高斯噪声** $\mathbf{x}_T$。

整个过程由一个固定的、预先设定的**方差表（variance schedule）** $\{\beta_t\}_{t=1}^T$ 来控制。

---

#### 1. 核心定义：单步转移（Markov Kernel）

你给出的第一个公式定义了这个马尔可夫链的**转移核（transition kernel）**，即如何从 $t-1$ 时刻的状态 $\mathbf{x}_{t-1}$ 转移到 $t$ 时刻的状态 $\mathbf{x}_t$：

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

我们来拆解这个公式：

* **$q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$**：这是一个条件概率分布。它描述了“在给定 $\mathbf{x}_{t-1}$ 的条件下，$\mathbf{x}_t$ 的概率分布是什么？”
* **$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$**：这是高斯分布（正态分布）的标准记法。$\mathbf{x}$ 是变量，$\boldsymbol{\mu}$ 是均值向量，$\boldsymbol{\Sigma}$ 是协方差矩阵。
* **均值 $\boldsymbol{\mu} = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}$**：
    * $\beta_t$ 是一个在 $(0, 1)$ 之间的小值（例如 $10^{-4}$）。
    * 因此 $\sqrt{1 - \beta_t}$ 是一个略小于 1 的标量。
    * 这意味着新的状态 $\mathbf{x}_t$ 的均值，是**将上一个状态 $\mathbf{x}_{t-1}$ 的信号进行轻微“衰减”（scale down）**。
* **协方差 $\boldsymbol{\Sigma} = \beta_t\mathbf{I}$**：
    * $\mathbf{I}$ 是单位矩阵。
    * 这表示我们添加的噪声在所有维度上是**独立同分布**的。
    * $\beta_t$ 就是这个噪声的**方差（variance）**。$\beta_t$ 越大，添加的噪声就越多。
    * （注意：$\sqrt{\beta_t}$ 才是标准差）。

#### 关键特性：重参数化技巧（Reparameterization Trick）

上面的分布定义了“是什么”，但没有告诉我们“怎么算”。为了从 $\mathbf{x}_{t-1}$ 实际*采样*（计算）出 $\mathbf{x}_t$，我们使用**重参数化技巧**。

一个从 $\mathcal{N}(\boldsymbol{\mu}, \sigma^2\mathbf{I})$ 采样的变量 $\mathbf{z}$，可以被写成：
$\mathbf{z} = \boldsymbol{\mu} + \sigma \boldsymbol{\epsilon}$，其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是一个标准高斯噪声。

将这个技巧应用于我们的单步转移公式：

$$
\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}
\quad \text{其中 } \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

这个等式非常直观地展示了文本所说的：“我们取 $\mathbf{x}_{t-1}$，将它按 $\sqrt{1 - \beta_t}$ 缩放，然后添加按 $\sqrt{\beta_t}$ 缩放的噪声。”

---

#### 2. 关键推导：“一步到位”的采样公式

如文本所述，我们不希望通过 $t$ 次迭代来计算 $\mathbf{x}_t$。我们希望有一个**封闭解（closed-form solution）**，可以直接从 $\mathbf{x}_0$ 采样 $\mathbf{x}_t$。这就是你给出的第二个公式，现在我们来推导它。

**首先，引入新符号（这纯粹是为了代数上的方便）：**
* 令 $\alpha_t = 1 - \beta_t$
* 令 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i = \alpha_1 \times \alpha_2 \times \dots \times \alpha_t$

$\bar{\alpha}_t$ 被称为“累积乘积”（cumulative product）。它代表了从 $\mathbf{x}_0$ 开始，信号强度**总共被衰减了多少**。

**开始推导：**

我们从 $\mathbf{x}_t$ 的重参数化公式开始，并递归地展开它：

$\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$
*(其中 $\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$)*

现在，我们将 $\mathbf{x}_{t-1}$ 也展开：
$\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2}$
*(其中 $\boldsymbol{\epsilon}_{t-2} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$)*

**将 (t-1) 式代入 (t) 式：**

$\mathbf{x}_t = \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$

$\mathbf{x}_t = \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1}$

**核心数学性质：高斯分布的可加性**
推导的关键在于，**两个独立高斯分布的和仍然是一个高斯分布**。
如果 $Z_1 \sim \mathcal{N}(0, \sigma_1^2 \mathbf{I})$ 且 $Z_2 \sim \mathcal{N}(0, \sigma_2^2 \mathbf{I})$，那么：
$Z_1 + Z_2 \sim \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2) \mathbf{I})$

在我们的展开式中，后两项都是均值为 0 的独立高斯噪声。我们可以把它们合并：
* 噪声1的方差：$(\sqrt{\alpha_t (1 - \alpha_{t-1})})^2 = \alpha_t (1 - \alpha_{t-1})$
* 噪声2的方差：$(\sqrt{1 - \alpha_t})^2 = 1 - \alpha_t$
* **合并后总方差**：
    $\sigma_{\text{total}}^2 = \alpha_t (1 - \alpha_{t-1}) + (1 - \alpha_t)$
    $\sigma_{\text{total}}^2 = \alpha_t - \alpha_t \alpha_{t-1} + 1 - \alpha_t$
    $\sigma_{\text{total}}^2 = 1 - \alpha_t \alpha_{t-1}$

利用 $\bar{\alpha}_t$ 的定义（$\bar{\alpha}_2 = \alpha_1 \alpha_2$）：
* $\sqrt{\alpha_t \alpha_{t-1}} = \sqrt{\bar{\alpha}_t / \bar{\alpha}_{t-2}}$ （如果从 $t=2$ 开始）
* $1 - \alpha_t \alpha_{t-1} = 1 - \bar{\alpha}_t / \bar{\alpha}_{t-2}$ （如果从 $t=2$ 开始）
    *注：这里用 $\bar{\alpha}_t = \alpha_t \alpha_{t-1}$ 来表示 $t=2$ 时的 $\bar{\alpha}_2$ 更清晰。*

我们令 $t=2$ 来看这个模式：
$\mathbf{x}_2 = \sqrt{\alpha_2 \alpha_1} \mathbf{x}_0 + \sqrt{1 - \alpha_2 \alpha_1} \bar{\boldsymbol{\epsilon}}_1$
*(其中 $\bar{\boldsymbol{\epsilon}}_1$ 是一个新的 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 噪声)*

**推广到任意 $t$：**
通过数学归纳法，我们可以证明这个模式会一直持续下去：

$\mathbf{x}_t = \sqrt{\alpha_t \alpha_{t-1} \dots \alpha_1} \mathbf{x}_0 + \sqrt{1 - \alpha_t \alpha_{t-1} \dots \alpha_1} \boldsymbol{\epsilon}$
*(其中 $\boldsymbol{\epsilon}$ 是所有 $\boldsymbol{\epsilon}_0, \dots, \boldsymbol{\epsilon}_{t-1}$ 合并后的新 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 噪声)*

**使用我们的 $\bar{\alpha}_t$ 符号替换：**

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}
\quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

**转换回概率分布的记法：**
这个等式在数学上等价于定义了 $q(\mathbf{x}_t \vert \mathbf{x}_0)$ 的分布：
* 均值 $\boldsymbol{\mu} = \sqrt{\bar{\alpha}_t} \mathbf{x}_0$
* 协方差 $\boldsymbol{\Sigma} = (\sqrt{1 - \bar{\alpha}_t})^2 \mathbf{I} = (1 - \bar{\alpha}_t) \mathbf{I}$

所以，我们得到了你给出的第二个公式：

$$
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$



### Differences in training objective

预测噪声和预测成品是信息等价 (informationally equivalent)的。但是，**预测噪声（$\boldsymbol{\epsilon}$）是一个比预测成品图像（$\mathbf{x}_0$）更简单、更稳定的机器学习任务。**

---

#### 1. 为什么说这两个任务是“等价”的？

让我们回到“一步到位”的加噪公式：
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}
$$

在这个公式中：
* $\mathbf{x}_t$ 是神经网络的**输入**（加了 $t$ 步噪声的图）。
* $t$ 是神经网络的**输入**（告诉它现在是第几步）。
* $\mathbf{x}_0$ 是**原始图像**（我们想要的成品）。
* $\boldsymbol{\epsilon}$ 是被添加的**标准高斯噪声**。
* $\bar{\alpha}_t$ 是由 Noise Schedule 决定的**已知常数**。

神经网络的任务是接收 $\mathbf{x}_t$ 和 $t$，然后预测出一些有用的东西。

* **选项A：预测成品图像 $\mathbf{x}_0$**
    如果模型 $\text{model}(\mathbf{x}_t, t)$ 预测出了 $\mathbf{\hat{x}}_0$（对 $\mathbf{x}_0$ 的预测），我们可以用上面的公式反推出噪声：
    $\mathbf{\hat{\epsilon}} = (\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{\hat{x}}_0) / \sqrt{1 - \bar{\alpha}_t}$

* **选项B：预测噪声 $\boldsymbol{\epsilon}$** (DDPM的选择)
    如果模型 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 预测出了 $\mathbf{\hat{\epsilon}}$（对 $\boldsymbol{\epsilon}$ 的预测），我们同样可以反推出原始图像：
    $\mathbf{\hat{x}}_0 = (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \mathbf{\hat{\epsilon}}) / \sqrt{\bar{\alpha}_t}$

**结论：** 从数学上讲，只要模型能准确预测出 $\mathbf{x}_0$ 或 $\boldsymbol{\epsilon}$ 中的**任何一个**，我们就能 100% 准确地计算出另一个。

那么，真正的问题是：**让神经网络去学习哪一个目标（Target）更容易？**

---

#### 2. 为什么预测噪声 ($\boldsymbol{\epsilon}$) 更好？

DDPM 论文的作者发现，“预测噪声”是一个**条件更良好 (better-conditioned)** 的任务。

关键在于**考虑 $t$ 变得非常大（例如 $t \to T$）时的极端情况**。

当 $t$ 非常大时（比如 $t=950$），Noise Schedule 会使得 $\bar{\alpha}_t \approx 0$。
此时，加噪公式变为：
$$
\mathbf{x}_t \approx \sqrt{0} \cdot \mathbf{x}_0 + \sqrt{1 - 0} \cdot \boldsymbol{\epsilon} \quad \implies \quad \mathbf{x}_t \approx \boldsymbol{\epsilon}
$$
这意味着，在接近最后的时间步，$t$ 时刻的输入 $\mathbf{x}_t$ **本身就几乎是纯噪声**。

现在我们来比较两个选项：

##### 选项A：预测 $\mathbf{x}_0$ （困难模式）

* **输入 (Input):** $\mathbf{x}_t$ (一张几乎纯白的雪花噪声图)。
* **目标 (Target):** $\mathbf{x}_0$ (一张结构复杂的《蒙娜丽莎》)。

**这几乎是一个不可能完成的任务！** 就像我给你一张完全随机的雪花屏，问你：“这张图原本是《星空》还是《向日葵》？”
输入 $\mathbf{x}_t$ 和目标 $\mathbf{x}_0$ 之间几乎没有任何关联。神经网络会非常困惑，训练很难收敛。

##### 选项B：预测 $\boldsymbol{\epsilon}$ （简单模式）

* **输入 (Input):** $\mathbf{x}_t$ (一张几乎纯白的雪花噪声图，$\mathbf{x}_t \approx \boldsymbol{\epsilon}$)。
* **目标 (Target):** $\boldsymbol{\epsilon}$ (被添加的噪声本身)。

**这个任务就变得非常简单！** 神经网络的输入和它要预测的目标**几乎是同一个东西**！
模型可以轻松学到：**“哦，当 $t$ 很大的时候，我只需要把我看到的输入 $\mathbf{x}_t$ 直接当作 $\boldsymbol{\epsilon}$ 输出就行了。”**

---

#### 3. 训练稳定性和损失函数

这个选择也简化了损失函数的设计。

DDPM 的作者们发现，如果选择“预测噪声”，他们可以**使用一个非常简单的损失函数**：
$$
L_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ || \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) ||^2 \right]
$$
这个公式的意思是：
1.  随机选一张图 $\mathbf{x}_0$ 和一个噪声 $\boldsymbol{\epsilon}$。
2.  随机选一个时间 $t$。
3.  用 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$ 算出 $\mathbf{x}_t$。
4.  让模型 $\boldsymbol{\epsilon}_\theta$ 去看 $\mathbf{x}_t$ 和 $t$，并预测一个 $\mathbf{\hat{\epsilon}}$。
5.  计算**模型预测的噪声 $\mathbf{\hat{\epsilon}}$** 和**我们真正放进去的噪声 $\boldsymbol{\epsilon}$** 之间的**均方误差（L2 损失）**。

这个简单的损失函数，被证明在效果上等同于一个对 $\mathbf{x}_0$ 预测任务**进行了复杂加权**的损失函数。它会**自动地（隐式地）更加关注那些噪声多的、更困难的时间步**，而这正是训练出高质量模型的关键。

### Timestep conditioning

“Timestep Conditioning” (时间步条件化) 是 DDPM 成功的关键，用于告诉神经网络（U-Net）现在正在处理第几步（$t$）的噪声。
- **$t$ 很小 (如 $t=10$)：** 任务是“**轻微去噪**”（输入几乎是原图）。  
- **$t$ 很大 (如 $t=114514$)：** 任务是“**生成内容**”（输入几乎是纯噪声）。
用于区分不同情况下的行为设计了Timestep Conditioning.

### Sampling approaches

“Sampling”（采样）就是**生成图像**的过程，也就是我们常说的“跑图”。

它是在模型**训练完成之后**，我们实际**使用**模型来创造新东西的步骤。

这个过程与训练时的“前向加噪”相反，它是一个 **“反向去噪” (Reverse Process)** 的过程。

**核心思想：**
1.  **开始：** 从一张**纯高斯噪声**图像 $\mathbf{x}_T$ 出发。
2.  **迭代：** 使用我们训练好的**噪声预测模型** $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$，一步一步地（例如从 $t=1000$ 减到 $t=1$）把噪声去掉。
3.  **结束：** 最终得到一张清晰的图像 $\mathbf{x}_0$。

不同的 “Sampling Approaches”（采样方法或采样器）就是实现这个“反向去噪”过程的**不同策略**。它们的主要区别在于**速度**、**随机性**和**质量**。

以下是最关键的两种采样方法：

---

### 1. DDPM (Ancestral Sampling) - 祖先采样

这是 DDPM 论文**原始**的采样方法，通常被称为“祖先采样”(Ancestral Sampling)。

计算 $\mathbf{x}_{t-1}$ 的公式：$$  \mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta \right)$$
* **工作原理：**
    它严格地 **“倒放”** 我们在训练时学习的前向加噪过程。
    它假设从 $\mathbf{x}_t$ 到 $\mathbf{x}_{t-1}$ 也是一个马尔可夫过程。
    在每一步 $t$：
    1.  模型预测出噪声 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$。
    2.  用这个噪声估算出 $\mathbf{x}_{t-1}$ 的**均值（方向）**。
    3.  在这个均值的基础上，**再加上一点新的随机高斯噪声**。

* **为什么叫“祖先采样”？**
    因为它是一个**随机过程 (Stochastic Process)**。每一步都会引入**新的随机性**（新噪声）。这意味着即使你从同一个 $\mathbf{x}_T$ 出发，只要中间任何一步 $t$ 新加的噪声不同，最终得到的 $\mathbf{x}_0$ 也会不同。每一步的 $\mathbf{x}_t$ 都是 $\mathbf{x}_{t-1}$ 的“祖先”。

* **优点：**
    * **高保真度：** 生成的图像质量非常高，多样性好。
    * **理论扎实：** 完美对应了训练时的概率模型。

* **缺点：**
    * **极慢！** 它**必须**走完训练时的**所有步骤**（例如 $T=1000$ 步）。如果训练用了1000步，采样也必须用1000步，一步都不能少。这导致生成一张图可能需要几十秒甚至几分钟。

##### 什么（DDPM）采样要加一点新噪声？

因为“反向去噪”这个步骤在数学上不是一个“确定的点”，而是一个“概率分布”。加噪声，就是在从这个概率分布中进行一次“抽样”。

**详细解释：**

1.  **模型的工作：** 在第 $t$ 步，模型 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 预测出了噪声 $\boldsymbol{\epsilon}_\theta$。
2.  **去噪（均值）：** 利用这个预测的噪声，我们可以计算出一个“**最佳猜测**”的 $\mathbf{x}_{t-1}$ 应该是什么样子。这个“最佳猜测”就是高斯分布的**均值 $\boldsymbol{\mu}_\theta$**（也就是您手写的公式 5）。
3.  **不确定性（方差）：** DDPM 的理论证明，即使在 $\mathbf{x}_t$ 给定的情况下，能产生它的 $\mathbf{x}_{t-1}$ 也**不是唯一的**，而是在这个“最佳猜测”（均值）附近的一个**很小的高斯分布**。这个分布是有**方差**（variance）的（您手写的公式 6，$\sigma_t^2$）。
4.  **“加噪声”的真正含义：**
    为了**严格遵守**这个数学模型，我们不能只取那个“最佳猜测”的均值 $\boldsymbol{\mu}_\theta$ 作为下一步的结果。我们必须从这个“均值为 $\boldsymbol{\mu}_\theta$、方差为 $\sigma_t^2$”的完整高斯分布中**随机抽取一个样本**。
    
    如何从 $\mathcal{N}(\boldsymbol{\mu}, \sigma^2)$ 中抽样？
    答案就是：**$\text{样本} = \boldsymbol{\mu} + \sigma \times \mathbf{z}$** （其中 $\mathbf{z}$ 是一个标准高斯噪声 $\mathcal{N}(\mathbf{0}, \mathbf{I})$）
    
    这就是您手写的最后一步：
    $\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{z}$
    
    **所以，“加一点新噪声 $\mathbf{z}$”这个动作，就是在数学上完成这个“抽样”步骤。**

**这么做的好处：**

1.  **忠于理论：** 这是对反向概率分布的忠实模拟。
2.  **增加多样性：** 在1000步的每一步都加入一点点新的随机性，这保证了**即使你从同一个 $\mathbf{x}_T$ 出发，每次采样（跑图）也能得到不一样的 $\mathbf{x}_0$ 结果**。这对生成模型来说是至关重要的。
3.  **纠错能力：** 这种随机的“抖动”可以帮助模型在生成过程中“跳出”一个错误的局部最优解（比如模型在某一步画崩了），在下一步有机会修正回来。

**重要补充（DDIM）：**
后来的 **DDIM** 采样器发现，这个新加的噪声 $\mathbf{z}$ **其实可以被设为 0**（即 $\eta=0$）。当不加新噪声时，采样就变成了一个**“确定性”**（Deterministic）的过程，不仅速度快（可以跳步），而且同一个开局噪声（seed）永远产生同一张图。

* **DDPM 加噪声：** 随机采样（Stochastic），慢，多样性高。
* **DDIM 不加噪声：** 确定性采样（Deterministic），快，多样性低（但保真度高）。

---

### 2. DDIM (Denoising Diffusion Implicit Models)

DDIM 是对 DDPM 采样的**第一个重大改进**，也是目前（包括 Stable Diffusion 在内）绝大多数模型**默认使用**的采样器（或其变体）的基础。

* **算法：** DDIM 的核心思想是，每一步都先“**预测最终的 $\mathbf{\hat{x}}_0$**”，然后再用这个 $\mathbf{\hat{x}}_0$ 来计算 $\mathbf{x}_{t-1}$。
* **计算 $\mathbf{x}_{t-1}$ 的公式：**
    1.  先用模型预测的 $\boldsymbol{\epsilon}_\theta$ 算出“最终成品”的预测值：   $$\mathbf{\hat{x}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta)$$
    2.  然后用一个**完全不同**的公式来计算 $\mathbf{x}_{t-1}$（当 $\eta=0$ 时）：
        $$
        \mathbf{x}_{t-1} = \underbrace{\sqrt{\bar{\alpha}_{t-1}} \mathbf{\hat{x}}_0}_{\text{“最终成品”的信号}} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1}} \cdot \boldsymbol{\epsilon}_\theta}_{\text{“指向 } \mathbf{x}_t \text{”的噪声}}
        $$

* **工作原理：**
    DDIM 的作者发现了一个数学上的“捷径”。他们重新推导了反向过程，并提出了一个**更通用**的公式。
    这个新公式最关键的一点是：它**允许“跳步” (Jumping steps)**。

    它不再是一个马尔可夫过程（每一步都依赖上一步），而是一个**非马尔可夫 (non-Markovian)** 过程。

* **优点：**
    * **极快！** DDIM 最大的贡献。我们不再需要 $T=1000$ 步。我们可以只采样其中的一小部分（例如 $50$ 步，甚至 $20$ 步），直接从 $t=1000 \to t=950 \to t=900 \to \dots \to t=0$ 这样“大步快走”。这使得采样速度提升了 **10 到 50 倍**。
    * **可控的随机性 (Controllable Stochasticity)：** DDIM 引入了一个参数 $\eta$ (eta)：
        * **$\eta = 1$ (DDPM 模式)：** 采样过程与 DDPM 完全一样，每一步都添加随机噪声（随机过程）。
        * **$\eta = 0$ (DDIM 模式)：** 采样过程**完全不添加**任何新的随机噪声。它变成了一个**确定性过程 (Deterministic Process)**。

* **“确定性过程”($\eta=0$) 的惊人特性：**
    如果你使用**同一个**初始噪声 $\mathbf{x}_T$（同一个 seed）和**同样**的提示词 (prompt)，DDIM ($\eta=0$) **每一次**生成的 $\mathbf{x}_0$ 都会是**一模一样**的。
    这还带来一个额外好处：你可以对 $\mathbf{x}_T$ 这个“潜空间”进行插值，实现平滑的图像过渡。



