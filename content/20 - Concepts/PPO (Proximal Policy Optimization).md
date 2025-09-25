---
{"publish":true,"created":"2025-09-19T15:47:06.491-04:00","modified":"2025-09-24T23:30:08.515-04:00","tags":["ai","ppo","rl","blog"],"cssclasses":""}
---


## 论文公式解释：
 
所有公式号码的标注与原论文相同。
 
### Background: Policy Optimization
 
#### Policy Gradient Methods

我们先看公式2。

##### 公式 (2): 策略梯度目标函数 (The Policy Gradient Objective Function)

$$L^{PG}(\theta) = \hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t | s_t) \hat{A}_t \right] \quad (2)$$

$L^{PG}$ 即 **目标函数** , 需要 **最大化**。在论文中被称为**Loss**，是指最小化其负值。

让我们把它拆解成几个部分来理解：

1.  **$\pi_\theta(a_t | s_t)$：策略 (Policy)**
    * 这代表智能体的“策略”或“大脑”。
    * $s_t$ 是在时间步 $t$ 时，智能体所处的**状态 (state)**，也就是它观察到的环境。
    * $a_t$ 是智能体在状态 $s_t$ 下选择采取的**动作 (action)**。PPO
    * $\theta$ 是策略网络的参数（比如神经网络的权重）。我们的目标就是通过训练来优化这些参数 $\theta$。即`weight`
    * 整个 $\pi_\theta(a_t | s_t)$ 表示的是：在给定状态 $s_t$ 的情况下，采取动作 $a_t$ 的**概率**。离散动作空间里是概率列表，连续动作空间里是概率函数（PDF）

2.  **$\log \pi_\theta(a_t | s_t)$：动作概率的对数 (Log-Probability of the Action)**
    * 在数学和计算上，直接优化概率的乘积很困难，所以我们通常取其对数，将乘法问题转化为加法问题，这样更稳定且更容易求导。
    * 你可以简单地把它理解为对我们所采取的那个动作的“倾向性”的打分。

3.  **$\hat{A}_t$：优势函数估计值 (The Advantage Function Estimator)**
    * “帽子” `^` 表示它是一个**估计值**
    * 这是整个公式中最关键的部分之一。$\hat{A}_t$ 衡量的是：在状态 $s_t$ 下，采取动作 $a_t$ **比平均水平好多少**。
    * **如果 $\hat{A}_t > 0$**：意味着动作 $a_t$ 是一个**好于平均**的动作，它带来了比预期更好的结果。
    * **如果 $\hat{A}_t < 0$**：意味着动作 $a_t$ 是一个**差于平均**的动作，它带来了比预期更差的结果。

4.  **$\hat{\mathbb{E}}_t[\dots]$：经验平均 (Empirical Average)**
    * 在整个batch中按时间步对期望进行平均来作为对策略表现实际情况的估计。


注意：
神经网络的输出如下：

| 特性                     | 离散动作空间 (Discrete)  | 连续动作空间 (Continuous)       |
| :--------------------- | :----------------- | :------------------------ |
| **任务**                 | 从 **有限个选项** 中选择一个  | 从 **一个区间** 内选择一个值         |
| **网络输出**               | Logits (每个选项一个分数)  | 分布的参数 (如均值 `μ` 和标准差 `σ`)  |
| **最终结果**               | 一个描述各选项概率的**分类分布** | 一个描述整个区间的**概率密度函数 (PDF)** |
| **转换成$\pi_\theta$的工具** | **Softmax 函数**     | **概率分布** (如高斯分布)          |
|                        |                    |                           |

而公式1不过是对公式2的求梯度。

##### 公式 (1): 策略梯度估计 (The Policy Gradient Estimator)

$$\hat{g} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t \right] \quad (1)$$

这个公式 $\hat{g}$ 表示**目标函数 $L^{PG}(\theta)$ 的梯度 (gradient)**。

* **$\nabla_\theta$** 是梯度算子，表示对参数 $\theta$ 求导。
* 梯度 $\hat{g}$ 本质上是一个向量，它指向了能使 $L^{PG}(\theta)$ **增长最快**的方向。

在实际训练中，我们遵循一个叫做**梯度上升** 的算法：
1.  使用当前的策略 $\pi_\theta$ 与环境互动，收集一批数据（状态、动作、奖励等）。
2.  利用这些数据计算出优势函数估计值 $\hat{A}_t$。
3.  根据公式 (1) 计算出梯度 $\hat{g}$。
4.  沿着梯度的方向更新参数：$\theta \leftarrow \theta + \alpha \hat{g}$ （其中 $\alpha$ 是学习率）。

论文中提到了这种算法的缺陷在于，一个batch只能进行一个epoch的更新，数据利用率极低；这是因为策略的更新不受任何约束。为了解决这个问题，Trust Region Methods被提了出来。


#### Trust Region Methods

TRPO (Trust Region Policy Optimization) 是对Policy Gradient Method限制步长以求稳定的产物：

##### 公式 (3) 和 (4): 带约束的优化 (Constrained Optimization)

这两个公式必须放在一起看，因为它们共同定义了一个**带约束的优化问题**。

$$
\begin{align*}
\text{maximize}_\theta \quad & \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} \hat{A}_t \right] \quad &(3) \\
\text{subject to} \quad & \hat{\mathbb{E}}_t \left[ \text{KL}[\pi_{\theta_{old}}(\cdot | s_t), \pi_\theta(\cdot | s_t)] \right] \le \delta \quad &(4)
\end{align*}
$$

##### 公式 (3): 目标函数 (The Objective)

这是我们要**最大化**的目标。

* **$\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$**: **重要性采样比率 (importance sampling ratio)**。
    * $\pi_\theta$ 是我们想要优化的**新策略**。
    * $\pi_{\theta_{old}}$ 是我们用来收集这批训练数据的**旧策略**。
    * 这个比率衡量的是：对于同一个状态和动作，新策略选择该动作的可能性是旧策略的多少倍。这是一种“校正因子”，它允许我们用旧策略的数据来评估新策略的好坏（这被称为"Off-Policy" / "离策略"方法）。

* **$\hat{A}_t$**: 优势函数估计，和之前一样，衡量动作 $a_t$ 的好坏。

**公式 (3) 的整体含义**:
我们希望调整新策略 $\pi_\theta$，使得对于那些有**正优势**（$\hat{A}_t > 0$）的动作，其重要性采样比率尽可能**变大**（即新策略更倾向于选择这些好动作）；对于有**负优势**（$\hat{A}_t < 0$）的动作，其比率尽可能**变小**。这和我们之前讨论的策略梯度思想是一致的。

#### 公式 (4): 约束条件 (The Constraint)

这是 TRPO 实现“安全更新”的关键，它为优化设置了一个**“安全边界”**。

* **$\text{KL}[\pi_{\theta_{old}}, \pi_\theta]$**: KL散度，用新策略 $\pi_\theta$ 去近似旧策略 $\pi_{\theta_{old}}$ 时的损失。
* **$\le \delta$**: 我们**强制**要求，新旧策略之间的平均 KL 散度（即策略变化的幅度）**不能超过**一个很小的常数 $\delta$。
这个**$\le \delta$就是所谓trust region，防止过大幅度的优化。


##### 公式 (5): 带惩罚的优化 (Penalized Optimization)

(5)是达成同一目标的另一种并不等价的实现：

$$\text{maximize}_\theta \quad \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} \hat{A}_t - \beta \, \text{KL}[\pi_{\theta_{old}}(\cdot | s_t), \pi_\theta(\cdot | s_t)] \right] \quad (5)$$

这个公式把约束项直接整合进了目标函数里。

* **第一部分**: $\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} \hat{A}_t$ 和公式(3)完全一样，是我们的主要优化目标。
* **第二部分**: $- \beta \, \text{KL}[\dots]$ 是一个**惩罚项 (penalty term)**。
    * 如果新旧策略的差异（KL散度）变大，这个惩罚项的绝对值也会变大。
    * 由于它前面是**负号**，一个大的惩罚项会拉低整个目标函数的总分。
    * $\beta$ 是一个超参数，用来控制我们对这个惩罚的重视程度。

**公式 (5) 的工作方式**:
优化器现在需要做一个**权衡 (trade-off)**。它一方面想最大化第一部分（获得更高收益），但同时它又必须避免让第二部分（惩罚项）变得太大。如果策略更新得太激进，KL散度就会剧增，导致总分下降。因此，优化器会自动寻找一个既能提升策略、又不会离旧策略太远的“甜点区”。

但是事实是，在不同问题甚至同一问题的不同参数都表现良好的$\beta$ 值几乎**不可能找到**。所以TRPO仍然选择了**公式(3)+(4)的约束方法**。但是TRPO 的约束方法在计算上非常**复杂和昂贵**（需要用到“共轭梯度”等二阶优化方法）。因此PPO提出了一种全新的、更简单的目标函数（带有裁剪`clip`），既能达到 TRPO 这种限制更新幅度的效果，又比 TRPO 更简单、更容易实现，计算效率也更高。

### Clipped Surrogate Objective

好的，这些公式是 PPO 算法的核心与精髓。它们在 TRPO 的思想上进行了简化和改进，从而变得更加实用。

我们来一步步解析它们。

-----

##### 公式 (6): 一切的起点 ：保守策略迭代 (Conservative Policy Iteration Objective)

$$L^{CPI}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} \hat{A}_t \right] \quad (6)$$

  * 这个公式称为“保守策略迭代 (Conservative Policy Iteration)”的目标函数。请注意，它和我们之前讨论的TRPO 的目标函数（公式3） **是完全一样的**。
  * 它仍然使用重要性采样比率 $r_t(\theta) = \frac{\pi\_\theta(a_t | s_t)}{\pi\_{\theta_{old}}(a_t | s_t)}$ 来修正新旧策略的差异。
  * 它的目标依然是：对于优势 $\hat{A}_t$ 为正的动作，增大其概率比率 $r_t(\theta)$；对于优势为负的动作，减小其概率比率。

而用于替代公式(4)，PPO 提出了一个更简单、更巧妙的公式 (7)。

-----

##### 公式 (7): PPO 的核心创新 (The Clipped Surrogate Objective)

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right] \quad (7)$$

这个公式就是 PPO 算法的灵魂。它通过一个非常聪明的“裁剪 (clipping)”技巧，用一种简单得多的方法实现了 TRPO 的“信任区域”思想。

让我们把它拆开来看：

  * **`min(...)`**: 最终的目标值是在两个计算结果中取**较小**的那一个。
  * **第一项: $r_t(\theta)\hat{A}_t$**: 这就是我们原始的、未经任何修改的目标，和公式(6)一样。
  * **第二项: $clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t$**: 这是被“裁剪”过的目标。
      * $clip(r_t(\theta), 1-\epsilon, 1+\epsilon)$函数会把概率比率 $r_t$ 强制限制在一个 $[1-\epsilon, 1+\epsilon]$ 的区间内。$\epsilon$ 是一个很小的超参数（比如0.2，那么这个区间就是 $[0.8, 1.2]$）。

此处论文中提供了两张插图。


![[99 - Attachments/images/schulman2017proximal-fig1.png]]

Figure 1 展示了 **“裁剪 (clipping)”** 是如何在一个时间步上工作的。
- **x轴 ($r$)**: 新旧策略的概率比率。$r\>1$ 意味着新策略更倾向于采取该动作。$r<1$ 意味着更不倾向。
-  **y轴 ($L^{CLIP}$)**: 最终的目标得分。
函数的目的是求图片中的最高点。

![[99 - Attachments/images/schulman2017proximal-fig2.png]]

这张图从一个更宏观的视角展示了 PPO 的一次完整更新过程。
  * **x轴 (Linear interpolation factor)**: 线性插值因子。$0$ 代表更新前的旧策略 $\theta_{old}$，$1$ 代表 PPO 算法优化后找到的新策略 $\theta$。其实$x<0$或$x>1$的部分没有什么意义。
  * **y轴**: 各种目标函数的值。

**图中各曲线的含义**:
  * **蓝色 (KL散度)**: 显示了新策略与旧策略的“距离”。从0开始，随着更新的进行而增加，在最终更新点（x=1）达到约 `0.02`，这是一个很小的、受控的变化。
  * **橙色 ($L^{CPI}$)**: 未经任何限制的原始目标。你可以看到，它在 `x=1` 之后还在继续飙升。如果没有限制，优化器会“贪婪地”冲向更高的目标，导致策略更新过大，KL散度也会变得非常大。
  * **绿色 (裁剪项)**: 被裁剪后的目标函数项。
  * **红色 ($L^{CLIP}$)**: **PPO 的最终目标函数**，它是$L^{CPI}$ **（橙色曲线）** 和  **裁剪项（绿色曲线）** 中的**较小值**。注意红色比绿色曲线还要更低，这是因为 “先取最小值，再求期望（平均）”的结果，小于等于 “先求期望（平均），再取最小值”。即 $E[min(X,Y)]\leq min(E[X],E[Y])$（杰森不等式）。

### Algorithm

#### 公式 (9): 最终的完整目标函数

$$L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb{E}}_t \left[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t) \right] \quad (9)$$

这个公式是 PPO 在训练时实际使用的**总目标（或总损失）**。它由三个部分加权组成，共同优化。我们希望**最大化**这个总目标。

1.  **$L^{CLIP}_t(\theta)$ (策略提升项)**

      * 这就是我们之前详细讨论过的 PPO 核心的**裁剪替代目标函数 (公式7)**。
      * 它的作用是提升策略（Actor），让“好”动作的概率增加，“坏”动作的概率减小，同时通过裁剪来保证更新的稳定性。

2.  **$- c_1 L^{VF}_t(\theta)$ (价值函数误差项)**

      * PPO 通常采用 **Actor-Critic (演员-评论家)** 架构。除了“演员”（策略 $\pi$）外，还有一个“评论家”（价值函数 $V$），它的工作是**评估当前状态 `s` 有多好**。
      * $L^{VF}$ 是**评论家（Critic）的损失函数**，论文中是一个简单的均方误差：$L^{VF} = (V(s_t) - V_t^{targ})^2$。它衡量了评论家对状态价值的**预测值 $V(s_t)$** 和**实际值 $V_t^{targ}$**（通常用收集到的真实奖励来估算）之间的差距。
      * 我们希望这个误差**越小越好**，所以在总目标中，它前面有一个**负号**（最大化 $-L^{VF}$ 就等于最小化 $L^{VF}$）。
      * $c_1$ 是用来平衡策略提升和价值函数拟合这两个任务的重要性的系数，。

3.  **$+ c_2 S[\pi_\theta](s_t)$ (熵奖励项)**

      * **熵 (Entropy) ** 在这里衡量的是策略输出的**随机性或不确定性**。
      * 熵越高，策略越随机，探索性越强（比如对多个动作的输出概率都差不多）。
      * 熵越低，策略越确定，探索性越弱（比如对某个动作的输出概率接近100%）。
      * 我们在总目标中**加上**这一项，是为了**鼓励探索**。它奖励那些更随机的策略，防止策略过早地收敛到一个局部最优解而停止探索。
      * $c_2$是鼓励探索的强度的系数。

-----

### 公式 (10), (11), (12): 优势函数的计算 (GAE)

这三个公式展示了如何计算优势函数 $\hat{A}_t$。使用的算法是**广义优势估计 (Generalized Advantage Estimation, GAE)**。

#### 公式 (12): TD 误差 (Temporal-Difference Error)

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad (12)$$

  * 这是计算优势函数的基础。
  * $r_t + \gamma V(s_{t+1})$ 是对当前回报的**一步估计**（我们得到的即时奖励 $r_t$，加上对下一步状态价值的折扣估计 $\gamma V(s_{t+1}$）。
  * $\gamma$ 是**折扣因子**
  * $V(s_t)$ 是我们评论家对当前状态价值的**原始估计**。
  * $\delta_t$ 衡量了 **“现实”与“预期”之间的差距** 。如果 $\delta_t > 0$，说明事实结果这一步比预想的要好。你可以把它看作是**单步优势**。

#### 公式 (11): 广义优势估计 (GAE)

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t-1}\delta_{T-1} \quad (11)$$

  * GAE 把**单步优势**扩展到了**所有步**。
  * 它将未来所有的 TD 误差 $\delta$ 进行加权求和，来作为当前优势 $\hat{A}_t$ 的最终估计。
  * $\lambda$ (lambda) 是**时序差分误差**，用于在**偏差 (bias)** 和 **方差 (variance)** 之间做权衡。
      * 当 $\lambda=0$ 时, $\hat{A}_t = \delta_t$，退化成单步优势。（偏差高，方差低）。
      * 当 $\lambda=1$ 时（对应公式10的情况），退化成蒙特卡洛估计（不自举）。这种情况过度依赖采样的“运气”，即方差过高。关于GAE的详细讨论见[[20 - Concepts/Generalized advantage estimation (GAE) 广义优势估计]]
      * 通常取 $\lambda$ 为 0.95 ~ 0.99 可以在偏差和方差之间取得很好的平衡。

这里给出一个 $\gamma$ 和 $\lambda$ 的对比

| 特性       | `γ` (Gamma)             | `λ` (Lambda)                           |
| -------- | ----------------------- | -------------------------------------- |
| **名称**   | 折扣因子 (Discount Factor)  | GAE 参数 / 迹衰减参数 (Trace-Decay Parameter) |
| **作用对象** | 未来的**奖励 (Rewards)**     | 未来的**时序差分误差 (TD Errors)**              |
| **核心问题** | 未来的**奖励**有多重要？          | 如何平衡**估计**中的偏差和方差？                     |
| **影响**   | 定义智能体的**目标**（短视 vs. 远见） | 影响优势函数估计的**准确性**和**稳定性**               |
| **所属范畴** | 强化学习**问题定义**的一部分 (MDP)  | 强化学习**算法**的一部分 (例如 GAE                 |

省流即：$\gamma$ 是定义参数，调整 $\gamma$ 即调整对未来回报的重视程度；而降低 $\lambda$ 是为了防止方差过高。

-----

### 最终的伪代码 (Algorithm 1)

这是论文原文中提供的伪代码。这个伪代码将上面所有的部分整合成了 PPO 的完整训练流程。

```
for iteration=1, 2, ... do
    // ---- 1. 数据收集阶段 ----
    for actor=1, 2, ..., N do
        // 让当前的策略 π_θ_old 与环境互动 T 步，收集经验
        Run policy π_θ_old in environment for T timesteps
    end for
    
    // ---- 2. 优势计算阶段 ----
    // 用收集到的数据，通过公式(11)和(12)计算出每一步的优势函数估计值
    Compute advantage estimates Â₁, ..., Â_T

    // ---- 3. 优化学习阶段 ----
    // 用收集到的数据，反复优化总目标函数(公式9) K 个轮次 (epoch)
    Optimize surrogate L wrt θ, with K epochs and minibatch size M <= NT
    
    // 更新策略，为下一次迭代做准备
    θ_old ← θ
end for
```

## 实现

借用[[30 - Resources/Code Repos/CleanRL]]中的`ppo.py`，我们来看一下如何在三百行左右实现PPO算法

### 首先当然是import
```python
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
```

### Args




```python
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None  # type: ignore
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None  # type: ignore
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


```

### 超参数 (True Hyperparameters)

这些参数直接**控制学习算法的行为和性能**，是在训练开始前设置的，并且通常需要调整（调参）以获得最佳结果。

* `total_timesteps`: 训练总步数，决定了训练的充分程度。
* `learning_rate`: 学习率，控制模型参数更新的幅度。
* `num_envs`: 并行环境的数量，影响数据采集的效率和多样性。
* `num_steps`: 每个环境在一次策略更新前运行的步数，决定了 rollout 轨迹的长度。
* `anneal_lr`: 是否对学习率进行衰减，一种学习率调度策略。
* `gamma`: 折扣因子，决定了未来奖励的重要性。
* `gae_lambda`: GAE (Generalized Advantage Estimation) 的 $\lambda$ 参数，用于平衡优势函数估计中的偏差和方差。
* `num_minibatches`: 将一个 batch 的数据分成多少个 mini-batch。
* `update_epochs`: 对同一批数据进行优化的轮数。
* `norm_adv`: 是否对优势函数进行归一化，一种稳定训练的技巧。
* `clip_coef`: PPO 算法中代理目标函数的裁剪系数。
* `clip_vloss`: 是否对价值函数的损失进行裁剪。
* `ent_coef`: 熵损失的系数，用于鼓励探索。
* `vf_coef`: 价值函数损失的系数，用于平衡策略学习和价值学习。
* `max_grad_norm`: 梯度裁剪的最大范数，防止梯度爆炸。
* `target_kl`: 目标 KL 散度，用于一些 PPO 的变体中，作为提前停止更新的条件。

---

### 实验配置/管理参数 (Configuration & Management Parameters)

这些参数用于**设置实验环境、记录和复现**，它们不直接参与算法的核心数学计算，而是管理整个实验流程。

* `exp_name`: 实验名称，用于日志和文件保存。
* `seed`: 随机种子，用于保证实验的可复现性。
* `torch_deterministic`: 是否使用 PyTorch 的确定性算法，也是为了可复现性。
* `cuda`: 是否使用 GPU。
* `track`, `wandb_project_name`, `wandb_entity`: 用于 Weights and Biases 实验跟踪的配置。
* `capture_video`: 是否录制智能体表现的视频。
* `env_id`: 指定要运行的强化学习环境（例如 "CartPole-v1"），它定义了问题本身，而不是解决问题的算法。

---

### 运行时计算的变量 (Runtime-Calculated Variables)

这些变量的值不是预先设定的，而是**在程序运行时根据其他参数计算得出**的。放在args里是为了方便函数传递

* `batch_size`: 批处理大小，通常由 `num_envs * num_steps` 计算得出。
* `minibatch_size`: 小批量大小，通常由 `batch_size / num_minibatches` 计算得出。
* `num_iterations`: 迭代次数，通常由 `total_timesteps / batch_size` 计算得出。



### Gymnasium库的实验设置

```python
def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
```

返回值是一个闭包函数`thunk`。
**Thunk** 是一个编程概念，特指一个**用于封装和延迟某段计算的无参数函数**。
**闭包** 是指一个**函数**以及其**创建时所在的词法环境（Lexical Environment）** 的组合。

这个函数将会被用于创建数个并行的`env`环境：
```python
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ],
    )
```



### Actor Critics类


在 [[30 - Resources/Code Repos/CleanRL]] 的`ppo.py`中，Actor-Critics框架是这样实现的：



正交初始化权重，常数初始化偏置
```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
```

```python
import torch.nn as nn
import numpy as np

class Agent(nn.Module):
    def __init__(self, envs): # env 来自gymnasium库，对象包含了关于环境的重要信息，如观测空间（observation space）和动作空间（action space）的维度
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)



```

结构非常简单，只包含两个神经网络：`critic`和`actor`.

对于Critic:
  * `self.critic = nn.Sequential(...)`: 这里定义了评论家网络。`nn.Sequential` 是一个容器，它会将一系列的层（layers）按顺序串联起来，形成一个神经网络。数据会依次通过这些层。
  * **第一层 (输入层)**:
      * `nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)`: 这是一个全连接层（Linear Layer）。
      * **输入维度**: `np.array(envs.single_observation_space.shape).prod()`。这行代码计算了环境观测空间的总维度。
          * `envs.single_observation_space.shape` 获取观测空间的形状（例如，对于一个 84x84 的灰度图像，形状是 `(84, 84)`）。
          * `.prod()` 计算形状中所有元素的乘积（例如 `84 * 84 = 7056`）。这样做的目的是将可能的多维观测数据（如图像）\*\*展平（flatten）\*\*成一个一维向量，以便输入到全连接层。
      * **输出维度**: `64`。该层会将输入向量转换为一个包含 64 个特征的向量。
      * `layer_init(...)`: 这是一个自定义的权重初始化函数（代码中未提供，但通常用于设置初始权重，以帮助模型更稳定地训练）。

  * **第二层 (激活函数)**:
      * `nn.Tanh()`: 这是一个 Tanh 激活函数。它将上一层的输出值压缩到 $[-1, 1]$ 的范围内，为网络增加非线性。

  * **第三层 (隐藏层)**:
      * `layer_init(nn.Linear(64, 64))`: 另一个全连接层，输入和输出维度都是 64。

  * **第四层 (激活函数)**:
      * `nn.Tanh()`: 再次使用 Tanh 激活函数。

  * **第五层 (输出层)**:
      * `layer_init(nn.Linear(64, 1), std=1.0)`: 这是输出层。
      * **输出维度**: `1`。这是因为评论家的目标是预测一个**标量值（scalar）**，即当前状态的价值 $V(s)$。所以输出维度是 1。
      * `std=1.0`: 在初始化这一层时，可能使用了标准差为 1.0 的方法，这是根据具体的算法需求设定的。

对于Actor:

  * **输入层和隐藏层**: 与评论家网络完全相同。它们都接收相同的状态观测作为输入，并提取相似的特征。在某些更高级的架构中，这两部分网络可能会共享这些前面的层，以提高计算效率。
  * **输出层**:
      * `layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)`: 这是演员网络的输出层，也是它与评论家网络最关键的区别。
      * **输出维度**: `envs.single_action_space.n`。这个值代表了环境中**离散动作的数量**。例如，如果一个游戏有“上”、“下”、“左”、“右”四个动作，那么 `envs.single_action_space.n` 就是 4。
      * **输出含义**: 这一层的输出是每个动作的 **logits**（原始的、未经归一化的预测值）。这些 logits 之后通常会通过一个 **Softmax** 函数转换成一个概率分布，表示在当前状态下选择每个动作的概率。
      * `std=0.01`: 为策略网络的最后一层使用较小的权重初始化标准差是一种常见技巧。这可以确保在训练开始时，网络输出的动作概率接近于均匀分布，从而鼓励智能体进行更多的**探索（exploration）**。

此外还有两个类函数：

`get_value(self, x)`: 接受一个状态`x`，输出对状态x的价值估计`V(x)`。这个`x`在强化学习论文中常称为s。

`get_action_and_value(self, x, action=None):` 接受一个状态x，输出其判断的`action`，对数概率 $\log(\pi(action|x))$ , 动作分布的熵（Entropy）， 和当前状态的价值`V(x)`。当输入`action`的时候则只输出执行这个`action`对应的参数。





### 进iteration循环前的准备工作

```python
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # 这里利用之前定义的函数并行开num_envs个gym来做采集
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # type: ignore
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)  # type: ignore
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
```

### 训练循环内

### 最外层循环框架


对于PPO的一个Iteration而言，[[30 - Resources/Code Repos/CleanRL]]的实现有以下内容：

循环`num_iterations`个iteration
```python
    for iteration in range(1, args.num_iterations + 1):
```


即训练前期使用高学习率，随着学习的进行逐渐降低。
一个简单的**线性学习率调度器 (Linear Learning Rate Scheduler)** 的实现如下：

```python
        if args.anneal_lr: # 如果打开了退火
            frac = 1.0 - (iteration - 1.0) / args.num_iterations # 非常简单的从1到0线性递减
            lrnow = frac * args.learning_rate # 实际的learning rate = 设定的lr * 系数
            optimizer.param_groups[0]["lr"] = lrnow # optimizer是一个pytorch优化器对象。比如，可以是AdamW。优化器可以管理多组参数，通常我们只用一组，所以用 [0] 来访问第一组。
```


On Policy的数据采集特点是，每个iteration采集一次，存在Rollout Buffer中，进行若干轮epochs的更新，然后进入到下一个buffer。

PPO的实现如下。以下代码片段来自[[30 - Resources/Code Repos/CleanRL]]

```python
        for step in range(0, args.num_steps):
            global_step += args.num_envs  # 因为我们一次性驱动 `num_envs` 个环境向前走一步，所以总步数要增加 `num_envs`
            obs[step] = next_obs # 记录行动前的状态。
            dones[step] = next_done # 这是游戏结束与否的信号。

            # ALGO LOGIC: action logic
            with torch.no_grad():  # 只做计算不做更新的不用追踪梯度
                action, logprob, _, value = agent.get_action_and_value(next_obs) # actor-critcs网络算操作和评分了
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob # 

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            # 解释一下这些参数吧。
            # obs：观测。在有些游戏（如围棋）中obs即state，有些游戏（如的德州扑克）中不是
            # reward：从环境中获得的奖励。这里还是并行操作，不命名成rewards是因为这个名字被整个序列的rewards缓冲区给占了
            # terminateions： True 指游戏正常结束
            # truncations： True 指游戏被截断，通常是因为超时了
            # infos: 调试辅助信息
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1) # -1即自动推断tensor大小
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]: # 遍历所有刚刚结束了回合的环境
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}" # 分别是该回合的**总奖励 (return)** 和**总长度 (length)**。
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        ) # Tensorboard日志文件
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )
```


回顾一下公式：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad (1)
$$
$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t-1}\delta_{T-1} \quad (2)
$$

倒倒序循环是为了递归计算。把公式（2）变形一下：
$$
\begin{align*}
\hat{A}_t &= \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \dots \\
&= \delta_t + \gamma\lambda (\delta_{t+1} + (\gamma\lambda)\delta_{t+2} + \dots) \\
&= \delta_t + \gamma\lambda \hat{A}_{t+1}
\end{align*}
$$

```python
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):# 经典倒序循环，为了递归
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value # 这里其实是循环的第一步
                else:
                    nextvalues = values[t + 1]
                delta = ( # 公式1计算
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = ( # 公式2计算
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values


```





当我们不再关心数据分别来自哪个序列的时候，我们就可以把num_env维的tensor给扁平化了。

```python
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)  # type: ignore
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
```


还是先来点公式经典回顾
$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} \hat{A}_t \right] \quad (6)
$$
$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right] \quad (7)
$$
$$
L^{CLIP+VF+S}_t(\theta) = \hat{\mathbb{E}}_t \left[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t) \right] \quad (9)
$$
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad (12)
$$
$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t-1}\delta_{T-1} \quad (11)
$$


代码如下：

```python
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size) # 创建一个0到batch_size-1的数组供后面打乱用
        clipfracs = []
        for epoch in range(args.update_epochs):
	        # 打乱索引数组（用于训练时打破数据间的时序关联）
            np.random.shuffle(b_inds) 
            # 将打乱后的索引切片，取出一个mini-batch用于后续计算
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
				
				# 计算重要性采样比率 (Ratio)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds] # 除法的对数=对数的剪发
                ratio = logratio.exp() # 还原

				# PPO算法的诊断指标
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean() # 这个算法是无偏的，但是高方差，仅放在这里供人参考。实际用的是下面这个approx_kl
                    approx_kl = ((ratio - 1) - logratio).mean() # 这是新的算法，比较准。
                    # 把这个minibatch中的值被clip的比率加到列表里
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

				
                mb_advantages = b_advantages[mb_inds]
                # 把原始优势值转换为其z-score
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                # 就是L                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss 即L                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean() # 顺便一提这个0.5是用来抵消求导那个指数2的，实际上乘不乘都没有关系
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    
				
				# 公式9第三部分
                entropy_loss = entropy.mean()
                # 公式9完整
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
				
				# 标准训练。梯度清零-反向传播-梯度裁剪-更新参数
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
			
			# 如果KL散度变化过大，就停止继续利用这套mini batch里的东西，进下一个epoch
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
```


[[20 - Concepts/Explained Variance 解释方差]]
```python
# y_pred是模型的预测值, y_true是目标真实值
# b_values 是价值网络(Critic)对一批状态(states)的价值预测 V(s)
# b_returns 是根据实际经验(trajectory)计算出的一批回报(returns) G_t
y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()

# 计算真实回报 y_true 的方差
# 这是数据本身固有的波动程度
var_y = np.var(y_true)

# 计算解释方差
# np.var(y_true - y_pred) 是预测误差的方差
# (1 - 误差的方差 / 真实值的方差) 就是解释方差
explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
```

写log
```python
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
```









