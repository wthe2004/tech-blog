---
{"publish":true,"created":"2025-09-19T20:43:02.098-04:00","modified":"2025-09-25T10:20:57.355-04:00","tags":["ai","ppo","rl"],"cssclasses":""}
---

## PPO论文中的GAE
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
      * 当 $\lambda=1$ 时（对应公式10的情况），退化成蒙特卡洛估计（不自举）。这种情况过度依赖采样的“运气”，即方差过高。
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

### 为什么在 $\lambda=1$ 时会退化成不自举的蒙特卡洛估计呢？

答案是数学魔法：**级数求和抵消 (Telescoping Sum)**。**

让我们来手动展开一下公式，看看当 $\lambda=1$ 时发生了什么。

---

此时，GAE 公式 (11) 变为：
$$\hat{A}_t = \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} + \dots$$

现在，我们把 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (公式 12) 代入进去：

$\hat{A}_t = \underbrace{(r_t + \gamma V(s_{t+1}) - V(s_t))}_{\delta_t} + \gamma \underbrace{(r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1}))}_{\delta_{t+1}} + \gamma^2 \underbrace{(r_{t+2} + \gamma V(s_{t+3}) - V(s_{t+2}))}_{\delta_{t+2}} + \dots$

接下来，我们把括号展开，重新组合一下：

$\hat{A}_t = (r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots) + (\gamma V(s_{t+1}) - V(s_t)) + (\gamma^2 V(s_{t+2}) - \gamma V(s_{t+1})) + (\gamma^3 V(s_{t+3}) - \gamma^2 V(s_{t+2})) + \dots$

现在，神奇的事情发生了！我们来看所有带 `V` 的项：
* 第一个括号里的 `+ γV(s_{t+1})`
* 第二个括号里的 `- γV(s_{t+1})`
* 第二个括号里的 `+ γ²V(s_{t+2})`
* 第三个括号里的 `- γ²V(s_{t+2})`
* ...

你会发现，**所有中间的价值项 $V(s_{t+1}), V(s_{t+2}), \dots$ 都被完美地正负抵消了！**

唯一没有被抵消的 `V` 项，只剩下一开始的那个 `-V(s_t)`。

所以，当 $\lambda=1$ 时，整个 GAE 公式最终简化为：

$$\hat{A}_t = \underbrace{(r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots)}_{\text{蒙特卡洛回报 }G_t} - V(s_t)$$

* 公式的第一部分 $(r_t + \gamma r_{t+1} + \dots)$ 就是**纯粹的蒙特卡洛回报 (Monte Carlo Return)**，它的计算**完全不依赖任何中间状态的价值估计**，只和真实的、观测到的奖励 `r` 有关。这部分是**非自举**的。
* 整个公式里唯一存在的价值项是 `-V(s_t)`，它只是被用作一个基线（baseline）来降低方差，而**不是被用在“目标”的计算里**。

因此，当 $\lambda=1$ 时，虽然计算过程中的每一个 $\delta_t$ 单独看是自举的，但将它们按照 GAE 公式加权求和后，所有用于自举的中间项 $V(s_{t+1}), V(s_{t+2}), \dots$ 都被神奇地消除了。

### 实现


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


