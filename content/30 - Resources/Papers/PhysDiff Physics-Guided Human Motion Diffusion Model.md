---
{"publish":true,"created":"2025-11-26T18:40:08.879-05:00","modified":"2025-11-28T18:04:53.845-05:00","cssclasses":""}
---

### Abstract

Goal: Eliminate artifacts such as floating, foot sliding, and ground penetration

1. a physicsbased motion projection module that uses motion imitation in a physics simulator to project the denoised motion of a diffusion step to a physically-plausible motion
2. The projected motion is further used in the next diffusion step to guide the denoising diffusion process
3. ntuitively, the use 1of physics in our model iteratively pulls the motion toward a physically-plausible space, which cannot be achieved by simple post-processing

### Introduction

PhysDiff:
- a physics-based motion projection module (details provided later) that projects an input motion to a physically-plausible space
	- During the diffusion process, we use the motion projection module to project the denoised motion of a diffusion step into a physicallyplausible motion.
- Tasks:
	- text-to-motion generation
	- action-to-motion generation
	- (the approach is agnostic to the specific instantiation)
- trade-off between physical plausibility and motion quality when varying the number of physics-based projection steps
	- while more projection steps always lead to better physical plausibility, the motion quality increases before a certain number of steps and decreases after that, i.e., the resulting motion satisfies the physical constraints but still may look unnatural.
- adding the physics-based projection to late diffusion steps performs better than early steps
	- hypothesize that motions from early diffusion steps may tend toward the mean motion of the training data and the physics-based projection could push the motion further away from the data distribution, thus hampering the diffusion process

### Related Works

- Denoising Diffusion Model
	- repeatedly injecting known information to the diffusion proces (motion infilling)
- Human Motion Generation
- Physics-Based Human Motion Modeling
	- Deep Reinforcement Learning

### Method

- human motion $x^{1:H} = \{x^h\}_{h=1}^H$ of length $H$
- pose $x^h \in \mathbb{R}^{J \times D}$
- $D$-dimensional features of $J$ joints
- Starting from a noisy motion $x_T^{1:H}$
- PhysDiff models the denoising distribution $q(x_s^{1:H}|x_t^{1:H}, \mathcal{P}_\pi, c)$
	- denoises the motion from diffusion timestep $t$ to $s$ ($s < t$)
- clean motion $x_0^{1:H}$
- physics-based motion projection module $\mathcal{P}_\pi$
	- enforces physical constraints

#### Physics-Guided Motion Diffusion

##### Motion Diffusion

- the condition $c$
	- uncondition: universal null token $\varnothing$
- the data distribution $p_0(\boldsymbol{x})$
- a series of time-dependent distributions $p_t(\boldsymbol{x}_t)$, $p_t(\boldsymbol{x}_t|\boldsymbol{x}) = \mathcal{N}(\boldsymbol{x}, \sigma_t^2\mathbf{I})$
	- defined by by injecting _i.i.d._ Gaussian noise to samples from $p_0$, i.e., $p_t(\boldsymbol{x}_t|\boldsymbol{x}) = \mathcal{N}(\boldsymbol{x}, \sigma_t^2\mathbf{I})$
		- The possibility for $\boldsymbol{x}$ to be $\boldsymbol{x}_t$ when the mean of the noise is $\boldsymbol{x}$ and the Variance is Noise intensity squared times the identity matrix
		- $\sigma_t$ defines a series of _noise levels_ that is increasing over time such that $\sigma_0 = 0$ and $\sigma_T$ for the largest possible $T$ is much bigger than the data's standard deviation.
- $\mathrm{d}\boldsymbol{x} = -(\beta_t + \dot{\sigma}_t)\sigma_t \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})\mathrm{d}t + \sqrt{2\beta_t}\sigma_t \mathrm{d}\omega_t$
	- $\nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})$ is the score function
	- $\omega_t$ is the standard Wiener process
		- In mathematics, the Wiener process is a _real-valued continuous-time stochastic process_ 
	- $\dot{\sigma}_t = \frac{\mathrm{d}\sigma_t}{\mathrm{d}t}$
	- $\beta_t$ controls the amount of stochastic noise injected in the process
		- when $\beta_{t}$ is zero, the SDE becomes and ordinary differential equation (ODE)
	- The first part of $\mathrm{d}\boldsymbol{x}$ is 

A notable property of the score function $\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)$ is that it recovers the minimum mean squared error (MMSE) estimator of $\boldsymbol{x}$ given $\boldsymbol{x}_t$:

$$\tilde{\boldsymbol{x}} := \mathbb{E}[\boldsymbol{x}|\boldsymbol{x}_t] = \boldsymbol{x}_t + \sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) \quad \tag{2}$$

where we can essentially treat $\tilde{\boldsymbol{x}}$ as a “denoised” version of $\boldsymbol{x}_t$. Since $\boldsymbol{x}_t$ and $\sigma_t$ are known during sampling, we can obtain $\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)$ from $\tilde{\boldsymbol{x}}$, and vice versa.

This function is Tweedie's Formula, which can predict the denoised step but in practice only be used to calculate one step.

Diffusion models approximate the score function with the following denoising autoencoder objective:

$$\mathbb{E}_{\boldsymbol{x} \sim p_0(\boldsymbol{x}), t \sim p(t), \epsilon \sim p(\epsilon)} [\lambda(t) \|\boldsymbol{x} - D(\boldsymbol{x} + \sigma_t\epsilon, t, c)\|_2^2] \quad (3)$$

- $D$ is the denoiser that depends on the noisy data
- the time $t$
- the condition $c$
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$
- $p(t)$ is a distribution from which time is sampled
- $\lambda(t)$ is the loss weighting factor

The optimal solution to $D$ would be one that recovers the MMSE estimator $\tilde{\boldsymbol{x}}$ according to Eq. (2).

![[PhysDiff algo1.png]]

