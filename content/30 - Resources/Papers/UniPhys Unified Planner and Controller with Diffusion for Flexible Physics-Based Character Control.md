---
{"publish":true,"created":"2025-11-24T20:46:45.784-05:00","modified":"2025-11-25T00:01:52.301-05:00","tags":["ai","diffusion"],"cssclasses":""}
---


[archive link](https://arxiv.org/pdf/2504.12540)

### Abstract

- Goal
	- Generating natural and physically plausible character motion
	- long-horizon control with diverse guidance signals
- Prior Works
	- combines high-level diffusion-based motion planners with low-level physics controllers
		- domain gaps that degrade motion quality and require task-specific fine-tuning
- Solution
	- UniPhys, a diffusion-based behavior cloning framework that unifies motion planning and control into a single model
		- Inputs: multi-modal inputs such as text, trajectories, and goals
		- Output: physically plausible, long-horizon motions
		- Principle: Diffusion Forcing paradigm
			1. denoise noisy motion histories
			2. handle discrepancies introduced by the physics simulator
			3. no task-specific find-tuning

![[UniPhys fig1.png]]

Figure 1
 - (a) text-driven control with dynamic language instructions
 - (b) precise velocity control
 - (c) sparse goal reaching
 - (d) adapting to dynamic environments with moving object avoidance.

### Introduction

- Prior research
	1. RL dataset tracking + Policy Distillation
		- supervised learning
		- multi-model signals input
		- Limitation
			- lack diversity
			- rely on hand-crafted heuristics or manually designed intermediate targets to guide
	2. diffusion-based generative motion models
		- generate intermediate targets to enable controllers to achieve longer-term goals and complex tasks
		- input multimodal conditioning signals
		- support arbitrary loss guidances
		- Limitation
			- gap between planner and controller
				- low motion quality
				- controler failed to track
			- require fine-tune (lack generalization)
- UniPhys
	- Principles
		- behavior cloning framework
		- diffusion-based policy model
		- input arbitrary guidance signals
	- Diffusion Forcing paradigm [arxiv](https://arxiv.org/abs/2407.01392) 
		- denoise sequences with frames with varying noise levels
		- assume history slightly noisy
	- task-specific guided sampling techniques
	- a large-scale physics character motion state-action dataset with frame-level text annotation from [BABEL](https://babel.is.tue.mpg.de/)

### Related Work

- Human Motion Synthesis:
	- diversity-physical trade-off
- Physiscs based character animation:
	- no generalization
	- lack of diversity
	- hierarchical frameworks cause unnatural behavior
- Diffusion model for planning and control
	- separate planning and control

### Preliminary

- Physics Simulation Setup
	- action $\mathbf{a}_t \in \mathbb{R}^{J \times 3}$ 
	- state $\mathbf{s}_t$
	- prediction $\mathbf{s}_{t+1} = \mathrm{SIM}(\mathbf{s}_t, \mathbf{a}_t)$
- Physics-based character tracking policy
	- [PHC](https://arxiv.org/abs/2305.06456)
	- goal state $\mathbf{s}_t^{g}$
	- $\mathbf{a}_t = \pi_{\text{PHC}}(\mathbf{s}_t, \mathbf{s}_t^{g})$
	- PPO optimization to align next state $\mathbf{s}_{t+1} = \mathrm{SIM}(\mathbf{s}_t, \mathbf{a}_t)$ and goal state $\mathbf{s}_t^{g}$
- Physics-based motion latent space
	- [PULSE](https://arxiv.org/abs/2310.04582)
	- distills the PHC tracking policy into a physics-based latent motion space with conditional variation autoencoder (cVAE) for generative control
	- Encoder $\mathbf{z}_t \sim \mathcal{E}(\mathbf{z}_t \mid \mathbf{s}_t, \mathbf{s}_t^{g})$
	- Decoder $\mathbf{a}_t = \mathcal{D}(\mathbf{s}_t, \mathbf{z}_t)$
	- $\mathbf{z}_t = \pi_{\text{task}}(\mathbf{o}_t, \mathbf{g}_t^{\text{task}})$ , where $\mathbf{o}_t$ is the current observation and $\mathbf{g}_t^{\text{task}}$ is the task goal.
	- embedding $z_{t}$ captures the dynamic transition between consecutive frames

![[UniPhys fig2.png]]
Figure 2:We construct a large-scale paired state-action dataset by tracking MoCap dataset with [PULSE](https://arxiv.org/abs/2310.04582) tracking policy

### UniPhys: Unified Planner and Controller
#### Dataset Curation

A dataset with state-action sequences and text descriptions suitable for learning physics-based control policies

- Principle
	- tracked motions from the AMASS dataset using PULSE tracking policy
	- $(s_{t}, a_{t}, z_{t})$
	- frame-level text annotations from the BABEL dataset

#### Diffusion-Based Behavior Generative Model

##### Capabilities
- end-to-end control driven by high-level text instructions
- precise state-space control via gradient-based guidance during the diffusion denoising process
- long-horizon planning by simultaneously predicting future states and actions.

##### Behavior representation

(Canonicalized: Calculated in a coordinate system centered on the root node)
- behavior sequences $\mathbf{X} = \mathbf{x}_{1:T}$
- $\mathbf{x}_t = (\mathbf{s}^c_t, \mathbf{z}_t)$ , in which $s^c_{t}$ is the canonicalized state sequence
- canonicalized state sequences  $\mathbf{s}^c_{1:T}$
- latent action embedding sequences $\mathbf{z}_{1:T}$
- high-dimensional action space $\mathbf{a}_t$
- well-regularized latent action representation $\mathbf{z}_t \in \mathbb{R}^{32}$ (encoded by the PULSE encoder)
- state sequences $\mathbf{S}^c = (\mathbf{r}^c_{1:T}, \mathbf{p}^c_{1:T}, \mathbf{v}^c_{1:T}, \mathbf{q}_{1:T}, \mathbf{w}_{1:T})$
	- Global root trajectory $\mathbf{r}^c_{1:T} = (\gamma_t, \phi_t, \dot{\gamma}_t, \dot{\phi}_t)_{1:T}$ canonicalized to the first-frame coordinate system
		- root position $\gamma_t \in \mathbb{R}^3$,  
		- orientation $\phi_t \in \mathbb{R}^6$,  
		- linear velocity $\dot{\gamma}_t \in \mathbb{R}^3$,  
		- angular velocity $\dot{\phi}_t \in \mathbb{R}^3$.
		- The canonicalized root trajectory always starts from the origin, and the first frame faces the +y axis.
	- Local joint features, canonicalized to per-frame local coordinate frames, including:  
		- local joint positions $\mathbf{p}^c_t \in \mathbb{R}^{J \times 3}$,  
		- joint velocities $\mathbf{v}^c_t \in \mathbb{R}^{J \times 3}$,  
		- joint rotation $\mathbf{q}_t \in \mathbb{R}^{J \times 6}$,  
		- angular velocity $\mathbf{w}_t \in \mathbb{R}^{J \times 3}$.
		- The per-frame local coordinate system is set at the pelvis joint projected on the ground.  
- canonicalized state $\mathbf{s}_t$ 
- canonicalized state sequenced $\mathbf{S}$ 

##### Independent noise injection per frame

refer to Fig(3) a

the sequence  $\mathbf{X}^0$  is corrupted to  $\mathbf{X}^{\mathbf{k}} = \left(\mathbf{x}_1^{k_1}, \mathbf{x}_2^{k_2}, \dots, \mathbf{x}_T^{k_T}\right)$ where $\mathbf{x}_t^{k_t} = \sqrt{\bar{\alpha}_{t}}\,\mathbf{x}_t^{0} + \sqrt{1 - \bar{\alpha}_{t}}\,\boldsymbol{\epsilon}^{k_t},$  in which $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s = \alpha_1 \times \alpha_2 \times \dots \times \alpha_t$ Cumulative signal retention rate, and $\boldsymbol{\epsilon}^{k_t} \sim \mathcal{N}(0, \mathbf{I})$ is a random noise vector

Per-frame random noise levels $\mathbf{k} = k_{1:T} \in [K]^T$ are independently randomly sampled.

The model is parameterized as $\mathcal{M}_{\theta}(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c})$ to predict the clean behavior sequence, where **c** is the text embedding.

The training loss $\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{k}, \mathbf{X}^0} \left[ \left\lVert \mathbf{X}^0 - \mathcal{M}_{\theta}(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c}) \right\rVert^2 \right]$ is a MSE loss, calculating data sequence $\mathbf{X}^0$ and predicted sequence $\mathcal{M}_{\theta}(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c})$

![[UniPhys Algo1.png]]

##### Guided Behavior Synthesis for Flexible Control

Overall, our guided denoising-based control framework follows a receding horizon strategy with autoregressive behavior synthesis
- condition: past behavior
- denoise future action tokens
- decode action tokens into executable actions

CFG: $\hat{\mathbf{X}}_c^0 = \mathcal{M}_\theta(\mathbf{X}^\mathbf{k}, \mathbf{k}, \emptyset) + \lambda_c(\mathcal{M}_\theta(\mathbf{X}^\mathbf{k}, \mathbf{k}, \mathbf{c}) - \mathcal{M}_\theta(\mathbf{X}^\mathbf{k}, \mathbf{k}, \emptyset)) \tag{2}$ where $\lambda_c$ controls the guidance strength.

Task-Specific Loss-Guided Sampling: $\hat{\mathbf{X}}_l^0 = \mathcal{M}_\theta(\mathbf{X}^\mathbf{k}, \mathbf{k}, \mathbf{c}) - \lambda_l \nabla_{\mathbf{X}^\mathbf{k}}\mathcal{G}(\hat{\mathbf{X}}^0)$ where $\lambda_l$ controls the guidance strength. (Gradient descent)
- Monte-Carlo Guidance: $\nabla' \mathcal{G}(\hat{\mathbf{X}}^0) = \frac{1}{N} \sum_{i=1}^N \nabla_{\mathbf{X}^{\mathbf{k}}} \mathcal{G}(\hat{\mathbf{X}}_{(i)}^0)$, where $N$ is the number of samples

##### Flexible denoising schedule

1. full-sequence diffusion denoising
2. autoregressive denoising schedule
	1. enoises the sequence sequentially
	2. a more stable roll-out
	3. robust
3. gradual denoising process
	1. prioritizes denoising near-future frames while preserving uncertainty in distant ones
	2. improves the performance in long-horizon planning tasks
	3. balance robustness and efficiency

#### Long-horizon rollouts with stabilization

A noise indicator $k$ to signal that previous states are slightly noisy, without adding noise to the state-action predictions

![[UniPhys fig3.png]]
Figure 3:Framework overview. 
1. The model takes a behavior sequence of length T as input and is conditioned on the clip-based text embedding. At training time, each frame is corrupted with different noise levels, and the model learns to predict the clean behavior sequence.
2. At test time, guided denoising with task-specific guidance enables flexible multi-task control. We highlight the flexibility in different test-time denoising conditions and configurations, and the stabilization trick that promotes stable long-horizon autoregressive control.