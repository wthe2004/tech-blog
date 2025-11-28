---
{"publish":true,"created":"2025-11-25T09:26:16.311-05:00","modified":"2025-11-28T13:19:45.312-05:00","tags":["human-motion-generation","rl","ppo","distillation"],"cssclasses":""}
---

[arxiv](https://arxiv.org/abs/2502.20390)

![[InterMimic fig1.png]]

Figure 1:
- dynamic diverse multi object interaction (top)
- scalable skill learning (top)
- application (bottom)

### Abstract

Goal: HOI simulation, physics, zero-shot

Intermimic: a framework that enables a single policy to robustly learn from hours of imperfect MoCap data covering diverse full-body interactions with dynamic and varied objects.

Insight: a curriculum strategy - perfect first, then scale up
1. subject-specific teacher policies to mimic, retarget, and refine motion capture data
2. distill these teachers into a student policy
	- the teachers acting as online experts providing direct supervision, as well as high-quality references
	- incorporate RL fine-tuning on the student policy to surpass mere demonstration replication and achieve higher-quality solutions

### Introduction

Prior:
- MoCap Data
	- need to correct contact errors caused by sensor limitations and occlusions between humans and objects
	- unscalable, as refining a single motion demands a delicate balance between preserving the captured data and ensuring its physical plausibility.
- Physics-based human motion imitation
	- training control policies to mimic reference MoCap data within a physics simulator
	- scaling up human-object interaction imitation presents significant challenges
		- MoCap Imperfection
			- Contact artifacts
			- expected contacts to fluctuate instead of maintaining consistent zero distance
			- missing hand capture
			- diverse human shapes
				- retargeting process is imperfect and can introduce new contact artifacts or exacerbate existing ones
		- Scaling-up
			- whole-body interactions involving dynamic and diverse objects underexplored
This Paper:
- Goal:
	- utilize rich yet imperfect motion capture interaction datasets
	- train a control policy capable of learning diverse motor skills while enhancing the plausibility of these actions by correcting errors
	- tackling the challenges of **skill perfection** and **skill integration** progressively
- Principle:
	- a curriculum-based teacher-student distillation framework
	- Multiple teacher policies
		- focus on imitating and refining small subsets of interactions
		- Principle
			- retargeting
				- to a canonical human model
					- by embedding HOI retargeting directly into the imitation
					- by reframing the policy learning to optimize both imitation and retargeting objectives.
			- recovering
				- teacher model refine interaction motion through learning from it
				- accurate contact dynamics enforced by a physics simulator inherently correct inaccuracies in the reference kinematics
		- Benefits
			- distill raw MoCap data into refined HOI references with a unified embodiment and enhanced physical fidelity
			- These refined references guide the subsequent student policy training
				- reducing the negative impact of errors in the original MoCap data
		- Hurdle: Long training time
			- space-time trade-off: multiple teacher policies are trained in parallel on smaller, more manageable data subsets, and their expertise is then distilled into a single student policy
	- One student policy integrates these skills from the teachers

### Related Works

- Kinematic Interaction Animation
	- Prior
		- physical inaccuracies
		- no hand motion
- Physics-based Interaction Animation
	- Prior
		- DRL track reference motion
		- specific scenarios
		- non-scalable

### Methodology

![[InterMimic fig 2.png]]
1. training each teacher policy (MLP) on a small data subset with initialization corrected via Physical State Initialization (PSI)
2. freezing the teacher policies to provide refined references for training a student policy (Transformer). The student leverages teacher supervision for effective scaling and is fine-tuned through RL

**Task Formulation**
- policy $\pi$
- simulated human-object motion $\{q_t\}_{t=1}^T$
- ground-truth reference $\{\hat{q}_t\}_{t=1}^T$
- pose $q_t$
	- has two components: the human pose $q_t^h$ and the object pose $q_t^o$.
	- $q_t^h = \{\theta_t^h, p_t^h\}$
		- $\theta_t^h \in \mathbb{R}^{51 \times 3}$ represents the joint rotations
		- $p_t^h \in \mathbb{R}^{51 \times 3}$ specifies the joint positions
		- 30 hand joints and 21 joints for the rest of the body
	- $q_t^o = \{\theta_t^o, p_t^o\}$
		- $\theta_t^o \in \mathbb{R}^3$ denotes the object’s orientation
		- $p_t^o \in \mathbb{R}^3$ the position
- All simulation states have corresponding ground-truth values, denoted by the hat symbol
	- For instance, the reference object rotation is $\{\hat{\theta}_t^o\}_{t=1}^T$

**Overview**

interaction imitation: Markov Decision Process(MDP), defined by states, actions, simulator-provided transition dynamics

#### Policy Representation

- State $s_t = \{s_t^s, s_t^g\}$
	- $s_t^s$ contains human proprioception and object observations, expressed as, $\{\{\theta_t^h, p_t^h, \omega_t^h, v_t^h\}, \{\theta_t^o, p_t^o, \omega_t^o, v_t^o\}, \{d_t, c_t\}\}$
		- $\{\theta_t^h, p_t^h, \omega_t^h, v_t^h\}$ represent the rotation, position, angular velocity, and velocity of all joints, respectively
		- $\{\theta_t^o, p_t^o, \omega_t^o, v_t^o\}$ represent the orientation, location, velocity, and angular velocity of the object, respectively.
		- $d_t$ represent vectors from human joints to their nearest points on each object surface
		- $c_t$ represent contact markers indicating whether the human’s rigid body parts experience applied forces
	- goal state $s_t^g = \{s_{t,t+k}^g\}_{k \in K}$
		- $s_{t,t+k}^g = \{\{\hat{\theta}_{t+k}^h \ominus \theta_t^h, \hat{p}_{t+k}^h - p_t^h\}, \{\hat{\theta}_{t+k}^o \ominus \theta_t^o, \hat{p}_{t+k}^o - p_t^o\}, \\ \{\hat{d}_{t+k} - d_t, \hat{c}_{t+k} - c_t\}, \{\hat{\theta}_{t+k}^h, \hat{p}_{t+k}^h, \hat{\theta}_{t+k}^o, \hat{p}_{t+k}^o\}\}$
		- $\hat{\theta}_{t+k}^h, \hat{p}_{t+k}^h, \hat{d}_{t+k}, \hat{c}_{t+k}$ represent the reference information at time step $t+k$
		- $\ominus$ denotes the calculation of **rotation difference**
		- All continuous elements of $s_t$ are normalized relative to the current direction of view of the human and the position of the root (represented by the $\ominus$).
	- reference contact markers $\hat{c}_{t+k}$
- Action
	- $a_t \in \mathbb{R}^{51 \times 3}$
	- 51 joints

#### ImitatIon as Perfecting

1. trajectory collection
2. policy updating

##### Imitation as Retargeting

the same base human model
- It demonstrates possible integration with real-world humanoid deployment, which requires retargeting to a consistent physical embodiment
-  real-world humanoid deployment requires retargeting to a consistent physical embodiment

RL-driven HOI reward
- an embodiment-aware component that loosely aligns the simulated kinematics with the reference interaction
- an embodiment-agnostic reward component that encourages dynamics to be close to the reference.

##### Embodiment-Aware Reward

when human and object are far apart, retargeting should prioritize capturing rotational motion
when they are close, accurate position tracking becomes crucial for achieving contact

- weights $w_d$ that are inversely proportional to the distances between joints and the object
- cost functions
	- joint position $E^h_p = \langle \Delta^h_p, w_d \rangle$
	- rotation $E^h_\theta = \langle \Delta^h_\theta, 1 - w_d \rangle$
	- interaction tracking $E_d = \langle \Delta_d, w_d \rangle$
		- $\langle \cdot, \cdot \rangle$ is the inner product (weighted sum)
		- $\Delta^h_p[i] = \|\hat{p}^h[i] - p^h[i]\|$
		- $\Delta^h_\theta[i] = \|\hat{\theta}^h[i] \ominus \theta^h[i]\|$, 
		- $\Delta_d[i] = \|\hat{d}[i] - d[i]\|$ 
		- these three represent the displacement for the variables defined in Sec. 3.1 with timestep $t$ omitted.
		- $w_d[i] = 0.5 \times \frac{1/\|d[i]\|^2}{\sum_i 1/\|d[i]\|^2} + 0.5 \times \frac{1/\|\hat{d}[i]\|^2}{\sum_i 1/\|\hat{d}[i]\|^2}$
	- contact promotion cost function $E^c_b = \sum \|\hat{c}_b - c\| \odot \hat{c}_b$
		- the adaptive contact marker $\hat{c}_b$ (serve as both binary target and mask)
		- $c$ is the simulated contact extracted from the force detected
		- $\odot$ is Hadamard Product (Element-wise Product)
	- contact penalty cost function $E^c_p = \sum \|c\| \odot \hat{c}_p$
		- same as promotion cost
	- hand contact guidance $E^c_h = \sum \|c^{\text{lhand}} - \hat{c}^{\text{lhand}}\| \odot \hat{c}^{\text{lhand}} + \sum \|c^{\text{rhand}} - \hat{c}^{\text{rhand}}\| \odot \hat{c}^{\text{rhand}}$
		- where $c^{\text{lhand}}$ and $c^{\text{rhand}}$ represent contact labels for rigid body components of the hands.
		- The reference contact markers, $\hat{c}^{\text{lhand}}$ and $\hat{c}^{\text{rhand}}$, are defined when any hand vertices are within an adaptive threshold distance $\sigma$ to the object
	- Energy cost $E^e_h = \sum \|a_h\|$, $E^e_o = \sum \|a_o\|$, and $E^e_c = \max \|f\|$
		- $a_h$ represents the acceleration of human joints
		- $a_o$ the object's acceleration
		- $f$ the force detected on human rigid bodies
		- penalize all these three
	- Multiplicative Reward Aggregation $R = \exp\left(-\sum \lambda_i E_i\right)$
		- namely $R = \exp(-\lambda_1 E_1) \times \exp(-\lambda_2 E_2) \times \dots$
		- Forcing all rewards to be high enough

![[InterMimic fig3.png]]
1. left figure
	1. red: promote contact
	2. green: natural
	3. blue: penalize contact
2. right figure
	1. Initializing the rollout with reference (RSI)
	2. reference corrected via simulation (PSI)

##### Embodiment-Agnostic Reward

object tracking and contact tracking

- object tracking cost
	- position $E^o_p = \|\hat{p}^o - p^o\|$
	- rotation $E^o_\theta = \|\hat{\theta}^o \ominus \theta^o\|$
- body contact
	- promotion $E^c_b$
	- penalty $E^c_p$
- aligning the simulated contact $c$ with reference markers $\hat{c}$,`
	- three contact levels – promotion, penalty, and neutral

physics engine does not differentiate between object, ground, and self-contact
Solution:
1. model foot-ground contact promotion and penalty
	1. ensures proper foot lifting during cyclic walking and mitigates foot hobbling
2. allow self-collision to avoid self-contact promotion but to promote object interaction
	1. poses minimal risk as the policy is guided by MoCap reference

##### Hand Interaction Discovery

Since MoCap dataset lacks hand motion, use RL(a reference contact marker)

##### Policy Learning

$\pi$ is trained using PPO  with the policy gradient $L(\psi) = \mathbb{E}_t[\min(r_t(\psi)A_t, \text{clip}(r_t(\psi), 1 - \epsilon, 1 + \epsilon)A_t]$
- $\psi$ are the parameters of $\pi$
- $r_t(\psi)$ quantifies the difference in action likelihoods between updated and old policies
- $\epsilon$ is a small constant
- $A_t$ is the advantage estimate given by the generalized advantage estimator GAE($\lambda$)

##### Physical State Initialization

Reference State Initialization (RSI) sets the current pose $q_t$ to a reference pose $\hat{q}_t$ at a random timestep $t$, for initializing the rollout
- However, initializing with the imperfect reference can introduce _critical artifacts_, such as contact floating or incorrect hand motion, leading to unrecoverable failures, e.g., object falling, as depicted in Figure 3(ii).
###### Why RSI fails

![[InterMimic figC.png]]
A sanity check on why Reference State Initialization (RSI) can fail:
- bar: the reference interaction sequence that the policy imitates
	- red regions indicate that initializing in those regions leads to immediate failure
	- green regions signify that successful initialization is possible
		- grey regions: the policy cannot collect trajectories for updates due to roll out max length

**_Physical State Initialization_ (PSI)**:

1. creating an initialization buffer that stores reference states from MoCap and simulation states from prior rollouts
2. Cycle
	1. Rollout: an initial state is randomly selected from this buffer, which increases the likelihood of starting from advantageous positions.
	2. (Once a rollout is completed) trajectories are evaluated based on their expected discounted rewards
		1. those above a certain threshold are added to the buffer using a first-in-first-out (FIFO) strategy
		2. while older or lower-quality trajectories are discarded.

##### Interaction Early Termination

Interaction Early Termination
- regular Early Termination
- Object points deviate from their references by more than 0.5 m on average
- Weighted average distances between the character’s joints and the object surface exceed 0.5 m from the reference
- Any required body-object contact is lost for over 10 consecutive frames

#### Imitation with Distillation

##### Reference Distillation

teacher policies trained on smaller-scale data

##### Policy Distillation

#### Architecture

Teacher: MLP
Student: Transformer


![[Pasted image 20251126181305.png]]

#### Applying Physical Constraints

to fix approximation errors

- the denoised motion $\tilde{x}^{1:H}$ (the desired outcome)
- a physics-based motion projection $\mathcal{P}_\pi : \mathbb{R}^{H \times J \times D} \to \mathbb{R}^{H \times J \times D}$
	- maps the original motion $\tilde{x}^{1:H}$ to a physically-plausible one, denoted as $\hat{x}^{1:H}$

![[PhysDiff algo2.png]]

#### Physics-Based Motion Projection

##### Motion Imitation Formulation

- Markov decision process (MDP) defined by a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, R, \gamma)$ of states, actions, transition dynamics, a reward function, and a discount factor
- A character agent acts in a physics simulator according to a motion imitation policy $\pi(a^h|s^h)$, which models the distribution of choosing an action $a^h \in \mathcal{A}$ given the current state $s^h \in \mathcal{S}$. The state $s^h$ consists of the character's physical state (e.g., joint angles, velocities, positions) as well as the next pose $\tilde{x}^{h+1}$ from the input motion. Including $\tilde{x}^{h+1}$ in the state informs the policy $\pi$ to choose an action $a^h$ that can mimic $\tilde{x}^{h+1}$ in the simulator. Starting from an initial state $s^1$, the agent iteratively samples an action $a^h$ from the policy $\pi$ and the simulator with transition dynamics $\mathcal{T}(s^{h+1}|s^h, a^h)$ generates the next state $s^{h+1}$, from which we can extract the simulated pose $\hat{x}^{h+1}$. By running the policy for $H$ steps, we can obtain the physically-simulated motion $\hat{x}^{1:H}$.


the objective is to maximize the expected discounted return $J(\pi) = \mathbb{E}_\pi [\sum_h \gamma^h r^h]$

The reward $r^h$ at each timestep consists of four sub-rewards:

$$r^h = w_{\text{p}} r^h_{\text{p}} + w_{\text{v}} r^h_{\text{v}} + w_{\text{j}} r^h_{\text{j}} + w_{\text{q}} r^h_{\text{q}}, \quad (4)$$
The four errors are all in the same form$$r = \exp [-\alpha \times \text{error}^2]$$
$$r^h_{\text{p}} = \exp [-\alpha_{\text{p}} (\sum_{j=1}^J \|o_j^h \ominus \bar{o}_j^h\|^2)], \quad (5)$$

$$r^h_{\text{v}} = \exp [-\alpha_{\text{v}} \|v^h - \bar{v}^h\|^2], \quad (6)$$

$$r^h_{\text{j}} = \exp [-\alpha_{\text{j}} (\sum_{j=1}^J \|p_j^h - \bar{p}_j^h\|^2)], \quad (7)$$

$$r^h_{\text{q}} = \exp [-\alpha_{\text{q}} (\sum_{j=1}^J \|q_j^h - \bar{q}_j^h\|^2)]. \quad (8)$$

where $w_{\text{p}}, w_{\text{v}}, w_{\text{j}}, w_{\text{q}}, \alpha_{\text{p}}, \alpha_{\text{v}}, \alpha_{\text{j}}, \alpha_{\text{q}}$ are weighting factors.
- The pose reward $r^h_{\text{p}}$ measures the difference between the local joint rotations $o_j^h$ and the ground truth $\bar{o}_j^h$
	- $\ominus$ denotes the relative rotation between two rotations
	- $\|\cdot\|$ computes the rotation angle
- The velocity reward $r^h_{\text{v}}$ measures the mismatch between joint velocities $v^h$ and the ground truth $\bar{v}^h$, 
	- which are computed via finite difference. 
- The joint position reward $r^h_{\text{j}}$ encourages the 3D world joint positions $p_j^h$ to match the ground truth $\bar{p}_j^h$.
- Finally, the joint rotation reward $r^h_{\text{q}}$ measures the difference between the global joint rotations $q_j^h$ and the ground truth $\bar{q}_j^h$.

**States**. The agent state shs^h consists of the character's current physical state, the input motion's next pose x^h+1\hat{x}^{h+1}, and a character attribute vector ψ\psi. The character's physical state includes its joint angles, joint velocities, and rigid bodies' positions, rotations, and linear and angular velocities. For the input pose x^h+1\hat{x}^{h+1}, the state shs^h contains the difference of x^h+1\hat{x}^{h+1} w.r.t. the agent in joint angles as well as rigid body positions and rotations. Using the difference informs the policy about the pose residual it needs to compensate for. All the features are computed in the character's heading coordinate to ensure rotation and translation invariance. Since our character is based on the SMPL body model, the attribute ψ\psi includes the gender and SMPL shape parameters to allow the policy to control different characters.

**Actions**. We use the target joint angles of proportional derivative (PD) controllers as the action representation, which enables robust motion imitation as observed in prior work. We also add residual forces in the action space to stabilize the character and compensate for missing contact forces required to imitate motions such as sitting.

**Policy**. We use a parametrized Gaussian policy π(ah∣sh)=N(μθ(sh),Σ)\pi(a^h|s^h) = \mathcal{N}(\mu_\theta(s^h), \Sigma) where the mean action μθ\mu_\theta is output by a simple multi-layer perceptron (MLP) network with parameters θ\theta, and Σ\Sigma is a fixed diagonal covariance matrix.






