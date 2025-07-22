# Methodology

## 3.1 Problem Formulation and Threat Model

### 3.1.1 Federated Learning Framework Definition

We consider a federated learning system with $N$ participating clients, where each client $k \in \{1, 2, \ldots, N\}$ possesses a local dataset $\mathcal{D}_k = \{(x_i, y_i)\}_{i=1}^{|\mathcal{D}_k|}$ that remains private and cannot be shared. The objective is to collaboratively train a global model $\theta^*$ that minimizes the federated learning loss:

$$
\theta^* = \arg\min_\theta F(\theta) = \sum_{k=1}^N \frac{|\mathcal{D}_k|}{|\mathcal{D}|} F_k(\theta)
$$

where $F_k(\theta) = \frac{1}{|\mathcal{D}_k|} \sum_{(x,y) \in \mathcal{D}_k} \ell(\theta; x, y)$ represents the local loss function for client $k$, $\ell(\cdot)$ is the loss function (e.g., cross-entropy), and $|\mathcal{D}| = \sum_{k=1}^N |\mathcal{D}_k|$ is the total dataset size.

The federated learning process operates over $T$ communication rounds. In each round $t$, a subset of clients $\mathcal{S}_t \subseteq \{1, 2, \ldots, N\}$ participates in training, where $|\mathcal{S}_t| = C \cdot N$ with client participation ratio $C \in (0, 1]$.

### 3.1.2 Threat Model and Attack Capabilities

We consider a Byzantine threat model where a fraction $f \in [0, 0.5)$ of clients may exhibit malicious behavior. Specifically, we assume that up to $\lfloor f \cdot N \rfloor$ clients can be compromised and controlled by an adversary. The threat model encompasses the following attack capabilities:

**Data Poisoning Attacks**: Malicious clients can corrupt their local training data by:
- **Label Flipping**: Randomly flipping labels with probability $p_{\text{flip}} \in [0, 1]$:
  $$\tilde{y}_i = \begin{cases}
    y_i & \text{with probability } 1 - p_{\text{flip}} \\
    \text{random}(\{0, 1, \ldots, C-1\} \setminus \{y_i\}) & \text{with probability } p_{\text{flip}}
  \end{cases}$$

**Gradient Manipulation Attacks**: Malicious clients can modify their gradient updates before sending to the server:
- **Scaling Attack**: $\tilde{g}_k = \lambda \cdot g_k$ where $\lambda \gg 1$
- **Partial Scaling Attack**: $\tilde{g}_k = g_k \odot (1 + (\lambda-1) \cdot M)$ where $M$ is a binary mask with density $\rho \in (0, 1]$
- **Sign Flipping Attack**: $\tilde{g}_k = -g_k$
- **Additive Noise Attack**: $\tilde{g}_k = g_k + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

### 3.1.3 Security Assumptions

Our approach operates under the following assumptions:
1. **Honest Majority**: Less than 50% of clients are malicious ($f < 0.5$)
2. **Server Integrity**: The central server remains trusted and cannot be compromised
3. **Validation Data Availability**: A small, clean validation dataset $\mathcal{D}_{\text{val}}$ is available at the server
4. **Communication Security**: Secure communication channels prevent eavesdropping and man-in-the-middle attacks

## 3.2 Dataset Description and Non-IID Data Distribution

### 3.2.1 Medical Imaging Datasets

**MNIST Dataset**: We utilize the Modified National Institute of Standards and Technology (MNIST) handwritten digit dataset as a controlled baseline, containing 70,000 grayscale images (60,000 training, 10,000 testing) of size $28 \times 28$ pixels across 10 digit classes (0-9).

**Alzheimer's Disease MRI Dataset**: Our primary medical imaging dataset consists of brain MRI scans categorized into four cognitive stages:
- No Impairment (Healthy): 3,200 images
- Very Mild Impairment: 2,240 images  
- Mild Impairment: 896 images
- Moderate Impairment: 64 images

All MRI images are preprocessed to $224 \times 224$ pixels with intensity normalization to $[0, 1]$ and augmentation techniques including rotation ($\pm 15°$), horizontal flipping, and Gaussian noise injection ($\sigma = 0.02$).

**CIFAR-10 Dataset**: For complex natural image evaluation, we employ CIFAR-10 containing 60,000 color images ($32 \times 32$ pixels) across 10 object classes, split into 50,000 training and 10,000 testing samples.

### 3.2.2 Non-IID Data Distribution Modeling

Real-world federated learning scenarios exhibit significant data heterogeneity. We model three types of non-IID distributions:

**Dirichlet Distribution**: For each client $k$, we sample class proportions from a Dirichlet distribution:
$$p_k(y = c) \sim \text{Dir}(\alpha \cdot \mathbf{1}_C)$$
where $\alpha \in \{0.1, 0.5, 1.0, \infty\}$ controls heterogeneity level. Lower $\alpha$ values indicate higher heterogeneity, with $\alpha = \infty$ corresponding to IID distribution.

**Label Skew Distribution**: Each client receives data from only $Q \in \{1, 2, \ldots, C\}$ classes:
$$\mathcal{C}_k = \text{random\_sample}(\{0, 1, \ldots, C-1\}, Q)$$
where $\mathcal{C}_k$ represents the subset of classes available to client $k$.

**Quantity Skew Distribution**: Clients receive different amounts of data following a log-normal distribution:
$$|\mathcal{D}_k| \sim \text{LogNormal}(\mu, \sigma^2)$$

### 3.2.3 Data Preprocessing and Normalization

**MRI-Specific Preprocessing**:
1. **Skull Stripping**: Removal of non-brain tissue using FSL BET
2. **Intensity Normalization**: $I_{\text{norm}} = \frac{I - \mu_{\text{brain}}}{\sigma_{\text{brain}}}$
3. **Spatial Standardization**: Registration to MNI152 template space
4. **Slice Selection**: Extraction of axial slices from $z = 60$ to $z = 120$

**General Preprocessing Pipeline**:
- Pixel value normalization to $[0, 1]$: $x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$
- Data augmentation with probability $p = 0.3$
- Train/validation/test split: 70%/15%/15%

## 3.3 Core Federated Optimization Algorithms

### 3.3.1 Foundational Aggregation Methods

**FedAvg (Federated Averaging)**: The baseline approach performs simple weighted averaging:
$$\theta^{t+1} = \sum_{k \in \mathcal{S}_t} \frac{|\mathcal{D}_k|}{|\mathcal{D}_{\mathcal{S}_t}|} \theta_k^{t+1}$$
where $|\mathcal{D}_{\mathcal{S}_t}| = \sum_{k \in \mathcal{S}_t} |\mathcal{D}_k|$.

**FedProx (Proximal Federated Optimization)**: Addresses client drift by adding a proximal term:
$$\min_{w} F_k(w) + \frac{\mu}{2}\|w - \theta^t\|^2$$
where $\mu > 0$ is the proximal coefficient controlling the regularization strength.

**FedBN (Federated Batch Normalization)**: Maintains local batch normalization statistics while aggregating other parameters:
$$\theta^{t+1}_{\text{BN}} = \theta_k^{t+1}_{\text{BN}}, \quad \theta^{t+1}_{\text{others}} = \sum_{k \in \mathcal{S}_t} \frac{|\mathcal{D}_k|}{|\mathcal{D}_{\mathcal{S}_t}|} \theta_{k,\text{others}}^{t+1}$$

### 3.3.2 Hybrid FedBNP Algorithm

We propose **FedBNP** (Federated Batch Normalization + Proximal), combining the benefits of FedBN and FedProx:

**Local Training Phase**:
$$\theta_k^{t+1} = \arg\min_{\theta} \left[ F_k(\theta) + \frac{\mu}{2}\|\theta - \theta^t\|^2 \right]$$

**Global Aggregation Phase**:
$$\theta^{t+1} = \begin{cases}
\sum_{k \in \mathcal{S}_t} w_k^t \theta_{k,\text{non-BN}}^{t+1} & \text{for non-BN parameters} \\
\theta_{k,\text{BN}}^{t+1} & \text{for BN parameters (kept local)}
\end{cases}$$

where $w_k^t$ represents the trust-based weights computed by our detection mechanism.

**Rationale for FedBNP**:
- **FedBN Component**: Addresses feature shift in medical imaging data across different MRI scanners and acquisition protocols
- **FedProx Component**: Reduces client drift in heterogeneous medical datasets where local distributions vary significantly
- **Combined Effect**: Maintains local adaptation while ensuring global consistency

## 3.4 Multi-Criteria Gradient Fingerprinting

### 3.4.1 Comprehensive Feature Extraction Framework

We extract six complementary features from each client's gradient $g_k^t$ to create a comprehensive behavioral profile:

**Feature 1 - VAE Reconstruction Error**:
$$e_{\text{VAE}}(g_k) = \|g_k - \text{VAE}_{\text{decode}}(\text{VAE}_{\text{encode}}(g_k))\|_2^2$$

The VAE is trained on gradients from honest clients to learn the normal gradient distribution. Higher reconstruction errors indicate anomalous behavior.

**Feature 2 - Cosine Similarity to Reference**:
$$\cos(g_k, g_{\text{ref}}) = \frac{g_k^T g_{\text{ref}}}{\|g_k\|_2 \|g_{\text{ref}}\|_2}$$

where $g_{\text{ref}}$ is computed as the median gradient across all participating clients in round $t$.

**Feature 3 - Average Cosine Similarity to Peers**:
$$\bar{\cos}(g_k) = \frac{1}{|\mathcal{S}_t| - 1} \sum_{j \in \mathcal{S}_t, j \neq k} \cos(g_k, g_j)$$

**Feature 4 - Gradient L2 Norm**:
$$\|g_k\|_2 = \sqrt{\sum_{i=1}^d g_{k,i}^2}$$

where $d$ is the gradient dimensionality.

**Feature 5 - Sign Consistency Ratio**:
$$\text{SCR}(g_k) = \frac{1}{d} \sum_{i=1}^d \mathbb{I}(\text{sign}(g_{k,i}) = \text{sign}(g_{\text{ref},i}))$$

where $\mathbb{I}(\cdot)$ is the indicator function.

**Feature 6 - Shapley Value**:
$$\hat{\phi}_k = \frac{1}{M} \sum_{m=1}^M [v(S_m \cup \{k\}) - v(S_m)]$$

where $M$ is the number of Monte Carlo samples, $S_m$ is a random subset of clients, and $v(\cdot)$ represents the validation accuracy achieved by aggregating gradients from the specified client subset.

### 3.4.2 VAE Architecture for Gradient Anomaly Detection

**Encoder Network**:
$$\begin{align}
h_1 &= \text{ReLU}(W_1 g + b_1) \\
h_2 &= \text{ReLU}(W_2 h_1 + b_2) \\
\mu &= W_\mu h_2 + b_\mu \\
\log \sigma^2 &= W_\sigma h_2 + b_\sigma
\end{align}$$

**Reparameterization Trick**:
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Decoder Network**:
$$\begin{align}
h_3 &= \text{ReLU}(W_3 z + b_3) \\
h_4 &= \text{ReLU}(W_4 h_3 + b_4) \\
\hat{g} &= W_5 h_4 + b_5
\end{align}$$

**Loss Function**:
$$\mathcal{L}_{\text{VAE}} = \|\hat{g} - g\|_2^2 + \beta \cdot \text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I))$$

### 3.4.3 Shapley Value Computation via Monte Carlo Sampling

The Shapley value quantifies each client's marginal contribution to the global model performance:

$$\phi_k = \sum_{S \subseteq \mathcal{N} \setminus \{k\}} \frac{|S|! (N-|S|-1)!}{N!} [v(S \cup \{k\}) - v(S)]$$

Due to computational complexity, we use Monte Carlo approximation:

**Algorithm 1: Monte Carlo Shapley Estimation**
```
Input: Client gradients {g_1, ..., g_N}, validation set D_val, samples M
Output: Shapley values {φ_1, ..., φ_N}

1: Initialize φ_k = 0 for all k
2: for m = 1 to M do
3:    π = random_permutation({1, 2, ..., N})
4:    θ_curr = θ^t  // Start with current global model
5:    v_prev = evaluate_model(θ_curr, D_val)
6:    for i = 1 to N do
7:        k = π[i]
8:        θ_curr = update_model(θ_curr, g_k)  // Apply gradient g_k
9:        v_curr = evaluate_model(θ_curr, D_val)
10:       φ_k += (v_curr - v_prev)
11:       v_prev = v_curr
12: return {φ_k/M for all k}
```

### 3.4.4 Feature Normalization and Scaling

To ensure feature comparability, we apply z-score normalization:
$$f_{\text{norm}}^{(i)} = \frac{f^{(i)} - \mu_f^{(i)}}{\sigma_f^{(i)}}$$

where $\mu_f^{(i)}$ and $\sigma_f^{(i)}$ are the running mean and standard deviation of feature $i$ computed across recent communication rounds.

## 3.5 Dual Attention Mechanism and Reinforcement Learning Integration

### 3.5.1 Motivation for Hybrid Approach

The combination of dual attention and reinforcement learning addresses complementary aspects of malicious client detection:

- **Dual Attention**: Provides immediate, reactive analysis of gradient features within each communication round, capturing instant anomaly patterns
- **Reinforcement Learning**: Enables adaptive, long-term policy learning across multiple rounds, allowing the system to evolve its detection strategy based on historical attack patterns

**Mathematical Integration**:
$$t_k^t = \sigma(\alpha \cdot a_k^t + (1-\alpha) \cdot \pi_\theta(s_k^t))$$

where:
- $t_k^t \in [0,1]$ is the final trust score for client $k$ at round $t$
- $a_k^t$ is the dual attention output
- $\pi_\theta(s_k^t)$ is the RL policy output given state $s_k^t$
- $\alpha \in [0,1]$ is the combination weight (learned or fixed)
- $\sigma(\cdot)$ is the sigmoid activation function

### 3.5.2 Transformer-Based Dual Attention Architecture

**Multi-Head Self-Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention**:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$
where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**Dual Attention Layers**:

*Gradient Feature Attention*: Focuses on relationships between different gradient features
$$H_1 = \text{LayerNorm}(X + \text{MultiHead}(X, X, X))$$
$$H_1' = \text{LayerNorm}(H_1 + \text{FFN}(H_1))$$

*Temporal Attention*: Captures temporal dependencies across communication rounds
$$H_2 = \text{LayerNorm}(H_1' + \text{MultiHead}(H_1', H_{\text{memory}}, H_{\text{memory}}))$$

**Final Classification Layer**:
$$a_k^t = \sigma(W_{\text{out}} \cdot \text{GlobalAvgPool}(H_2) + b_{\text{out}})$$

### 3.5.3 Reinforcement Learning Agent Design

**State Space Definition**:
The RL agent observes a state vector $s_k^t \in \mathbb{R}^{d_s}$ comprising:
$$s_k^t = [f_1(g_k^t), f_2(g_k^t), \ldots, f_6(g_k^t), h_1^t, h_2^t, \ldots, h_p^t]$$

where $f_i(\cdot)$ are the six gradient features and $h_j^t$ are historical context features:
- Global model accuracy at round $t$
- Round number normalized by total rounds
- Client participation rate
- Consensus measure: $\frac{1}{|\mathcal{S}_t|^2} \sum_{i,j \in \mathcal{S}_t} \cos(g_i^t, g_j^t)$

**Action Space**:
The action $a_k^t \in [0, 1]$ represents the trust weight assigned to client $k$'s update.

**Actor Network Architecture**:
$$\begin{align}
h_{\text{actor}}^{(1)} &= \text{ReLU}(W_{\text{actor}}^{(1)} s_k^t + b_{\text{actor}}^{(1)}) \\
h_{\text{actor}}^{(2)} &= \text{ReLU}(W_{\text{actor}}^{(2)} h_{\text{actor}}^{(1)} + b_{\text{actor}}^{(2)}) \\
\pi_\theta(s_k^t) &= \text{Sigmoid}(W_{\text{actor}}^{(3)} h_{\text{actor}}^{(2)} + b_{\text{actor}}^{(3)})
\end{align}$$

**Critic Network Architecture**:
$$\begin{align}
h_{\text{critic}}^{(1)} &= \text{ReLU}(W_{\text{critic}}^{(1)} [s_k^t; a_k^t] + b_{\text{critic}}^{(1)}) \\
h_{\text{critic}}^{(2)} &= \text{ReLU}(W_{\text{critic}}^{(2)} h_{\text{critic}}^{(1)} + b_{\text{critic}}^{(2)}) \\
Q_\phi(s_k^t, a_k^t) &= W_{\text{critic}}^{(3)} h_{\text{critic}}^{(2)} + b_{\text{critic}}^{(3)}
\end{align}$$

### 3.5.4 Reward Function Design

The reward function balances model performance improvement and detection accuracy:

$$R^t = w_1 \cdot \Delta \text{Acc}^t + w_2 \cdot \text{F1}_{\text{detection}}^t - w_3 \cdot \text{FPR}^t$$

where:
- $\Delta \text{Acc}^t = \text{Acc}^t - \text{Acc}^{t-1}$ is the accuracy improvement
- $\text{F1}_{\text{detection}}^t$ is the F1-score for malicious client detection
- $\text{FPR}^t$ is the false positive rate
- $w_1 = 0.5$, $w_2 = 0.4$, $w_3 = 0.1$ are weighting coefficients

### 3.5.5 Training Algorithm for RL Agent

We employ the Deep Deterministic Policy Gradient (DDPG) algorithm:

**Actor Update**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho} [\nabla_a Q_\phi(s,a)|_{a=\pi_\theta(s)} \nabla_\theta \pi_\theta(s)]$$

**Critic Update**:
$$L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} [(Q_\phi(s,a) - y)^2]$$
where $y = r + \gamma Q_{\phi'}(s', \pi_{\theta'}(s'))$

**Soft Target Updates**:
$$\begin{align}
\theta' &\leftarrow \tau \theta + (1-\tau) \theta' \\
\phi' &\leftarrow \tau \phi + (1-\tau) \phi'
\end{align}$$

## 3.6 Comprehensive Algorithm Framework

### 3.6.1 Overall System Architecture

**Algorithm 2: Federated Learning with Dual Attention and RL**
```
Input: N clients, T communication rounds, malicious fraction f
Output: Global model θ^T

1: Initialize global model θ^0, VAE, dual attention network, RL agent
2: Train VAE on honest gradients from initial rounds
3: for t = 1 to T do
4:    // Client Selection and Local Training
5:    S_t = select_clients(C × N)
6:    for k ∈ S_t do
7:        θ_k^{t+1} = local_training(θ^t, D_k)  // FedBNP local update
8:        g_k^t = θ_k^{t+1} - θ^t
9:        send g_k^t to server
10:   
11:   // Server-Side Processing
12:   for k ∈ S_t do
13:       // Extract gradient features
14:       f_1^k = VAE_reconstruction_error(g_k^t)
15:       f_2^k = cosine_similarity(g_k^t, g_ref^t)
16:       f_3^k = average_cosine_similarity(g_k^t, {g_j^t}_{j≠k})
17:       f_4^k = gradient_norm(g_k^t)
18:       f_5^k = sign_consistency_ratio(g_k^t, g_ref^t)
19:       f_6^k = shapley_value(g_k^t, D_val)
20:       
21:       // Compute trust scores
22:       X_k = normalize([f_1^k, f_2^k, f_3^k, f_4^k, f_5^k, f_6^k])
23:       a_k^t = dual_attention(X_k, memory_bank)
24:       s_k^t = construct_state(X_k, historical_context)
25:       π_k^t = RL_policy(s_k^t)
26:       t_k^t = σ(α × a_k^t + (1-α) × π_k^t)
27:   
28:   // Federated Aggregation with Trust Weights
29:   θ^{t+1} = Σ_{k∈S_t} t_k^t × g_k^t / Σ_{k∈S_t} t_k^t
30:   θ^{t+1} = apply_FedBNP(θ^{t+1}, {θ_k^{t+1}}_{k∈S_t})
31:   
32:   // RL Agent Training
33:   R^t = compute_reward(θ^{t+1}, detection_results)
34:   update_RL_agent(s^t, a^t, R^t, s^{t+1})
35:   
36:   // Update Memory Bank
37:   update_memory_bank(X^t, t^t)
38: return θ^T
```

### 3.6.2 Illustrative Examples

**Example 1 - IID vs Non-IID Scenario**:
Consider four clients with different data distributions:
- Clients A, B: IID distribution with balanced classes
- Client C: Dirichlet $\alpha = 0.1$ (high heterogeneity)
- Client D: Label skew with $Q = 2$ classes

Expected trust score behavior:
- Clients A, B: $t_{A,B} \approx 0.8-0.9$ (high trust)
- Client C: $t_C \approx 0.6-0.7$ (moderate trust due to data heterogeneity)
- Client D: $t_D \approx 0.5-0.6$ (lower trust due to limited class diversity)

**Example 2 - Attack Detection Scenario**:
- Client 1: Honest behavior → $t_1 \approx 0.85$
- Client 2: Scaling attack with $\lambda = 10$ → $t_2 \approx 0.15$
- Client 3: Partial scaling attack → $t_3 \approx 0.30$
- Client 4: Label flipping attack → $t_4 \approx 0.25$

## 3.7 Computational Complexity Analysis

### 3.7.1 Time Complexity

**Per Communication Round**:
- Gradient feature extraction: $\mathcal{O}(N \cdot d \cdot F)$ where $F$ is feature computation cost
- VAE reconstruction: $\mathcal{O}(N \cdot d \cdot H_{VAE})$ where $H_{VAE}$ is VAE hidden dimension
- Shapley value computation: $\mathcal{O}(N \cdot M \cdot C_{eval})$ where $M$ is Monte Carlo samples
- Dual attention processing: $\mathcal{O}(N \cdot L_{att} \cdot H_{att}^2)$ where $L_{att}$ is attention layers
- RL agent inference: $\mathcal{O}(N \cdot H_{RL})$ where $H_{RL}$ is RL network size
- Model aggregation: $\mathcal{O}(N \cdot d)$

**Overall Complexity**: $\mathcal{O}(T \cdot N \cdot (d \cdot F + M \cdot C_{eval} + L_{att} \cdot H_{att}^2))$

### 3.7.2 Space Complexity

- Gradient storage: $\mathcal{O}(N \cdot d)$
- Feature vectors: $\mathcal{O}(N \cdot 6)$
- Attention memory bank: $\mathcal{O}(M_{bank} \cdot H_{att})$
- RL experience replay: $\mathcal{O}(M_{replay} \cdot (d_s + d_a + 1))$

### 3.7.3 Communication Complexity

**Per Round Communication**:
- Standard FL: $\mathcal{O}(N \cdot d)$ for gradient transmission
- Additional overhead: $\mathcal{O}(N \cdot 6)$ for feature vectors
- Relative overhead: $\frac{6}{d} \ll 1$ for typical deep networks

## 3.8 Implementation Details and Hyperparameters

### 3.8.1 Network Architectures

**Global Model (Medical CNN)**:
```python
Architecture:
- Conv2d(1, 32, kernel=3) → BatchNorm → ReLU
- Conv2d(32, 64, kernel=3) → BatchNorm → ReLU → MaxPool
- Conv2d(64, 128, kernel=3) → BatchNorm → ReLU
- Conv2d(128, 256, kernel=3) → BatchNorm → ReLU → MaxPool
- AdaptiveAvgPool(4,4) → Dropout(0.5)
- Linear(256×16, 128) → ReLU → Dropout(0.3)
- Linear(128, num_classes)

Parameters: ~1.2M for Alzheimer's classification
```

**VAE Configuration**:
- Encoder: [grad_dim, 1024, 512, 256, 128]
- Latent dimension: 64
- Decoder: [64, 128, 256, 512, 1024, grad_dim]
- Training epochs: 50 on honest gradients

**Dual Attention Network**:
- Input dimension: 6 (features)
- Hidden dimension: 128
- Number of heads: 8
- Number of layers: 3
- Dropout rate: 0.1

**RL Agent Architecture**:
- Actor: [state_dim=12, 128, 64, 1]
- Critic: [state_dim+action_dim=13, 128, 64, 1]
- Learning rates: Actor=1e-4, Critic=1e-3
- Discount factor: γ = 0.99
- Soft update rate: τ = 0.001

### 3.8.2 Training Hyperparameters

**Federated Learning Setup**:
- Number of clients: N = 20
- Client participation ratio: C = 0.5
- Communication rounds: T = 100
- Local epochs: E = 5
- Batch size: 32
- Learning rate: η = 0.01 with cosine annealing
- Malicious fraction: f ∈ {0.1, 0.2, 0.3}

**FedBNP Parameters**:
- Proximal coefficient: μ = 0.1
- Momentum: 0.9
- Weight decay: 1e-4

**Attack Parameters**:
- Scaling attack: λ ∈ {5, 10, 20}
- Partial scaling: ρ ∈ {0.3, 0.5, 0.7}
- Noise attack: σ ∈ {0.1, 0.5, 1.0}
- Label flipping: p_flip ∈ {0.2, 0.5, 0.8}

### 3.8.3 Evaluation Metrics

**Model Performance**:
- Global test accuracy
- Per-class F1-scores
- Area Under Curve (AUC) for medical diagnosis
- Convergence rate (rounds to reach target accuracy)

**Detection Performance**:
- Precision: TP/(TP+FP)
- Recall: TP/(TP+FN)
- F1-score: 2×(Precision×Recall)/(Precision+Recall)
- False Positive Rate: FP/(FP+TN)
- Area Under ROC Curve (AUROC)

**System Robustness**:
- Attack Success Rate: 1 - Recall
- Robustness Index: Final_Accuracy/Initial_Accuracy
- Byzantine Resilience: Performance degradation under maximum attack intensity

## 3.9 Ablation Studies and Sensitivity Analysis

### 3.9.1 Component-wise Ablation

**Study 1 - Individual Feature Importance**:
We systematically remove each of the six gradient features to assess their individual contributions:
- Baseline (all features): Performance metric P
- Remove VAE error: P - ΔP₁
- Remove cosine similarity: P - ΔP₂
- Remove peer similarity: P - ΔP₃
- Remove gradient norm: P - ΔP₄
- Remove sign consistency: P - ΔP₅
- Remove Shapley value: P - ΔP₆

**Study 2 - Architecture Comparison**:
- Dual Attention Only: Using only attention mechanism for trust scoring
- RL Only: Using only reinforcement learning for weight assignment
- Baseline Methods: FedAvg, FedProx, FedBN without detection
- Static Threshold: Fixed threshold-based detection

**Study 3 - Hyperparameter Sensitivity**:
- Combination weight α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- Shapley samples M ∈ {5, 10, 20, 50}
- RL learning rates ∈ {1e-5, 1e-4, 1e-3}
- Detection threshold sensitivity

### 3.9.2 Robustness Evaluation

**Cross-Dataset Generalization**:
- Train on MNIST, test on CIFAR-10
- Train on synthetic attacks, test on adaptive attacks
- Medical domain transfer: Train on one MRI dataset, test on another

**Adaptive Attack Resistance**:
- Gradient masking attacks
- Model replacement attacks
- Coordinated multi-client attacks

## 3.10 Statistical Analysis and Experimental Design

### 3.10.1 Experimental Setup

**Cross-Validation Strategy**:
- 5-fold cross-validation for client assignment
- Stratified sampling to ensure balanced malicious client distribution
- Multiple random seeds (42, 123, 456, 789, 1000) for statistical significance

**Statistical Tests**:
- Paired t-tests for accuracy comparisons
- McNemar's test for detection performance
- ANOVA for multi-method comparisons
- Bonferroni correction for multiple comparisons

### 3.10.2 Confidence Intervals and Significance Testing

All results are reported with 95% confidence intervals. Statistical significance is assessed at α = 0.05 level. Effect sizes are reported using Cohen's d for practical significance assessment.

**Sample Size Calculation**:
Power analysis conducted to ensure adequate sample size for detecting meaningful differences (effect size ≥ 0.5) with 80% power.

## 3.11 Ethical Considerations and Limitations

### 3.11.1 Privacy and Security

**Privacy Preservation**:
- Gradient-level differential privacy with ε = 2.0, δ = 1e-5
- Secure aggregation protocols for gradient transmission
- Client anonymization through cryptographic techniques

**Medical Data Compliance**:
- HIPAA compliance for medical imaging data
- IRB approval for human subject research
- Data de-identification procedures

### 3.11.2 System Limitations

**Computational Constraints**:
- Shapley value computation scales exponentially with client number
- Real-time processing requirements limit detection complexity
- Memory constraints for large-scale deployments

**Attack Model Assumptions**:
- Static attack strategies (not adaptive)
- Known attack types for training
- Honest majority assumption

**Generalization Limitations**:
- Domain-specific medical imaging focus
- Limited to computer vision tasks
- Assumption of centralized server trust

### 3.11.3 Future Research Directions

**Theoretical Extensions**:
- Convergence analysis under Byzantine conditions
- Privacy-utility trade-off optimization
- Adaptive attack modeling

**Practical Improvements**:
- Decentralized detection mechanisms
- Cross-modal federated learning
- Real-time deployment optimization

---

This comprehensive methodology provides a detailed, mathematically rigorous framework for federated learning with dual attention and reinforcement learning-based malicious client detection, specifically designed for medical imaging applications while maintaining generalizability to other domains. 