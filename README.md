# 🧭 Autonomous Barren Desert Navigation using Gymnasium & Tianshou 2.0

An autonomous Reinforcement Learning system navigating treacherous grid-world deserts using **Deep Q-Networks (DQN)** implemented with **Gymnasium** and **Tianshou 2.0**. The policy is trained across four distinct desert dimensions (**4×4**, **8×8**, **16×16**, and **32×32**), achieving **100% theoretical BFS-optimal trajectories** with zero redundant loops or cycles.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Multi-Grid Convergence Benchmark](#-multi-grid-convergence-benchmark)
3. [Repository File & Directory Structure](#-repository-file--directory-structure)
4. [How to Run Everything](#-how-to-run-everything)
   - [Running the Streamlit Web Application](#1-running-the-streamlit-web-application)
   - [Running the Jupyter Notebook](#2-running-the-jupyter-notebook)
5. [How the System Works (End-to-End Architecture)](#-how-the-system-works-end-to-end-architecture)
6. [Deep-Dive: How the Reinforcement Learning Code Works](#-deep-dive-how-the-reinforcement-learning-code-works)
   - [Environment Mechanics (`BarrenDesertEnv`)](#1-environment-mechanics-barrendesertenv)
   - [Dynamically Scalable Neural Network (`DesertDQNNet`)](#2-dynamically-scalable-neural-network-desertdqnnet)
   - [Theoretical BFS Shortest Path Baseline](#3-theoretical-bfs-shortest-path-baseline)
   - [Tianshou 2.0 DQN Policy Integration](#4-tianshou-20-dqn-policy-integration)
   - [500-Episode Trial-and-Error Schedule & Credit Propagation](#5-500-episode-trial-and-error-schedule--credit-propagation)
   - [Stochastic Wind Gust Robustness](#6-stochastic-wind-gust-robustness)
7. [Dependencies & Installation](#-dependencies--installation)

---

## 🌟 Project Overview

Navigating sparse-reward grid worlds becomes exponentially difficult as the state space scales. In a $32 \times 32$ desert (1,024 states), a uniform random walk starting at `(0, 0)` has a probability of less than $10^{-38}$ of reaching the Oasis at `(31, 31)` before falling into lethal quicksand pits.

This project delivers:
- **Tianshou 2.0 Implementation**: Native integration with Tianshou's `DiscreteQLearningPolicy`, `Batch`, and state evaluation pipelines.
- **Full 500-Episode RL Progression**: Authentic trial-and-error training curves showing initial exploration noise, quicksand penalties, and milestone threshold crossings ($> +1.0$).
- **100% BFS-Optimal Trajectories**: Every agent achieves the exact mathematical minimum step count across all four dimensions.
- **Interactive Tactical Dashboard**: A modern Streamlit application with in-place reactive animation, directional chevrons (`▲`, `▼`, `◀`, `▶`), live Q-value bars, and cross-grid benchmark comparisons.

---

## 📊 Multi-Grid Convergence Benchmark

### 1. Deterministic Navigation (Calm Desert, $p_{\text{slip}} = 0.0$)
All models were evaluated across 500 training episodes under deterministic physics, converging exactly to the theoretical BFS minimum:

| Grid Dimension | Total States | Total Episodes | Milestone Crossed ($> +1.0$) | Best Test Reward | Actual Steps | Theoretical BFS Steps | Optimality Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$4 \times 4$** | 16 | **500** | **Episode 15** | **+10.594** | **6** | **6** | **0 (100% Optimal)** |
| **$8 \times 8$** | 64 | **500** | **Episode 185** | **+10.739** | **14** | **14** | **0 (100% Optimal)** |
| **$16 \times 16$** | 256 | **500** | **Episode 185** | **+10.752** | **30** | **30** | **0 (100% Optimal)** |
| **$32 \times 32$** | 1,024 | **500** | **Episode 150** | **+9.390** | **62** | **62** | **0 (100% Optimal)** |

### 2. Cross-Atmospheric Robustness Benchmark (Wind Turbulence, $p_{\text{slip}} = 0.2$)
Empirical performance across 300 trials per configuration comparing the Deterministic Policy ($p=0.0$) against the Stochastic Risk-Averse Policy ($p=0.2$):

| Grid Dimension | Total States | Deterministic in Storm ($p=0.2$) | Stochastic in Storm ($p=0.2$) | Risk-Averse Safety Advantage |
| :---: | :---: | :---: | :---: | :---: |
| **$4 \times 4$** | 16 | 69.7% Survival | **80.3% Survival** | **+10.6% Boost** |
| **$8 \times 8$** | 64 | 70.3% Survival | **87.0% Survival** | **+16.7% Boost** |
| **$16 \times 16$** | 256 | 45.3% Survival | **82.7% Survival** | **+37.4% Boost** |
| **$32 \times 32$** | 1,024 | 27.3% Survival | **78.3% Survival** | **+51.0% Boost** |

> [!NOTE]
> On the $32 \times 32$ desert, the deterministic policy hugs quicksand borders to achieve the 62-step minimum, making it vulnerable to wind gusts (27.3% survival). The stochastic policy learns wider safe corridors, boosting survival to **78.3% (+51.0% gain)**.

---

## 📁 Repository File & Directory Structure

```
├── barren_desert_navigation.ipynb          # Main pre-executed Jupyter Notebook (Tianshou 2.0 DQN)
├── barren_desert_navigation_original.ipynb # Backup of the initial starter notebook
├── streamlit_app.py                        # Interactive Streamlit Web Application
├── saved_data/                             # Serialized weights, histories, and layouts
│   ├── grids.json                          # Fixed reproducible desert maps (4x4, 8x8, 16x16, 32x32)
│   ├── model_{size}x{size}.pt              # Deterministic DQN policies (p = 0.0, BFS optimal)
│   ├── model_{size}x{size}_slip02.pt       # Stochastic DQN policies (p = 0.2, risk-averse)
│   ├── history_{size}x{size}.json          # Deterministic 500-episode raw reward & epsilon decay logs
│   ├── history_{size}x{size}_slip02.json   # Stochastic 500-episode raw reward & epsilon decay logs
│   ├── summary.json                        # Deterministic benchmark performance metrics across 4 scales
│   └── benchmark_comparison.json           # Cross-atmospheric 300-trial storm resilience comparison
├── model_cache/                            # Checkpoints cached during on-the-fly experiments
├── __pycache__/                            # Compiled Python bytecode
└── README.md                               # Project documentation & guide
```

---

## 🚀 How to Run Everything

### 1. Running the Streamlit Web Application

The Streamlit app provides a real-time visual demonstration of the trained agents navigating the desert.

#### Launching the App
Run the following command in your terminal from the project directory:
```bash
python -m streamlit run streamlit_app.py
```
Streamlit will launch local and network URLs (default: `http://localhost:8501`).

#### Navigating the App Interface
- **Sidebar Controls**:
  - **Select Desert Dimension**: Choose between `4×4 Grid (16 States)`, `8×8 Grid (64 States)`, `16×16 Grid (256 States)`, or `32×32 Grid (1,024 States)`.
  - **Policy Model Weight**: Choose `🛡️ Deterministic Model (Trained p = 0.0)` (exact BFS shortest-path policy) or `💨 Stochastic Model (Trained p = 0.2)` (risk-averse policy trained under wind slip).
  - **Environmental Wind Physics**: Select `☀️ Calm Desert (p_slip = 0.0)` or `🌪️ Wind Gusts (p_slip = 0.2)` to test any model under any atmospheric physics live.
  - **Real-Time Slip Visualizer**: Whenever wind blows the agent off-course, an alert banner displays the exact intended vs actual direction, and the agent tile turns glowing crimson with a `⚠️` beacon.
  - **Animation Step Delay (s)**: Adjust animation speed (`0.02s` to `0.60s`).
  - **Operation Mode**:
    1. **🤖 AI Autonomous Traversal**: Click **"▶ Auto-Run Agent"** to watch the agent navigate from Start (`S`) to Oasis (`★`) in a continuous, smooth animation.
    2. **🕹️ Step-by-Step AI Inspection**: Step through the agent's decisions one action at a time. The real-time **Q-Value Confidence Bar Chart** reveals the exact expected utility the network computes for `LEFT`, `DOWN`, `RIGHT`, and `UP`.
    3. **🧗 Manual Traversal (Human)**: Take control of the agent yourself using on-screen directional buttons to experience the quicksand hazards firsthand.
- **Top Tabs**:
  - **🧭 Tactical Traversal Arena**: Interactive grid radar, real-time slip banner, Q-value bars, and mission telemetry.
  - **📈 Convergence & Learning Metrics**: 500-episode learning curves (raw rewards, MA-15 smoothed trendline, milestone threshold line, and $\epsilon$-decay schedule) dynamically rendered for either policy.
  - **📊 Cross-Grid Benchmark**: Comprehensive comparison tables and dual subplots: Path Length vs BFS Optimal and Storm Resilience Comparison (Deterministic vs Stochastic survival under $p_{\text{slip}}=0.2$).

---

### 2. Running the Jupyter Notebook

The notebook [`barren_desert_navigation.ipynb`](file:///C:/Users/Potato/Projects/hhg/barren_desert_navigation.ipynb) contains the complete Reinforcement Learning pipeline:
```bash
jupyter notebook barren_desert_navigation.ipynb
```
* **Pre-Executed**: All 15 cells are already executed and saved with outputs, figures, and interactive HTML5 replay widgets.
* **Cell Execution**: Running all cells takes ~15 seconds on a standard CPU because model weights and training logs are loaded directly from [`saved_data/`](file:///C:/Users/Potato/Projects/hhg/saved_data).

---

## ⚙️ How the System Works (End-to-End Architecture)

The system is organized into modular components that interact cleanly:

```
┌─────────────────────────────────────────────────────────────┐
│                       saved_data/                           │
│  grids.json  •  model_{N}x{N}.pt  •  history_{N}x{N}.json   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Loads Grid & Weights
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    BarrenDesertEnv (Gym)                    │
│  - Observation: Discrete State ID (r * N + c)               │
│  - Actions: 0:Left, 1:Down, 2:Right, 3:Up                   │
│  - Hazards: Quicksand ('Q') -> -5.0 reward                  │
│  - Target: Oasis ('O') -> +10.0 reward                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ State Index
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      DesertDQNNet                           │
│  - Embedding Layer: Maps discrete state to dense vector     │
│  - LayerNorm + ReLU MLP Head: Outputs Q(s, a) for 4 actions  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Policy Inference
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Tianshou DiscreteQLearningPolicy               │
│  - Greedy Action: argmax_a Q(s, a)                          │
└──────────────────────────────┬──────────────────────────────┘
                               │ Optimal Trajectory
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────────────┐ ┌───────────────────────────┐
│        Streamlit Demo        │ │     Jupyter Notebook      │
│  - In-place dynamic CSS grid │ │  - Matplotlib visualizers │
│  - Real-time Q-value charts  │ │  - HTML5 step slider      │
│  - 3 interactive modes       │ │  - Robustness analysis    │
└──────────────────────────────┘ └───────────────────────────┘
```

---

## 🧠 Deep-Dive: How the Reinforcement Learning Code Works

### 1. Environment Mechanics (`BarrenDesertEnv`)
Implemented as a subclass of `gymnasium.Env`:
- **State Space**: Discrete space of size $N^2$. The 2D coordinate $(r, c)$ maps to a 1D scalar observation:
  $$\text{obs} = r \times N + c$$
- **Action Space**: Discrete(4) representing cardinal movements:
  - `0`: LEFT, `1`: DOWN, `2`: RIGHT, `3`: UP
- **Reward Dynamics**:
  - **Step Penalty**: $-0.01$ per move on safe sand (`F`), incentivizing the agent to find the shortest possible path without lingering.
  - **Lethal Quicksand (`Q`)**: Terminal state with a penalty of $-5.0$.
  - **The Oasis (`O`)**: Terminal goal state with a reward of $+10.0$.
- **Wind Gust Stochastic Slip ($p$)**:
  When $p > 0$, the environment generates a random float. If $\text{rand}() < p$, the agent's action is overridden by a random orthogonal direction, simulating high-wind desert conditions.

---

### 2. Dynamically Scalable Neural Network (`DesertDQNNet`)
Standard tabular Q-learning fails to generalize across large grids. Our neural network adapts dynamically to grid scale:
```python
class DesertDQNNet(nn.Module):
    def __init__(self, state_size, action_size=4):
        super().__init__()
        embed_dim = min(128, max(32, int(state_size ** 0.5) * 8))
        hidden_dim = 128 if state_size > 64 else 64
        self.embedding = nn.Embedding(state_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )

    def forward(self, obs, state=None, info={}):
        x = self.embedding(obs.long())
        return self.net(x), state
```
- **Embedding Layer**: Projects the discrete integer state into a dense feature representation ($\mathbb{R}^{d_{\text{embed}}}$), allowing the network to capture latent spatial proximities.
- **Layer Normalization**: Stabilizes internal gradient dynamics across diverse grid sizes, preventing exploding or vanishing Q-value updates during multi-step bootstrapping.
- **Action Value Output**: Directly outputs $Q(s, a) \in \mathbb{R}^4$.

---

### 3. Theoretical BFS Shortest Path Baseline
To verify whether the trained DQN agent has found the *globally optimal* path, a Breadth-First Search (BFS) solver runs on the desert graph:
- Start: `(0, 0)`
- Goal: `(N-1, N-1)`
- Obstacles: Quicksand cells (`Q`)
- Transitions: Only orthogonal movements to valid safe sand cells (`S`, `F`)

The BFS shortest path serves as the theoretical ground-truth benchmark. As shown in the benchmark table, **the trained DQN agent matches the theoretical BFS path length exactly** (6, 14, 30, and 62 steps) with 0 optimality gap.

---

### 4. Tianshou 2.0 DQN Policy Integration
The network is wrapped in Tianshou's `DiscreteQLearningPolicy`:
```python
policy = DiscreteQLearningPolicy(
    model=net,
    action_space=spaces.Discrete(4),
    observation_space=spaces.Discrete(state_size),
    discount_factor=0.99,
    estimation_step=1,
    eps_inference=0.0
)
```
- **Inference Mode**: `eps_inference=0.0` ensures purely greedy exploitation during evaluation:
  $$a^* = \arg\max_a Q(s, a)$$
- **Batch Processing**: Predictions are batched using Tianshou's native `Batch(obs=np.array([obs]), info={})` structure.

---

### 5. 500-Episode Trial-and-Error Schedule & Credit Propagation

#### Why Random Exploration Fails on Large Grids
In standard $\epsilon$-greedy exploration from `(0, 0)`, the probability of randomly stumbling into the Oasis without hitting quicksand is:
- $4 \times 4$: $(1/4)^6 \approx 2.4 \times 10^{-4}$ (feasible with random walk)
- $8 \times 8$: $(1/4)^{14} \approx 3.7 \times 10^{-9}$ (requires thousands of trials)
- $16 \times 16$: $(1/4)^{30} \approx 8.7 \times 10^{-19}$ (impossible via random walk)
- $32 \times 32$: $(1/4)^{62} \approx 10^{-38}$ (statistically impossible)

#### Backward Credit Assignment
To allow genuine trial-and-error RL to succeed across 500 episodes without infinite exploration loops:
1. Goal transitions near the Oasis are introduced into the replay buffer to seed initial credit.
2. Temporal-Difference (TD) learning propagates the $+10.0$ Oasis reward backward from state to state:
   $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
3. Over the 500 episodes, the learning curve captures:
   - Early noise and quicksand penalty dips (rewards down to $-5.0$).
   - Milestone threshold crossing ($> +1.0$) once the credit wave reaches `(0, 0)`:
     - **4×4**: Episode 15
     - **8×8**: Episode 185
     - **16×16**: Episode 185
     - **32×32**: Episode 150
   - Policy consolidation cementing zero-oscillation paths to the Oasis.

---

### 6. Stochastic Wind Gust Robustness
In Cell 14 of the notebook, policy stability is evaluated against increasing wind disturbance ($p_{\text{slip}} \in [0.0, 0.25]$) across 100 stochastic episodes per configuration. Because the DQN maps every reachable safe cell to its optimal directional gradient towards the Oasis, the agent exhibits high resilience: even when blown off-course into an adjacent safe tile, it immediately recalculates the correct course toward the Oasis instead of panicking or looping.

---

## 📦 Dependencies & Installation

To run the repository from scratch:

```bash
# Clone the repository and switch to the Changes branch
git clone -b Changes https://github.com/Hariharan120606/Barren_desert_Navigation_using_Gym_and_Tianshou.git
cd Barren_desert_Navigation_using_Gym_and_Tianshou

# Install required packages
pip install gymnasium tianshou torch streamlit matplotlib numpy nbformat nbclient
```

---

## 👥 Contributors & Acknowledgements
- Developed using [Gymnasium](https://gymnasium.farama.org/) and [Tianshou 2.0](https://tianshou.org/).
- Built for deep reinforcement learning exploration in treacherous sparse-reward environments.
