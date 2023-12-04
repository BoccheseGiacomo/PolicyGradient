# PolicyGradient
An experiment on using PG to stabilize a divergent dynamical system:

![image](https://github.com/BoccheseGiacomo/PolicyGradient/assets/104854120/c2baef06-a3ea-4215-acc8-4af3be6a6757)

(where a(t) represents the action taken by the agent at the time t, given as an input to our dynamical system)


# Policy Gradient Reinforcement Learning for Control Applications

## Introduction
This project involves the implementation of a neural network-based agent that utilizes a policy gradient reinforcement learning algorithm. The core objective is to stabilize a divergent differential equation. The agent receives a reward at each timestep based on its position, and the algorithm aims to optimize the policy, typically converging in about 300 episodes. The policy gradient algorithm with a continuous action space is used, and the entire implementation is done in Python using PyTorch and NumPy. This document includes specific code examples but not the entire implementation.

## Problem Statement
The dynamic system is governed by the equation:

![image](https://github.com/BoccheseGiacomo/PolicyGradient/assets/104854120/c2baef06-a3ea-4215-acc8-4af3be6a6757)

Where `a(t)` represents the action taken by the agent at time `t`. The action is partly deterministic and partly stochastic, sampled from a normal distribution based on mean and standard deviation computed by the neural network.

### Step Function
```python
def step(self, action):
    c_rew0 = self.rew_fn(self.state)
    self.state += (np.array([action]) * 0.2 + self.state * 0.05) * DT
    c_rew = self.rew_fn(self.state)
    self.reward = c_rew - c_rew0
    return self.state, self.reward
```

### Reward Function
```python
def rew_fn(self, state):
    return 1 - 0.5 * state**2
```

The challenge is to stabilize the dynamic system by finding a policy for `a(t)` that keeps the system as close to zero as possible, maximizing the reward in the process.

## Algorithm Architecture
The policy is defined by a neural network:

```python
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.sigmoid = nn.Sigmoid()
        self.mean = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.logstd = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        mean = self.mean(x)
        logstd = self.logstd(x)
        logstd = torch.clamp(logstd, min=-20, max=2)
        std = torch.exp(logstd)
        return mean, std
```

The policy maps the state space to a probability distribution in the action space, specifically a Gaussian distribution centered at the mean with a standard deviation `std`. The network consists of a linear layer, a sigmoidal layer, and two parallel linear layers for mean and log standard deviation.

### Action Sampling Function
```python
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    mean, std = policy(state)
    dist = Normal(mean, std)
    action = dist.sample()
    lp = dist.log_prob(action)
    policy.log_probs.append(lp)
    return action.item()
```

### Training Method
```python
def finish_episode():
    gamma = GAMMA
    R = 0
    policy_loss = []
    returns = []
    tot_rew = 0
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(np.array(returns)).float()
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    for log_prob, R in zip(policy.log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    tot_rew = sum(policy.rewards).item()
    del policy.rewards[:]
    del policy.log_probs[:]
    return tot_rew
```

## Results
The algorithm generally converges after 200-400 episodes, with a computation time of about 30 seconds. The cumulative reward is plotted against the number of episodes. Plots of the policy mean are included to visualize its performance.

## Conclusion
This setup, combining a neural network with a continuous policy gradient algorithm, is notably simple and versatile, suitable for a wide range of problems.
