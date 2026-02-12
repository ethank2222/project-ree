# AI Concepts in rl_traffic_agent.py

## 1. Reinforcement Learning (the overall framework)

The entire file is structured around the RL loop: **observe → decide → act → get reward → learn**. This happens in `run_episode()` (lines 166-224):

- **Agent** = the PPO controller
- **Environment** = SUMO traffic simulation
- **State** = 14 numbers describing the intersection right now (lines 134-144)
- **Action** = which direction gets green light (line 131)
- **Reward** = negative wait time — less waiting = better (line 215)

The agent runs through the full simulation repeatedly (20 episodes, line 235), and each time it gets a little smarter because it updates its brain (the neural network) based on what worked and what didn't.

## 2. Neural Network (the "brain") — lines 33-58

The `ActorCritic` class is a **multi-layer perceptron** (MLP) built in PyTorch:

- **Shared layers** (lines 36-41): Two fully-connected layers with 64 neurons each, using Tanh activation. These extract features from the 14-value state — learning things like "lots of cars waiting on the north lane while east is empty"
- **Actor head** (line 42): A linear layer that outputs 2 values (one per action). These become a probability distribution — "70% chance NS green is better, 30% chance EW green is better"
- **Critic head** (line 43): A linear layer that outputs 1 value — an estimate of "how good is this situation overall?" This is the **value function**

The actor and critic sharing layers is key — features useful for deciding what to do are also useful for evaluating the situation.

## 3. Policy Gradient / Actor (action selection) — lines 49-53

`get_action()` implements a **stochastic policy**:

- Takes the actor's raw outputs (logits) and creates a `Categorical` probability distribution (line 51)
- **Samples** an action from that distribution (line 52) rather than always picking the best one — this is **exploration**. Early on, the agent tries random things. As it learns, the probabilities sharpen toward the better action
- Returns the action, its log-probability (needed for PPO math later), and the critic's value estimate

## 4. Temporal Difference / Return Computation — lines 90-94

```python
R = r + GAMMA * R * (1 - d)
```

This is the **discounted return** calculation. It answers: "what's the total future reward from this point?"

- `GAMMA = 0.99` means the agent cares about future rewards but slightly less than immediate ones
- It's computed **backwards** through time — start from the end and accumulate
- `(1 - d)` resets the return at episode boundaries

## 5. Advantage Estimation — lines 102-103

```python
advantages = returns - old_values
```

The **advantage** tells the agent: "was this action **better or worse than expected?**"

- If advantage > 0: this action was better than the critic predicted — reinforce it
- If advantage < 0: this action was worse than expected — discourage it
- Normalization (line 103) stabilizes training by keeping advantage values centered around zero

## 6. PPO (Proximal Policy Optimization) — lines 105-121

This is the core learning algorithm, inside the `update()` method.

**The clipped surrogate objective** (lines 108-112):

```python
ratio = (new_log_probs - old_log_probs).exp()
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-eps, 1+eps) * advantages
loss = -min(surr1, surr2)
```

- `ratio` measures how much the policy changed since collecting the data
- The **clamp** (line 110, `CLIP_EPS = 0.2`) is PPO's key innovation — it prevents the policy from changing too drastically in one update, keeping learning stable
- Without clipping, one bad update could destroy everything the agent learned

**Three loss components** (line 116):

- **Actor loss**: push the policy toward actions with positive advantage
- **Critic loss** (line 113): train the value function to better predict returns
- **Entropy bonus** (line 114): reward uncertainty in the policy, encouraging continued exploration

**Gradient clipping** (line 120): caps how large any single update can be — another stability measure.

The agent does **4 passes** (PPO_EPOCHS) over the same batch of experience before collecting new data (line 105).

## 7. Experience Collection — lines 78-84 and 191-194

The `store()` method saves each (state, action, log_prob, reward, value, done) tuple. This is the agent's **replay memory** — it accumulates a full episode of experience, then learns from it all at once in `update()`, then clears it (lines 123-128). This is **on-policy** learning: only learn from data collected by the current policy.

## Summary

| Concept | Where | What it does |
|---|---|---|
| RL loop | `run_episode()` | observe-act-reward cycle |
| Neural network | `ActorCritic` class | learns state → action mapping |
| Policy gradient | `get_action()` | probabilistic action selection |
| Value function | critic head | estimates how good a state is |
| Discounted returns | `update()` lines 90-94 | computes total future reward |
| Advantage | `update()` lines 102-103 | "better or worse than expected?" |
| PPO clipping | `update()` lines 108-112 | stable policy updates |
| Entropy regularization | `update()` line 114 | encourages exploration |
