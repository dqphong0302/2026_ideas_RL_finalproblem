# Assignment Brief - Microgrid Energy Optimization using Deep Reinforcement Learning

## Assignment Overview

Energy distribution in microgrids is a complex optimization problem involving dynamic energy demand, variable renewable generation, storage constraints, and cost minimization. Traditional rule-based scheduling or linear optimization often fails to adapt in real time to uncertainties such as sudden demand spikes, fluctuating solar/wind generation, or battery degradation.

**Objective**: Implement a Deep Reinforcement Learning (DQN or Policy Gradient) agent to optimize energy dispatch in a microgrid. The agent will decide how much energy to store, release, or purchase from the main grid at every time step, balancing cost efficiency, reliability, and renewable usage.

This assignment will test your ability to:
- Model real-world problems as a Markov Decision Process (MDP)
- Implement state-of-the-art RL algorithms
- Analyze optimization performance and limitations
- Critically evaluate deployment, ethics, and future enhancements

---

## Problem Scenario

The increasing integration of renewable energy sources into modern power systems has created significant challenges in energy distribution and optimization. In particular, microgrids, which are small-scale, localized power networks, must balance energy generation from solar panels and wind turbines with fluctuating consumer demand while maintaining reliability and minimizing costs.

The microgrid must decide, at every time step:
- How to allocate available energy to meet consumer demand
- How much excess energy to store in batteries
- When it is cost-effective to purchase energy from the main grid

The primary goal of this system is to maximize the use of renewable energy, minimize electricity costs, and avoid power shortages.

---

## Assignment Tasks and Structure

| Section | Weight | Details / Requirements | Key Deliverables |
|---------|--------|------------------------|------------------|
| **1. Problem Description** | 15% | Explain the microgrid system, including energy sources, storage, and consumer demand. Discuss why Reinforcement Learning is more effective than rule-based scheduling. | - Why energy distribution is a sequential decision problem<br>- Limitations of conventional optimization methods<br>- Real-world relevance and impact |
| **2. Environment Modelling as an MDP** | 20% | Translate the microgrid into a formal Markov Decision Process | - MDP diagram<br>- Detailed explanation of states, actions, rewards, transitions<br>- Justification for modeling choices |
| **3. RL Algorithm Selection and Implementation** | 25% | Choose an algorithm (DQN or Policy Gradient). Explain suitability for high-dimensional continuous state space. | - Well-commented Python code in PyTorch or TensorFlow |
| **4. AI Optimization Analysis** | 15% | Discuss how RL agent optimizes energy dispatch. Analyze reward trends, learning convergence, and policy efficiency. | - Critical analysis of optimization performance and efficiency |
| **5. Results and Evaluation** | 15% | Present graphs of cumulative reward, daily cost savings, renewable usage ratio, unmet demand frequency. | - Graphs, charts, tables<br>- Comparative analysis with expected outcomes |
| **6. Ethical, Practical, and Future Considerations** | 10% | Discuss ethical concerns, practical issues, future enhancements | - Recommendations for ethical deployment and system improvements |

---

## MDP Specifications (From Assignment)

### State Space
- Battery level
- Demand
- Renewable generation
- Previous actions

### Action Space
- Energy draw from grid
- Charge/discharge battery
- Allocate energy to loads

### Reward Function
- **Positive reward**: Meeting demand with renewable energy
- **Negative reward**: Grid energy purchase
- **Penalties**: Unmet demand

### Transition Dynamics
- Battery updates
- Stochastic renewable generation
- Probabilistic demand

### Episode Termination
- End of day/week
- Critical battery failure
- Unmet demand threshold

---

## RL Algorithm Requirements

- Choose DQN or Policy Gradient
- Discuss exploration vs exploitation strategy (ε-greedy or entropy-based)
- Network architecture (layers, neurons, activation functions)
- Hyperparameters (learning rate, discount factor, batch size, replay memory)
- Training process (episodes, convergence criteria)

---

## Learning Outcomes Assessed

- Understanding the fundamentals of Reinforcement Learning (RL)
- Ability to model real-world problems as Markov Decision Processes (MDPs)
- Competence in implementing deep RL algorithms effectively
- Capacity for critical analysis of optimization efficiency, convergence, and performance trade-offs
- Awareness of ethical considerations and practical deployment issues in AI optimization

---

## Academic Integrity

Plagiarism is strictly prohibited. All sources must be properly cited using APA or IEEE referencing style. While AI tools may be used for learning support, students must not rely on them to generate final solutions. All submitted work must reflect the student's own understanding, analysis, and coding efforts.

---

## Submission Notes

- Late submissions may incur penalties in accordance with institutional policy
- Report should include figures, tables, and code snippets to enhance explanations and support analysis
- Submissions should be clear, well-structured, and technically accurate
