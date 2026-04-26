---
title: Chip Flooring Env
emoji: "🚀"
sdk: docker
app_port: 8000
short_description: OpenEnv chip floorplanning benchmark
tags:
  - openenv
  - chip-flooring
  - placement
  - rl
---

# ChipMind-LH: A Long-Horizon Environment for Chip Floorplanning


## ChipMind-LH is a long-horizon, constraint-evolving environment built for the EDA community, enabling chip design agents to learn adaptive planning, iterative refinement, and decision-making under realistic design pressures.

## [1] Environment Overview
ChipMind-LH is a long-horizon chip design environment where agents must adapt to hidden constraints, repair suboptimal layouts, and commit irreversible decisions under pressure.

It is designed to simulate the real workflow of chip floorplanning, where:
- Early decisions impact future feasibility
- Constraints are not fully visible initially
- Iterative refinement is necessary
- Final decisions must be made with confidence

In this environment, agents are not just placing blocks - they are learning to plan, adapt, and optimize over time.

---

## [2] Workflow of the Environment
The workflow is designed to push agents toward realistic chip floorplanning behavior:

**Rough Placement -> Iterative Refinement -> Final Stabilization**

Instead of solving the problem in one step, the agent must:
- Build an initial layout
- Adapt when new constraints appear
- Repair earlier mistakes
- Converge to a stable and optimized solution

This transforms the task into a continuous decision-making process, not a one-shot optimization.

---

## [3] Why Long-Horizon Planning is the Core of This Environment
This environment is built around the **Long-Horizon Planning & Instruction Following** theme.

Unlike short-horizon tasks where each action has immediate payoff, chip floorplanning is:
- Sequential
- Interdependent
- Delayed in feedback

Early decisions directly constrain or enable future possibilities, making the problem unsuitable for greedy strategies.

---

## [4] How Long-Horizon Structure is Embedded

### 1. Multi-Phase Execution (Temporal Decomposition)
The task is divided into three phases:
- **Placement Phase** -> Build initial layout with partial information
- **Repair Phase** -> Hidden constraints are revealed
- **Finalize Phase** -> Lock decisions and stabilize layout

This forces the agent to:
- Make provisional decisions early
- Revisit and revise those decisions later
- Transition from exploration -> correction -> commitment

Note: A single-pass strategy will fail in this environment.

---

### 2. Delayed and Evolving Feedback
The reward structure is intentionally non-myopic:
- Early placements may appear optimal but become suboptimal later
- Wirelength (HPWL) optimization emerges only after multiple placements
- Hidden constraints introduce non-stationarity mid-episode

Note: The agent must anticipate future consequences, not just optimize immediate reward.

---

### 3. Action Irreversibility and Commitment
In later stages:
- The agent must commit to placements
- Movement becomes restricted
- Mistakes become increasingly costly

Note: This creates decision pressure, forcing the agent to learn:
- When to continue exploring
- When to stabilize decisions

---

## [5] Why Long-Horizon Planning is Essential
This environment inherently requires:
- Tracking state across many steps
- Revising earlier decisions
- Handling delayed rewards
- Planning under evolving constraints

These are defining characteristics of long-horizon reasoning problems.

---

## [6] How This Improves Agent Learning

### 1. Temporal Credit Assignment
Understanding how early actions influence future outcomes

### 2. Iterative Refinement
Learning to:
Build -> Evaluate -> Adjust -> Converge

### 3. Robust Planning Under Uncertainty
Adapting to hidden constraints and recovering from mistakes

### 4. Strategic Commitment
Learning when to stop exploring and finalize decisions

---

## [7] Key Characteristics of the Environment

| Standard Environments | ChipMind-LH |
|----------------------|------------|
| Short-horizon        | Long-horizon |
| Fully observable     | Partially observable |
| Immediate rewards    | Delayed rewards |
| One-shot optimization| Iterative refinement |

Note: This makes it a strong benchmark for real decision-making capability.

---

## [8] How This Benefits the EDA Community
ChipMind-LH provides a research-ready environment for developing next-generation AI systems in chip design.

### Key Contributions
- Moves beyond static benchmarks to dynamic, realistic simulation
- Enables research on adaptive and self-correcting agents
- Provides a testbed for long-horizon reinforcement learning in EDA
- Bridges the gap between academic RL setups and real-world chip workflows

This environment can support:
- RL-based floorplanning research
- Adaptive placement strategies
- Constraint-aware design automation
- Exploration of human-like iterative design behaviors

### Who Can Benefit From This?
### 1. AI/ML Researchers
- Study long-horizon decision making
- Develop agents that handle delayed feedback and evolving constraints

### 2. EDA Researchers and Engineers
- Prototype intelligent placement and optimization strategies
- Simulate realistic design workflows

### 3. Semiconductor Industry Practitioners
- Explore AI-assisted design tools
- Evaluate adaptive placement approaches

---

## [9] Key Insight
This is not a placement problem—it is a planning problem disguised as placement.

The agent is not rewarded for acting fast, but for:
- Thinking ahead
- Adapting to change
- Converging to a stable solution

That is why the Long-Horizon Planning theme is fundamental to this environment.

---
