---
title: Chip Flooring Env
emoji: 🤖
sdk: docker
app_port: 8000
short_description: OpenEnv chip floorplanning benchmark
tags:
  - openenv
  - chip-flooring
  - placement
  - rl
---

# Chip Flooring Env

Chip Flooring Env is an OpenEnv benchmark for training and evaluating agents on a chip floorplanning task.
The agent must place rectangular blocks on a grid while respecting legality, minimizing congestion, and reducing wirelength-related cost.

This environment is designed for agents that must make **structured spatial decisions** under constraints, not just generate text or code.

## What This Environment Is For

The goal is to simulate a realistic physical-design workflow where an agent:

- interprets a placement problem,
- selects which block to place next,
- chooses a legal coordinate for that block,
- tracks remaining blocks and current placement quality,
- improves the layout over an episode.

This is useful for:

- placement agents in chip design pipelines,
- research on constraint-aware decision making,
- reinforcement learning and agentic planning,
- benchmarking agents on sequential optimization tasks.

## Why This Matters

Large chip design workflows need more than a generic coding assistant.
They need an agent that can:

- reason over a stateful layout,
- act under geometry and legality constraints,
- balance local and global objectives,
- adapt across different task difficulties.

Chip Flooring Env gives that kind of environment in a compact, testable form.

## Environment Design

The environment exposes a standard OpenEnv-style interface:

- `reset()` initializes a new episode
- `step(action)` applies one placement action
- `state` returns the current episode state
 

### Observation

The observation is built to help the agent understand the current placement state.
It includes:

- grid representation,
- remaining blocks,
- placed blocks,
- connectivity summaries,
- candidate placement positions,
- density map,
- current HPWL-style cost,
- change in cost from the last move,
- task name,
- invalid action reason if applicable.

### Action

The action is a placement request with:

- `x`
- `y`
- `choosen_block_index`

The agent chooses a block from the remaining set and places it at a coordinate on the grid.

### Reward

The reward is shaped to encourage:

- legal placements,
- progress toward completion,
- better local structure,
- lower wirelength / HPWL-style cost,
- fewer invalid actions.

Edges can also carry a `criticality` value, which increases the reward pressure on more important connections.
That lets the task configs express both raw wirelength and connection importance.

The reward is intentionally non-binary so the agent gets feedback throughout the episode.
 
Each task has:

- a separate block set,
- a separate grid size,
- a separate grader,
- a stricter optimization burden as difficulty increases.

This gives a progression from easier placement reasoning to more demanding floorplanning decisions.

 

## Who This Is For

This benchmark is a better fit for agents that work like:

- physical design assistants,
- placement planners,
- layout optimization agents,
- chip-structure reasoning systems.

It is not the same as a normal coding agent or DevOps agent.
 

## Suggested Use

Use this environment if you want to benchmark:

- placement quality,
- legal action selection,
- progress over an episode,
- multi-step planning under constraints,
- behavior across easy, medium, and hard variants.

## Local Development

Typical local flow:

1. git clone https://github.com/Gopinathv19/chip_flooring_env.git
2. cd chip_flooring_env
3. python3 -m venv .venv
4. source .venv/source/activate
4. pip install openenv-core
5. pip install uv
6. uv sync
7. openenv validate
8. uv run inference.py


 

## Project Structure

- `server/app.py` - FastAPI app and task/grader endpoints
- `server/chip_flooring_env_environment.py` - environment logic
- `server/graders.py` - task graders
- `models.py` - typed action/observation/state models
- `inference.py` - baseline agent runner
- `client.py` - OpenEnv client wrapper
- `openenv.yaml` - benchmark metadata and task definitions

## Summary

Chip Flooring Env is a compact but realistic benchmark for chip placement agents.
It is meant to test whether an agent can plan, place, and improve layouts under constraints across multiple difficulty levels.

If you are building an agent for chip design or physical layout optimization, this environment gives you a clean place to measure that capability.
