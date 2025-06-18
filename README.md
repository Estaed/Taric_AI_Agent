# Taric AI Agent (IL + RL) (`taric_ai_agent`)

An AI agent for playing Taric in League of Legends using a combination of Imitation Learning (IL) from live game expert data and Reinforcement Learning (RL) within a custom simulation environment.

## Project Overview

This project develops a high-performing AI agent for playing Taric in a 2v2 laning scenario. The agent is pre-trained using Imitation Learning from human expert data collected from the live League of Legends game, and then further refined using Reinforcement Learning within the `lol_sim_env` simulation environment.

## Key Features

- **Imitation Learning Pipeline**: Collect and process expert gameplay data from live LoL games
- **Live Data Collection**: LCU API integration, screen recording, and input logging
- **State/Action Mapping**: Critical mapping between live game state and simulation environment
- **Vision Processing**: Screen analysis to extract game state information
- **RL Training**: Stable Baselines3 integration for reinforcement learning
- **Cloud Training Ready**: Designed for GPU-accelerated training on cloud platforms

## Architecture Overview

### Phase 1: Imitation Learning (IL)
1. **Data Collection**: Capture expert Taric gameplay from live LoL games
2. **Data Processing**: Process raw data into simulation-compatible format
3. **Model Training**: Train neural network to mimic expert actions

### Phase 2: Reinforcement Learning (RL)
1. **Policy Initialization**: Start with IL-trained policy
2. **Environment Training**: Train in custom simulation environment
3. **Performance Optimization**: Refine policy through RL

## Core Components

- **Live Data Collector**: Captures LCU data, screen frames, and player inputs
- **Vision Processor**: Extracts game state from screen recordings
- **State/Action Mapper**: Maps live game data to simulation environment format
- **IL Policy Network**: Neural network for action prediction
- **RL Agent**: Stable Baselines3 PPO agent for environment training

## Dependencies

This project critically depends on the **LoL Simulated Laning Environment** (`lol_sim_env`) project, which provides the training environment for the RL phase.

## Documentation

- **[Project Specification](taric_ai_agent.md)**: Complete project requirements, features, and detailed implementation plan
- **[Architecture Overview](Architecture_agent.md)**: File structure and architectural design

## Development Status

🚧 **Under Development** - This project is currently in active development with an iterative MVP-first approach, prioritizing state/action mapping feasibility.

## Related Projects

This agent is designed to be trained using the **LoL Simulated Laning Environment** (`lol_sim_env`) project for the reinforcement learning phase.

---

**Version**: 1.1  
**Last Updated**: December 2024 