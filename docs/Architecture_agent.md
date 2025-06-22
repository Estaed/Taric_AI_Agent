# Architecture & File Structures

This document outlines the proposed file and folder structures for the main projects:
1.  `lol_sim_env`: The Python-based simulated 2v2 LoL laning environment for RL training (uses MCP server for champion/game data)
2.  `taric_ai_agent`: The Taric AI agent, using IL from MCP-provided datasets and RL within `lol_sim_env`
3.  `lol_data_mcp_server`: Provides data for BOTH simulation environment AND IL training datasets

## Project 2: Taric AI Agent (`taric_ai_agent`)

**Goal:** To develop a Taric AI agent using **Imitation Learning (IL) from MCP-provided ready-to-train datasets** and **Reinforcement Learning (RL) within the `lol_sim_env`**. The MCP server provides sophisticated training data for IL and also supplies game data to the simulation environment.

**Updated Folder Structure (MCP-Enhanced):**

```
taric_ai_agent/
├── taric_ai_agent/                    # Core Python package
│   ├── __init__.py
│   ├── agents/                        # Agent model definitions
│   │   ├── __init__.py
│   │   ├── base_agent.py              # Abstract base class for agents
│   │   ├── il_policy_network.py       # IL neural network (enhanced for MCP features)
│   │   └── rl_agent_sb3.py           # RL agent setup using Stable Baselines3
│   ├── mcp/                          # MCP integration (NEW - PRIMARY for IL)
│   │   ├── __init__.py
│   │   ├── mcp_client.py             # MCP protocol client for LoL Data server
│   │   ├── dataset_fetcher.py        # Fetch ready-to-train IL datasets via MCP tools
│   │   ├── data_converter.py         # Convert MCP data to training-ready format
│   │   └── live_services.py          # Real-time MCP services for meta updates
│   ├── il/                           # Imitation Learning (MCP-based, SIMPLIFIED)
│   │   ├── __init__.py
│   │   ├── dataset_processor.py      # Process MCP datasets for training
│   │   ├── data_validator.py         # Validate MCP data quality
│   │   ├── augmentation.py           # Data augmentation techniques
│   │   └── il_dataset.py            # PyTorch Dataset class for MCP IL data
│   ├── rl/                           # Reinforcement Learning (Sim Env based)
│   │   ├── __init__.py
│   │   ├── rl_trainer.py            # RL training in lol_sim_env
│   │   ├── reward_functions.py       # Custom reward functions for Taric
│   │   └── env_interface.py          # Interface with lol_sim_env
│   ├── training/                     # Training pipelines (NEW)
│   │   ├── __init__.py
│   │   ├── il_trainer.py            # IL training loop using MCP data
│   │   ├── combined_trainer.py      # IL → RL pipeline trainer
│   │   ├── metrics_tracker.py       # Training metrics and validation
│   │   └── model_checkpointing.py   # Model saving/loading
│   ├── simulation/                   # Simulation integration
│   │   ├── __init__.py
│   │   ├── state_mapper.py          # Map between MCP data and sim env states
│   │   ├── action_mapper.py         # Map between IL actions and sim actions
│   │   └── env_integration.py       # Integration utilities with lol_sim_env
│   ├── evaluation/                   # Evaluation framework (NEW)
│   │   ├── __init__.py
│   │   ├── il_evaluator.py          # Evaluate IL performance
│   │   ├── rl_evaluator.py          # Evaluate RL performance
│   │   ├── combined_evaluator.py    # Evaluate IL+RL pipeline
│   │   ├── metrics.py               # Performance metrics
│   │   └── benchmarking.py          # Player comparison via MCP
│   ├── common/                       # Shared utilities
│   │   ├── __init__.py
│   │   ├── data_manager.py          # Dataset management and versioning
│   │   ├── config_manager.py        # Configuration management
│   │   ├── state_action_mapper.py   # Legacy mapping utilities
│   │   └── utils.py                 # General utility functions
│   └── utils/                        # Additional utilities
│       ├── __init__.py
│       ├── logger.py                # Logging utilities
│       ├── visualization.py         # Training visualizations
│       └── training_utils.py        # ML training utilities
├── scripts/                          # Execution scripts
│   ├── fetch_training_data.py       # Fetch IL datasets from MCP (NEW)
│   ├── train_il_model.py           # Train IL model using MCP data (UPDATED)
│   ├── train_rl_agent.py           # Train RL agent in lol_sim_env
│   ├── train_combined.py           # IL → RL combined training pipeline (NEW)
│   ├── evaluate_agent.py           # Comprehensive agent evaluation (NEW)
│   ├── benchmark_vs_players.py     # Compare against high-ELO players via MCP (NEW)
│   └── simulate_agent_performance.py # Test agent in simulation environment (NEW)
├── configs/                         # Configuration files
│   ├── mcp_config.yaml             # MCP server connection settings (NEW)
│   ├── il_training_config.yaml     # IL training parameters (UPDATED)
│   ├── rl_training_config.yaml     # RL training parameters
│   ├── combined_training_config.yaml # IL → RL pipeline config (NEW)
│   ├── model_config.yaml           # Model architectures
│   ├── evaluation_config.yaml      # Evaluation settings (NEW)
│   └── simulation_config.yaml      # Simulation environment settings (NEW)
├── data/                           # Data storage
│   ├── mcp_datasets/              # Downloaded MCP training datasets (NEW)
│   │   ├── taric_estaed_dataset_v1.pt    # Player-specific datasets
│   │   ├── taric_scenarios_v1.json       # Scenario-labeled data
│   │   └── taric_meta_builds_v1.json     # Meta information
│   ├── processed/                  # Processed training data
│   │   ├── il_training_data/       # IL training splits
│   │   └── rl_interaction_data/    # RL environment interaction logs
│   ├── evaluation_results/         # Performance evaluation results (NEW)
│   │   ├── il_performance/         # IL model evaluation
│   │   ├── rl_performance/         # RL agent evaluation
│   │   └── combined_performance/   # IL+RL pipeline evaluation
│   └── cache/                      # Cached data and temporary files
├── trained_models/                 # Model artifacts
│   ├── il_models/                  # IL trained models
│   │   ├── taric_il_v1.pth        # Checkpoints from MCP data training
│   │   └── taric_il_best.pth      # Best performing IL model
│   ├── rl_models/                  # RL trained models
│   │   ├── taric_rl_v1.zip        # SB3 RL models
│   │   └── taric_rl_best.zip      # Best performing RL model
│   ├── combined_models/            # IL+RL combined models (NEW)
│   │   └── taric_combined_v1.zip   # Best IL+RL pipeline model
│   └── checkpoints/                # Training checkpoints
├── notebooks/                      # Analysis notebooks
│   ├── mcp_data_exploration.ipynb  # Explore MCP datasets (NEW)
│   ├── il_training_analysis.ipynb  # IL training performance analysis
│   ├── rl_training_analysis.ipynb  # RL training performance analysis
│   ├── player_comparison.ipynb     # Compare agent to high-ELO players (NEW)
│   ├── simulation_testing.ipynb    # Test agent in simulation environment
│   └── combined_pipeline_analysis.ipynb # Analyze IL+RL pipeline (NEW)
├── tests/                          # Test suite
│   ├── test_mcp_integration.py     # Test MCP client functionality (NEW)
│   ├── test_il_training.py         # Test IL training pipeline
│   ├── test_rl_training.py         # Test RL training pipeline
│   ├── test_simulation_integration.py # Test lol_sim_env integration
│   ├── test_evaluation.py          # Test evaluation framework (NEW)
│   └── test_combined_pipeline.py   # Test IL+RL pipeline (NEW)
├── requirements.txt                # Dependencies (mcp-client, torch, sb3, etc.)
└── README.md                       # Project overview, MCP+Simulation setup
```

## Key Architecture Changes for MCP Integration

### 🎯 **Dual Data Flow Architecture**

```
MCP Server (lol_data_mcp_server)
    ├── Provides to lol_sim_env: Champion stats, abilities, items, game mechanics
    └── Provides to taric_ai_agent: Ready-to-train IL datasets, scenarios, player demos

IL Training Path (NEW):
MCP Datasets → IL Model → Performance Evaluation

RL Training Path (EXISTING + ENHANCED):
IL Model → Initialize RL Agent → Train in lol_sim_env → Final Agent

Combined Pipeline:
MCP Data → IL Training → RL Training (initialized with IL) → Expert Agent
```

### 🔄 **MCP Integration Points**

1. **For Simulation Environment Data:**
   - Champion abilities and stats
   - Item information and build paths
   - Game mechanics and formulas
   - Meta builds and trends

2. **For IL Training Data:**
   - Ready-to-train state-action pairs
   - High-ELO player demonstrations
   - Scenario-labeled training data
   - Enhanced features (positioning, combat metrics)

### 📊 **Data Management Strategy**

- **MCP Datasets** (`data/mcp_datasets/`): Raw and processed datasets from MCP server
- **IL Training Data** (`data/processed/il_training_data/`): Prepared data for IL model training
- **RL Interaction Data** (`data/processed/rl_interaction_data/`): Environment interaction logs
- **Evaluation Results** (`data/evaluation_results/`): Performance analysis across both IL and RL

### 🏗️ **Training Pipeline Architecture**

#### **Phase 1: IL Training (MCP-Powered)**
```python
# fetch_training_data.py
mcp_client.get_imitation_dataset(champion="Taric", players=["Estaed#TAR"])

# train_il_model.py  
il_trainer.train(mcp_dataset, validation_split=0.2)
```

#### **Phase 2: RL Training (Simulation-Powered)**
```python
# train_rl_agent.py
rl_agent = PPO(policy=il_model.to_sb3_policy(), env=lol_sim_env)
rl_agent.learn(total_timesteps=1000000)
```

#### **Phase 3: Combined Evaluation**
```python
# evaluate_agent.py
il_performance = evaluate_il_model(il_model, mcp_benchmark_data)
rl_performance = evaluate_rl_agent(rl_agent, lol_sim_env)
combined_score = evaluate_combined_pipeline(il_model, rl_agent)
```

## Integration with Other Projects

### **LoL Data MCP Server** (Data Provider)
- **Serves IL Training Data**: Ready-to-train datasets, player demonstrations, scenarios
- **Serves Simulation Data**: Champion stats, abilities, items for accurate simulation
- **Serves Meta Data**: Current builds, trends, performance benchmarks

### **LoL Simulation Environment** (RL Training Platform)
- **Receives Champion Data**: From MCP server for accurate game simulation
- **Provides RL Training**: Environment for reinforcement learning phase
- **Validates IL Models**: Test IL performance in simulated environment

### **Taric AI Agent** (Consumer & Trainer)
- **Consumes MCP Data**: For IL training and meta analysis
- **Consumes Simulation**: For RL training and evaluation
- **Produces Trained Agent**: High-performing Taric AI using both IL and RL

This architecture enables the agent to leverage the best of both approaches: sophisticated IL training from real expert data via MCP, and robust RL training in a controlled simulation environment.