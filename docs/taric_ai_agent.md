# Project 2 PRD: Taric AI Agent (IL + RL) (`taric_ai_agent`)

**Version:** 2.0
**Date:** December 2024
**Project Goal:** To develop a high-performing AI agent for playing Taric in a 2v2 laning scenario. The agent will be pre-trained using **Imitation Learning from MCP-provided ready-to-train datasets** from sophisticated high-ELO player analysis, and then further refined using **Reinforcement Learning within the `lol_sim_env`**.

---

## I. Vision & Scope

* **Problem:** Training an RL agent for Taric from scratch is sample-inefficient. Manual data collection and processing is extremely complex (as shown by previous 15MB per game preprocessing pipelines).
* **Solution:** **MCP-Powered IL + Simulation-Powered RL**
    1.  **IL via MCP:** Use LoL Data MCP Server to get **ready-to-train datasets** from high-ELO Taric players (like Estaed#TAR) with sophisticated preprocessing (state-action pairs, enhanced features, scenario labels).
    2.  **RL via Simulation:** Initialize an RL agent with the IL policy and train it in the `lol_sim_env` for robust performance and additional learning.
    3.  **MCP serves BOTH projects:** Provides champion/game data to `lol_sim_env` for accurate simulation AND provides ready-to-train IL datasets to `taric_ai_agent`.
* **Dependencies:** 
    - **[LoL Data MCP Server](../Lol_Data_MCP_Server/)**: Primary dependency for **both IL training datasets AND simulation environment data**
    - **[LoL Simulation Environment](../Lol_Sim_Env/)**: For RL training and evaluation
* **Development Philosophy:** **Dual-approach strategy leveraging MCP for both projects.** Use MCP for sophisticated IL training to get strong initial policy, then use RL in simulation for robust performance and continued learning. **M2.0 (MCP Integration & IL Training) is the highest initial priority.** Parallel development ensuring MCP serves data needs of both simulation environment (champion stats, abilities) and AI agent training (IL datasets).

---

## I.1. MCP Server's Dual Role

### 📊 **For LoL Simulation Environment**
The MCP server provides essential data to make the simulation environment accurate and up-to-date:
- **Champion Stats & Abilities**: Current patch champion data for accurate simulation
- **Item Information**: Complete item stats and interactions
- **Game Mechanics**: Formulas for damage, cooldowns, etc.
- **Meta Builds**: Current optimal builds for realistic opponent modeling

### 🎯 **For Taric AI Agent (IL Training)**
The MCP server provides sophisticated, ready-to-train datasets equivalent to previous 15MB per game processing:
- **State-Action Pairs**: Preprocessed training data from high-ELO players
- **Enhanced Features**: Positioning, combat metrics, decision context
- **Scenario Labels**: Team fights, clutch saves, ability combos (40+ scenarios)
- **Player Demonstrations**: Specific high-ELO player data (e.g., Estaed#TAR)

### 🔄 **Combined Workflow**
```
MCP Server
├── Serves lol_sim_env: Champion data, abilities, items → Accurate simulation
└── Serves taric_ai_agent: Ready-to-train IL datasets → Quick training start

Training Pipeline:
MCP IL Datasets → IL Model → Initialize RL Agent → Train in Simulation → Expert Agent
```

---

## II. Requirements & Features (R-Sections)

This section outlines target functionality leveraging MCP for both IL data and simulation environment data.

### R1: Imitation Learning (IL) via MCP (PRIMARY APPROACH)
* R1.1: MCP-Based IL Data Pipeline (`taric_ai_agent/mcp/`):
    * R1.1.1: `mcp_client.py`: MCP protocol client for LoL Data server
    * R1.1.2: `dataset_fetcher.py`: Fetch ready-to-train IL datasets via MCP tools:
        - `get_imitation_dataset()` for complete Taric datasets
        - `get_player_demonstrations()` for specific players (e.g., Estaed#TAR)
        - `get_scenario_training_data()` for scenario-labeled data
    * R1.1.3: `data_converter.py`: Convert MCP data to training-ready format
* R1.2: IL Data Processing (`taric_ai_agent/il/`):
    * R1.2.1: `dataset_processor.py`: Process MCP datasets for specific training requirements
    * R1.2.2: `data_validator.py`: Validate MCP data quality and completeness
    * R1.2.3: **Enhanced State/Action Mapping** using MCP data for simulation compatibility
* R1.3: IL Model Training (`taric_ai_agent/agents/` & `training/`):
    * R1.3.1: Enhanced IL neural network leveraging MCP's rich features (positioning, combat metrics)
    * R1.3.2: `il_trainer.py`: Supervised learning training loop for MCP datasets
    * R1.3.3: Model checkpointing and evaluation against MCP benchmarks

### R1.X: Alternative IL from Live Game Data (LEGACY/FALLBACK)
* R1.X.1: Live Data Collection System (maintained for edge cases or additional data collection)
* R1.X.2: Traditional processing pipeline (retained but de-prioritized in favor of MCP approach)

### R2: Reinforcement Learning (RL) using `lol_sim_env` (ENHANCED WITH MCP)
* R2.1: RL Agent Framework (`taric_ai_agent/agents/rl_agent_sb3.py`): Stable Baselines3 PPO
* R2.2: **IL Policy Initialization**: Initialize RL agent with MCP-trained IL policy for strong starting point
* R2.3: **MCP-Enhanced Simulation**: `lol_sim_env` uses MCP server for:
    - Accurate champion stats and abilities from current patch
    - Real-time item data and interactions  
    - Meta-aware opponent modeling
* R2.4: RL Training Pipeline (`scripts/train_rl_agent.py`):
    - Cloud-execution ready training loop
    - Dynamic adaptation to patch changes via MCP
    - Performance monitoring against MCP benchmarks
* R2.5: **Combined IL+RL Evaluation** (`scripts/evaluate_combined_agent.py`)

### R3: Agent Evaluation
* R3.1: `scripts/evaluate_agent_in_sim.py` for IL-only and IL+RL agents in `lol_sim_env`.
* R3.2: Metrics: Episode outcomes in sim (KDA, gold/XP diff, etc.), ability usage stats.

---

## III. Proposed Folder Structure & Key Files
(As defined in `architecture.md` for Project 2)

---
## IV. Detailed Implementation Plan (Task-Oriented for AI Assistant)

**Overall Strategy:** **MCP-First Development.** Prioritize MCP integration for IL training data, then integrate with simulation environment for RL. Focus on leveraging sophisticated preprocessing via MCP rather than rebuilding complex data collection pipelines.

**Phase 0: Project Setup & MCP Integration**

* **Task 0.1: Initialize Enhanced Project Structure for MCP Integration**
    * **Objective:** Set up project structure optimized for MCP-based IL and simulation-based RL
    * **File(s):** `taric_ai_agent/` (root folder), `requirements.txt`
    * **Instructions for AI:**
        1.  "Create root directory `taric_ai_agent/`."
        2.  "Initialize Git repository."
        3.  "Create enhanced `requirements.txt` with MCP and RL dependencies: `numpy gymnasium pyyaml torch stable-baselines3 mcp-client requests pydantic asyncio`"
        4.  "Add optional dependencies for legacy approaches: `opencv-python pynput psutil` (commented as optional)"
* **Task 0.2: Create MCP-Enhanced Package Structure**
    * **Objective:** Set up package structure as defined in `Architecture_agent.md`
    * **File(s):** Create directory structure per Architecture_agent.md
    * **Instructions for AI:**
        1.  Create `taric_ai_agent/` package with `__init__.py`
        2.  Create primary directories: `agents/`, `mcp/`, `il/`, `rl/`, `training/`, `simulation/`, `evaluation/`, `common/`, `utils/`
        3.  Create script directories: `scripts/`, `configs/`, `data/`, `trained_models/`, `notebooks/`, `tests/`
        4.  Create MCP-specific data directories: `data/mcp_datasets/`, `data/processed/il_training_data/`, `data/evaluation_results/`
        5.  Add all necessary `__init__.py` files

**Phase 1: MCP Integration & Dataset Access (PRIORITY)**

* **Task 1.1: Implement MCP Client Foundation**
    * **Objective:** Create robust MCP client for LoL Data server communication
    * **File(s):** `taric_ai_agent/mcp/mcp_client.py`, `configs/mcp_config.yaml`
    * **Instructions for AI:**
        1.  "Create `MCPClient` class using mcp-client library for protocol communication"
        2.  "Implement connection management with error handling and reconnection logic"
        3.  "Add configuration file `mcp_config.yaml` with server settings (host, port, authentication)"
        4.  "Implement basic tool calling functionality with response validation"
        5.  "Add comprehensive logging for debugging and monitoring"
        6.  "Create connection testing and health check methods"
    * **Verification:** Can connect to MCP server and call basic tools successfully

* **Task 1.2: Implement Dataset Fetcher for IL Training**
    * **Objective:** Fetch ready-to-train datasets from MCP server leveraging sophisticated preprocessing
    * **File(s):** `taric_ai_agent/mcp/dataset_fetcher.py`
    * **Instructions for AI:**
        1.  "Create `DatasetFetcher` class using MCPClient"
        2.  "Implement `fetch_imitation_dataset()` using MCP `get_imitation_dataset` tool:"
        3.  "  - Request Taric datasets: champion='Taric', players=['Estaed#TAR'], min_rank='CHALLENGER'"
        4.  "  - Include enhanced features from sophisticated preprocessing (15MB per game equivalent)"
        5.  "Implement `fetch_player_demonstrations()` for specific high-ELO players"
        6.  "Implement `fetch_scenario_data()` for labeled scenarios (team fights, clutch saves, etc.)"
        7.  "Add data caching, progress tracking, and format conversion (JSON ↔ PyTorch ↔ NumPy)"
    * **Verification:** Can fetch sophisticated training datasets equivalent to previous 15MB per game processing

**Phase 2: IL Model Development & Training**

* **Task 2.1: Implement Data Processing Pipeline**
    * **Objective:** Process MCP datasets for IL training
    * **File(s):** `taric_ai_agent/il/dataset_processor.py`, `taric_ai_agent/il/data_validator.py`
    * **Instructions for AI:**
        1.  "Create `DatasetProcessor` class for MCP dataset preprocessing"
        2.  "Implement data validation and quality checks for training suitability"
        3.  "Add data filtering and cleaning for noisy samples"
        4.  "Implement train/validation/test splits with reproducible seeds"
        5.  "Add data augmentation pipeline for improved training"
        6.  "Create dataset statistics and analysis reporting"
    * **Verification:** Processes MCP data into training-ready format with quality validation

* **Task 2.2: Implement Enhanced IL Neural Network**
    * **Objective:** Create IL model leveraging MCP's rich features
    * **File(s):** `taric_ai_agent/agents/il_policy_network.py`
    * **Instructions for AI:**
        1.  "Design multi-layer neural network architecture for Taric-specific gameplay"
        2.  "Add support for enhanced features from MCP (positioning, combat metrics, decision context)"
        3.  "Implement scenario-aware training branches (team fights, healing, positioning)"
        4.  "Add attention mechanisms for important game state features"
        5.  "Implement dropout and regularization for generalization"
        6.  "Add model interpretability features for analysis"
    * **Verification:** Advanced neural network handles MCP's enhanced features effectively

* **Task 2.3: Implement IL Training Pipeline**
    * **Objective:** Create comprehensive training system
    * **File(s):** `taric_ai_agent/training/il_trainer.py`, `taric_ai_agent/training/metrics_tracker.py`
    * **Instructions for AI:**
        1.  "Create `ILTrainer` class with supervised learning loop"
        2.  "Implement dynamic loss functions weighted by scenario importance"
        3.  "Add learning rate scheduling and early stopping"
        4.  "Create comprehensive training metrics and validation"
        5.  "Add tensorboard integration for training visualization"
        6.  "Implement model checkpointing and recovery"
    * **Verification:** Trains IL model with comprehensive monitoring and validation

**Phase 3: Simulation Integration & RL Training**

* **Task 3.1: Implement Simulation Integration Interface**
    * **Objective:** Create interface between IL model and simulation environment
    * **File(s):** `taric_ai_agent/simulation/env_integration.py`, `taric_ai_agent/simulation/state_mapper.py`
    * **Instructions for AI:**
        1.  "Create `SimulationInterface` class for lol_sim_env integration"
        2.  "Implement state mapping between MCP training data format and simulation observation space"
        3.  "Add action mapping from IL model outputs to simulation action space"
        4.  "Create validation system to ensure compatibility between IL and simulation"
        5.  "Add performance monitoring and debugging tools"
        6.  "Implement graceful error handling for integration issues"
    * **Verification:** IL model can interact with simulation environment successfully

* **Task 3.2: Implement RL Training Pipeline**
    * **Objective:** Create RL training system initialized with IL policy
    * **File(s):** `taric_ai_agent/rl/rl_trainer.py`, `taric_ai_agent/agents/rl_agent_sb3.py`
    * **Instructions for AI:**
        1.  "Create `RLTrainer` class using Stable Baselines3 PPO"
        2.  "Implement IL policy initialization for RL agent"
        3.  "Add custom reward functions optimized for Taric support role"
        4.  "Create training loop with performance monitoring"
        5.  "Implement dynamic hyperparameter adjustment"
        6.  "Add checkpoint saving and model evaluation during training"
    * **Verification:** RL training improves upon IL baseline performance

* **Task 3.3: Implement Combined IL+RL Training Pipeline**
    * **Objective:** Create seamless IL → RL training workflow
    * **File(s):** `taric_ai_agent/training/combined_trainer.py`, `scripts/train_combined.py`
    * **Instructions for AI:**
        1.  "Create `CombinedTrainer` orchestrating IL then RL training"
        2.  "Implement automatic IL→RL transition based on performance metrics"
        3.  "Add comprehensive evaluation comparing IL-only vs IL+RL performance"
        4.  "Create training progress visualization and reporting"
        5.  "Implement early stopping and hyperparameter optimization"
        6.  "Add model comparison and selection tools"
    * **Verification:** Combined training produces superior agent performance

**Phase 4: Evaluation & Performance Analysis**

* **Task 4.1: Implement Comprehensive Evaluation Framework**
    * **Objective:** Create evaluation system for IL, RL, and combined models
    * **File(s):** `taric_ai_agent/evaluation/evaluator.py`, `scripts/evaluate_agent.py`
    * **Instructions for AI:**
        1.  "Create `AgentEvaluator` with multiple evaluation metrics"
        2.  "Implement simulation-based performance testing"
        3.  "Add scenario-specific evaluation (team fights, healing efficiency, positioning)"
        4.  "Create performance comparison with MCP benchmark data"
        5.  "Implement meta-adherence scoring using MCP meta analysis"
        6.  "Add decision accuracy analysis and gameplay pattern assessment"
        7.  "Create comprehensive evaluation reports and visualizations"
    * **Verification:** Provides detailed performance analysis across multiple metrics

* **Task 4.2: Implement Player Benchmarking System**
    * **Objective:** Compare agent performance to high-ELO players via MCP
    * **File(s):** `taric_ai_agent/evaluation/benchmarking.py`, `scripts/benchmark_vs_players.py`
    * **Instructions for AI:**
        1.  "Create `PlayerBenchmarking` system using MCP player data"
        2.  "Implement gameplay similarity scoring against specific players (e.g., Estaed#TAR)"
        3.  "Add decision pattern comparison and analysis"
        4.  "Create performance gap identification and improvement recommendations"
        5.  "Implement statistical significance testing for performance comparisons"
        6.  "Add visualizations for performance comparison and gap analysis"
        7.  "Create detailed benchmarking reports with actionable insights"
    * **Verification:** Provides meaningful comparison with human expert performance

**Phase 5: Production & Deployment**

* **Task 5.1: Implement Model Deployment Pipeline**
    * **Objective:** Create production-ready model serving system
    * **File(s):** `scripts/deploy_model.py`, production configuration files
    * **Instructions for AI:**
        1.  "Create model packaging and versioning system"
        2.  "Implement model serving API for real-time inference"
        3.  "Add monitoring and logging for production performance"
        4.  "Create automated testing and validation pipeline"
        5.  "Implement rollback and recovery mechanisms"
        6.  "Add performance optimization and caching"
    * **Verification:** Model can be deployed and serves predictions reliably

* **Task 5.2: Create Comprehensive Testing Suite**
    * **Objective:** Ensure system reliability and performance
    * **File(s):** `tests/` directory with comprehensive test coverage
    * **Instructions for AI:**
        1.  "Create unit tests for all major components"
        2.  "Implement integration tests for MCP client and simulation interface"
        3.  "Add performance tests for training and inference"
        4.  "Create regression tests to prevent performance degradation"
        5.  "Implement automated testing pipeline with CI/CD"
        6.  "Add stress testing for production deployment"
    * **Verification:** Comprehensive test coverage with automated testing pipeline

---
## V. Risks & Challenges (Project 2)
(As previously detailed, with M2.0 and M2.1.5 mitigating biggest risks early.)

---