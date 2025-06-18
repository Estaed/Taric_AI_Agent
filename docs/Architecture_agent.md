# Architecture & File Structures

This document outlines the proposed file and folder structures for the two main projects:
1.  `lol_sim_env`: The Python-based simulated 2v2 LoL laning environment for RL training.
2.  `taric_ai_agent`: The Taric AI agent, using IL from live game data and RL within `lol_sim_env`.

## Project 2: Taric AI Agent (`taric_ai_agent`)

**Goal:** To develop a Taric AI agent using Imitation Learning (IL) from live game expert data, and Reinforcement Learning (RL) within the `lol_sim_env` created in Project 1.

**Proposed Folder Structure:**

taric_ai_agent_project/
├── taric_ai_agent/          # Core Python package for the agent logic
│   ├── init.py
│   ├── agents/              # Agent model definitions and training wrappers
│   │   ├── init.py
│   │   ├── base_agent.py      # Abstract base class for agents
│   │   ├── il_policy_network.py # Neural network architecture for IL (e.g., PyTorch nn.Module)
│   │   └── rl_agent_sb3.py    # RL agent setup using Stable Baselines3 (or similar)
│   ├── il/                  # Imitation Learning specific modules
│   │   ├── init.py
│   │   ├── live_data_collector/ # Tools for collecting data from the live LoL game
│   │   │   ├── init.py
│   │   │   ├── lcu_data_fetcher.py # Interacts with LCU API
│   │   │   ├── screen_recorder.py  # Captures screen (interfacing with OS-level tools)
│   │   │   ├── input_logger.py     # Logs keyboard/mouse (e.g., using pynput)
│   │   │   └── orchestrator.py     # Manages concurrent execution of collectors
│   │   ├── vision_processor.py  # Processes recorded screen frames (object detection, health estimation)
│   │   ├── data_processor.py    # Processes raw live game data into (sim_state, sim_action) pairs using mapper
│   │   └── il_dataset.py      # PyTorch Dataset class for the processed IL data
│   ├── common/                # Shared components
│   │   ├── init.py
│   │   ├── state_action_mapper.py # Logic to map live game state/actions to lol_sim_env space
│   │   └── observation_utils.py # Helper functions for processing/normalizing observations
│   └── utils/                 # General utilities for this project
│       ├── init.py
│       └── training_utils.py  # Callbacks, logging for ML training loops
├── scripts/                 # Executable scripts
│   ├── collect_il_data.py   # Runs the live data collection orchestrator
│   ├── process_il_data_session.py # Script to process a specific raw IL data session
│   ├── train_il_model.py    # Trains the IL model
│   ├── train_rl_agent.py    # Trains the RL agent using lol_sim_env (designed for cloud execution)
│   └── evaluate_agent_in_sim.py # Evaluates trained agents in lol_sim_env
│   └── prototype_state_action_mapper.py # Script for M2.0 spike
├── configs/                 # Configuration files for data collection, training, agent parameters
│   ├── live_data_collection_config.yaml # (e.g., LCU poll rate, screen FPS, save paths, practice_tool_scenario)
│   ├── vision_processing_config.yaml # (e.g., object detection model path, confidence thresholds, manual_annotation_mode_for_mvp)
│   ├── state_action_mapping_config.yaml # (e.g., normalization params, mapping heuristics, coord_map_strategy)
│   ├── il_agent_config.yaml     # (e.g., model architecture, training hyperparameters)
│   └── rl_agent_config.yaml     # (e.g., SB3 PPO hyperparameters, learning rate)
├── data/                    # Data storage
│   ├── il_raw_live_recordings/ # Raw output from collect_il_data.py
│   │   └── session_YYYYMMDD_HHMMSS/
│   │       ├── lcu_log.jsonl
│   │       ├── screen_frames_log.csv # Log of frame timestamps and paths if saved individually
│   │       ├── screen_video.mp4   # Alternative: save as video
│   │       └── input_log.csv
│   └── il_processed_for_sim/ # Processed (state_sim, action_sim) pairs
│       └── dataset_v1.h5    # (or other suitable batch data format)
├── trained_models/          # Saved model weights
│   ├── vision_object_detector/ # If you train/fine-tune your own detector
│   │   └── yolo_lol_detector.pt
│   ├── il_policy_vX.pth
│   └── rl_policy_vX.zip      # (Stable Baselines3 format)
├── notebooks/               # Jupyter notebooks for experimentation, data analysis, visualization
│   ├── 00_state_action_mapping_prototype.ipynb # For M2.0 spike
│   ├── 01_explore_raw_il_data.ipynb
│   ├── 02_test_vision_processing_prototype.ipynb # For M2.1.5 spike
│   └── 03_analyze_il_dataset.ipynb
├── requirements.txt         # Python package dependencies (lol_sim_env, PyTorch, SB3, OpenCV, PyYAML etc.)
└── README.md                # Project overview, setup, usage instructions for the agent, known limitations of state/action mapping