# Project 2 PRD: Taric AI Agent (IL + RL) (`taric_ai_agent`)

**Version:** 1.1
**Date:** May 28, 2025
**Project Goal:** To develop a high-performing AI agent for playing Taric in a 2v2 laning scenario. The agent will be pre-trained using Imitation Learning from human expert data collected from the live League of Legends game, and then further refined using Reinforcement Learning within the `lol_sim_env` (from Project 1).

---

## I. Vision & Scope

* **Problem:** Training an RL agent for Taric from scratch is sample-inefficient. Human expert data provides a strong inductive bias.
* **Solution:** Combine IL and RL.
    1.  **IL:** Collect data from human Taric gameplay in the *live LoL game* (Practice Tool for initial clean data). Train a policy to mimic human actions mapped to the `lol_sim_env`'s abstract state/action space.
    2.  **RL:** Initialize an RL agent with the IL policy. Train it in the `lol_sim_env`.
* **Dependency:** Critically depends on Project 1 (`lol_sim_env`) and a robust `state_action_mapper.py`.
* **Development Philosophy:** Iterative. **M2.0 (State/Action Mapping Feasibility Prototype) is the highest initial priority.** Simplify IL data collection and vision processing for MVP. Focus on an end-to-end IL->RL pipeline with simplified components first. Parallel development with Project 1, with Project 1 focusing on environment features and Project 2 focusing on data pipeline and agent, ensuring interfaces align through early integration tests.

---

## II. Requirements & Features (R-Sections)

This section outlines target functionality. "IV. Detailed Implementation Plan" breaks these into AI-codable tasks.

### R1: Imitation Learning (IL) from Live Game Data
* R1.1: Live Data Collection System (`taric_ai_agent/il/live_data_collector/`):
    * R1.1.1: `scripts/collect_il_data.py` orchestrator (configurable for Practice Tool scenarios).
    * R1.1.2: `lcu_data_fetcher.py`: Polls LCU (Taric stats, other player basic info, events). Configurable tick rate.
    * R1.1.3: `screen_recorder.py`: Captures game window. Configurable FPS (synced with LCU poll).
    * R1.1.4: `input_logger.py`: Logs keyboard/mouse with timestamps.
* R1.2: IL Data Processing Pipeline (`taric_ai_agent/il/` & `taric_ai_agent/common/`):
    * R1.2.1 (Iterative): `vision_processor.py`: Processes screen frames. **MVP: Manual annotation or very simple vision for key entity positions/health.** Post-MVP: Object detection (YOLO), health bar estimation.
    * R1.2.2: State Construction & Synchronization in `data_processor.py`: Aligns LCU, vision, input logs.
    * R1.2.3 (Critical & Iterative): `State Mapping to Sim-Env` in `state_action_mapper.py`:
        * Function `map_live_state_to_sim_observation(live_game_state, mapping_config)`: Maps live game state to `lol_sim_env.observation_space`. Handles coordinate transformation (e.g., screen-relative to sim-relative). Normalizes values.
    * R1.2.4 (Critical & Iterative): `Action Mapping to Sim-Env` in `state_action_mapper.py`:
        * Function `map_live_action_to_sim_action(live_input, live_game_state)`: Maps live inputs to `lol_sim_env.action_space`. Uses heuristics for mouse clicks.
    * R1.2.5: `scripts/process_il_data_session.py` generates `(sim_obs, sim_action)` pairs.
* R1.3: IL Model (`taric_ai_agent/agents/il_policy_network.py` & `il/il_dataset.py`):
    * R1.3.1: IL neural network (e.g., MLP, optional CNN for vision features if used in sim_obs).
    * R1.3.2: PyTorch `Dataset` for processed IL data.
    * R1.3.3: `scripts/train_il_model.py` for supervised learning.

### R2: Reinforcement Learning (RL) using `lol_sim_env`
* R2.1: RL Agent Framework (`taric_ai_agent/agents/rl_agent_sb3.py`): Stable Baselines3 PPO.
* R2.2: Policy Initialization from IL model.
* R2.3: RL Training Loop (`scripts/train_rl_agent.py`): Cloud-execution ready.
* R2.4: Reward Function Iteration (Feedback to Project 1).

### R3: Agent Evaluation
* R3.1: `scripts/evaluate_agent_in_sim.py` for IL-only and IL+RL agents in `lol_sim_env`.
* R3.2: Metrics: Episode outcomes in sim (KDA, gold/XP diff, etc.), ability usage stats.

---

## III. Proposed Folder Structure & Key Files
(As defined in `architecture.md` for Project 2)

---
## IV. Detailed Implementation Plan (Task-Oriented for AI Assistant)

**Overall Iterative Strategy:** Prioritize M2.0. Then, establish a minimal viable IL data pipeline. Test agent interface with Project 1's MVP environment early (M2.4).

**Phase 0: Foundational Setup & Prototyping**

* **Task 0.1: Initialize Project Repository and Core Files for Agent**
    * **Objective:** Create the main project directory, initialize Git, and create placeholder `requirements.txt`.
    * **File(s):** `taric_ai_agent_project/` (root folder), `requirements.txt`.
    * **Instructions for AI:**
        1.  "Create root directory `taric_ai_agent_project`."
        2.  "Inside, initialize Git."
        3.  "Create `requirements.txt`, add: `numpy gymnasium pyyaml torch torchvision torchaudio stable-baselines3 opencv-python pynput psutil requests` (add more as identified)."
* **Task 0.2: Create Initial Agent Package Structure**
    * **Objective:** Set up the basic Python package structure for `taric_ai_agent`.
    * **File(s):** Create directories and `__init__.py` files as per `architecture.md` under `taric_ai_agent_project/taric_ai_agent/`.
    * **Instructions for AI:** (Similar to Task 1.1.2 of Project 1, but for `taric_ai_agent` structure)
        1.  Create `taric_ai_agent` dir and `__init__.py`.
        2.  Create subdirs: `agents`, `il`, `common`, `utils` with `__init__.py` files.
        3.  Inside `il`, create `live_data_collector` subdir with `__init__.py`.
        4.  Create top-level `scripts`, `configs`, `data`, `trained_models`, `notebooks` dirs.
        5.  Inside `data`, create `il_raw_live_recordings` and `il_processed_for_sim`.
        6.  Inside `trained_models`, create `vision_object_detector`.

**Phase 1: State/Action Mapping Feasibility (Corresponds to old M2.0 - CRITICAL SPIKE)**

* **Task 1.1: Prototype State/Action Mapper (`scripts/prototype_state_action_mapper.py` or `notebooks/00_state_action_mapping_prototype.ipynb`)**
    * **Objective:** Prove conceptual feasibility of mapping minimal live-like data to MVP sim state/actions.
    * **File(s):** Script or Notebook. `taric_ai_agent/common/state_action_mapper.py` (initial version).
    * **Instructions for AI:**
        1.  "Define 5 simplified Python dictionaries representing diverse 'live game state snapshots' for Taric. Include keys like `taric_hp_live`, `taric_mana_live`, `taric_q_cd_live`, `taric_screen_pos_live: tuple(x,y)`, `enemy1_screen_pos_live: tuple(x,y)`, `enemy1_hp_live_percent`."
        2.  "For each snapshot, define a corresponding 'live player action' string (e.g., 'PRESSED_Q', 'CLICK_MOVE_FORWARD_RELATIVE_TO_TARIC', 'CLICK_ATTACK_ENEMY1')."
        3.  "In `state_action_mapper.py`, create function `map_live_state_to_MVP_sim_observation(live_state_snapshot)`. This function should take a snapshot dict and return a dict matching the Project 1 MVP `TaricLaningSimEnv` observation space (e.g., normalized Taric HP, Mana, Q_CD, and a *conceptual* normalized Taric position derived from `taric_screen_pos_live`). For enemy info, it can return placeholder/default values for now."
        4.  "In `state_action_mapper.py`, create function `map_live_action_to_MVP_sim_action(live_action_string)`. This function should take the action string and return an integer corresponding to one of the Project 1 MVP `TaricLaningSimEnv` discrete actions (e.g., Move_Up, Use_Q_Self)."
        5.  "In the script/notebook, iterate through your manual snapshots and actions. For each, call the mappers and print the original live data and the mapped sim observation/action. Verify the output format and logic."
    * **Verification:** Mapped data is in the correct format for the Project 1 MVP Env's obs/action space. Logic seems sound for these simple cases.

**Phase 2: IL Live Data Collection MVP (Corresponds to old M2.1 - Leverage Practice Tool)**

* **Task 2.1: Implement Basic `lcu_data_fetcher.py`**
    * **Objective:** Fetch minimal Taric data from LCU. (Ref: R1.1.2)
    * **File(s):** `taric_ai_agent/il/live_data_collector/lcu_data_fetcher.py`. `configs/live_data_collection_config.yaml`.
    * **Instructions for AI:**
        1.  "Use `requests` library to connect to LCU (handle finding port/auth from lockfile, or assume fixed for now)."
        2.  "Implement `fetch_taric_mvp_data()`: Get `/liveclientdata/activeplayer`. Extract and return Taric's current HP, MaxHP, current Mana, MaxMana, Q ability cooldown (from `activePlayer['abilities']['Q']['cooldown']`)."
        3.  "Implement a loop that calls this function at a rate defined in `live_data_collection_config.yaml` (e.g., `lcu_poll_hz: 5`) and logs timestamped data to a list/queue."
* **Task 2.2: Implement Basic `input_logger.py`**
    * **Objective:** Log Q key presses and right-clicks. (Ref: R1.1.4)
    * **File(s):** `taric_ai_agent/il/live_data_collector/input_logger.py`.
    * **Instructions for AI:**
        1.  "Use `pynput` library."
        2.  "Log timestamped 'Q' key presses."
        3.  "Log timestamped right-mouse clicks with their (x, y) screen coordinates."
* **Task 2.3: Implement Basic `screen_recorder.py` (Optional for LCU-only mapping prototype)**
    * **Objective:** Capture game window frames. (Ref: R1.1.3)
    * **File(s):** `taric_ai_agent/il/live_data_collector/screen_recorder.py`.
    * **Instructions for AI:**
        1.  "Use a library like `mss` for screen capture."
        2.  "Capture the primary monitor or a specified window region (configurable) at FPS from `live_data_collection_config.yaml`."
        3.  "Save frames to a session-specific directory (e.g., `data/il_raw_live_recordings/session_XYZ/frame_timestamp.png`). Log frame paths and timestamps."
* **Task 2.4: Implement `scripts/collect_il_data.py` Orchestrator**
    * **Objective:** Run collectors and save data for one short session.
    * **File(s):** `scripts/collect_il_data.py`. `data/il_raw_live_recordings/`.
    * **Instructions for AI:**
        1.  "Create a unique session ID (e.g., timestamp based)."
        2.  "Start LCU fetcher, input logger, (optional) screen recorder in separate threads/processes."
        3.  "Run for a configurable duration (e.g., 2-3 minutes from `live_data_collection_config.yaml`)."
        4.  "Save LCU log, input log, and frame log/video to the session directory."
        5.  "Recommendation: For first run, use League Practice Tool with Taric against static dummies to get clean data."

**Phase 2.5: Vision System Prototyping (Corresponds to old M2.1.5 - Independent Spike)**

* **Task 2.5.1: Manually Annotate Sample Frames or Test Simple Vision**
    * **Objective:** Assess vision requirements for MVP IL data processing.
    * **File(s):** `notebooks/02_test_vision_processing_prototype.ipynb`. `configs/vision_processing_config.yaml`.
    * **Instructions for AI:**
        1.  "Load ~20-50 diverse frames from the M2.4 data collection."
        2.  "If `vision_processing_config.yaml` specifies `manual_annotation_mode_for_mvp: true`, guide user to manually record rough screen X,Y for Taric (e.g., from HUD minimap icon) for these frames. Store these annotations."
        3.  "Alternatively, if aiming for automated vision for MVP: try simple template matching (OpenCV) for Taric's HUD portrait or a dominant visual feature to get a rough screen position. Evaluate its reliability on the sample frames."
        4.  "Do not implement full object detection (YOLO) at this MVP stage unless previous step proves utterly unfeasible and YOLO is determined as the only way forward for even basic position."
    * **Deliverable:** A small set of frames with corresponding (manual or very simply extracted) Taric screen positions. Understanding of vision difficulty for MVP.

**Phase 3: IL Data Processing & State/Action Mapping MVP (Corresponds to old M2.2)**

* **Task 3.1: Enhance `state_action_mapper.py` for MVP Data**
    * **Objective:** Map actual collected MVP live data to MVP sim state/actions.
    * **File(s):** `taric_ai_agent/common/state_action_mapper.py`.
    * **Instructions for AI:**
        1.  "Refine `map_live_state_to_MVP_sim_observation` from Task 1.1. It now takes: LCU data (Taric HP, Mana, Q_CD), and Taric's screen position (from M2.5.1 - manual annotation or simple vision output)."
        2.  "Map these to the Project 1 MVP `TaricLaningSimEnv` observation space (normalized Taric HP, Mana, Q_CD, and conceptual normalized Taric sim position based on screen pos). Pad other sim_obs features."
        3.  "Refine `map_live_action_to_MVP_sim_action`. It now takes: a live input log entry (Q press or right-click with coords), and current live Taric screen position (from M2.5.1)."
        4.  "Map 'Q' to sim `CAST_Q_SELF` (or equivalent MVP action). Map right-clicks to one of 4 discrete `MOVE_` sim actions based on direction relative to live Taric screen pos."
* **Task 3.2: Implement `scripts/process_il_data_session.py` for MVP**
    * **Objective:** Process one MVP raw data session.
    * **File(s):** `scripts/process_il_data_session.py`. `data/il_processed_for_sim/`.
    * **Instructions for AI:**
        1.  "Load LCU log, input log, and corresponding Taric screen positions (from M2.5.1 output) for the MVP session."
        2.  "Align data by timestamp (prioritize input log events)."
        3.  "For each relevant input action, construct live state snapshot using nearest LCU data and Taric screen position."
        4.  "Call mappers from `state_action_mapper.py`."
        5.  "Save `(sim_obs_mvp, sim_action_mvp)` pairs to `dataset_mvp.h5`."

**Phase 4: Basic Imitation Learning Model MVP (Corresponds to old M2.3)**

* **Task 4.1: Implement IL Model and Dataset Class for MVP**
    * **Objective:** Create and train a very simple IL model.
    * **File(s):** `taric_ai_agent/agents/il_policy_network.py`, `taric_ai_agent/il/il_dataset.py`.
    * **Instructions for AI:**
        1.  "In `il_policy_network.py`, define a simple PyTorch MLP. Input: flattened MVP sim_obs. Output: logits for MVP sim_actions."
        2.  "In `il_dataset.py`, create `IL_MVP_Dataset(Dataset)` to load `dataset_mvp.h5`."
* **Task 4.2: Implement `scripts/train_il_model.py` for MVP**
    * **Objective:** Train the MVP IL model.
    * **File(s):** `scripts/train_il_model.py`. `trained_models/il_policy_mvp.pth`.
    * **Instructions for AI:**
        1.  "Load `IL_MVP_Dataset`."
        2.  "Initialize MLP model, optimizer (Adam), loss (CrossEntropy)."
        3.  "Train for a small number of epochs (e.g., 10-50). Save the trained model."

**Phase 5: RL Agent Integration & Basic Run MVP (Corresponds to old M2.4, M2.5, M2.6)**

* **Task 5.1: Integrate with Project 1 Env & Test IL Policy (CRITICAL INTEGRATION POINT)**
    * **Objective:** Ensure Project 1 `lol_sim_env` (MVP version, e.g., after its M1.2.6 or M1.3) is installable/importable. Load IL policy and run in this sim env.
    * **File(s):** `scripts/evaluate_agent_in_sim.py`.
    * **Instructions for AI:**
        1.  "Ensure `lol_sim_env` can be imported."
        2.  "In `evaluate_agent_in_sim.py`, load the trained `il_policy_mvp.pth`."
        3.  "Instantiate `TaricLaningSimEnv` (MVP version from Project 1)."
        4.  "Run one episode: at each step, get `sim_obs` from env, pass to IL model to get `sim_action`, execute `sim_action` in env. Print obs, action, reward."
    * **Verification:** Agent takes actions in the sim env based on IL policy without crashing. Qualitative check if actions are vaguely sensible for the learned data.
* **Task 5.2: Setup Basic SB3 RL Agent & Short Training Run**
    * **Objective:** Confirm RL training loop runs with the MVP sim environment.
    * **File(s):** `taric_ai_agent/agents/rl_agent_sb3.py`, `scripts/train_rl_agent.py`.
    * **Instructions for AI:**
        1.  "In `rl_agent_sb3.py`, set up a PPO agent from Stable Baselines3 to use `TaricLaningSimEnv` (MVP)."
        2.  "Modify `scripts/train_rl_agent.py` to initialize this PPO agent (optionally load IL policy weights if SB3 allows easy policy network surgery for MLP-to-MLP)."
        3.  "Run a very short training loop (e.g., 1000-5000 PPO steps). Save the SB3 model."
        4.  "Update `scripts/evaluate_agent_in_sim.py` to also be able to load and run the SB3 PPO agent."
    * **Verification:** RL training loop completes. Evaluation script can run the PPO agent.

---
## V. Risks & Challenges (Project 2)
(As previously detailed, with M2.0 and M2.1.5 mitigating biggest risks early.)

---