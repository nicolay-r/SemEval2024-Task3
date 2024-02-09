# SemEval-2024 Task 3: Codalab Service for [THOR-ECAC](https://github.com/nicolay-r/THOR-ECAC) framework

### 📊 [Codalab Competiton Page](https://codalab.lisn.upsaclay.fr/competitions/16141)
### 🤖 [THOR-ECAC code implementation](https://github.com/nicolay-r/THOR-ECAC)

## Description

This repository shares data and submission-related code for training and handling results of  
[THoR-ECAC framework](https://github.com/nicolay-r/THOR-ECAC).

This project contributes with the following scripts:
* **Resources Preparation for `pair-based` experiments in context**:
  * [0_emotion_state.py](e3_pair_ft/0_emotion_state.py) -- script for pretraining data preparation, based on `states`;
  * [0_emotion_cause.py](e3_pair_ft/0_emotion_cause.py) -- script for fine-tuning data preparation, based on `causes`;
  * [1_ps_vocab.py](e3_pair_ft/1_ps_vocab.py) -- vocabulary preparation for manual spans corrections.
* **Codalab submissions forming**:  
  * [2_submit](e3_pair_ft/2_submit.py) -- script for forming `*.json.zip` archive, compatible for submitting on Codalab platfom.
* Conversation data analysis:
  * [task_statistics_json.py](task_statistics_json.py) -- `json` data analyzer;
  * [task_statistics_submission.py](task_statistics_submission.py) -- `*json.zip` submissions analyzer.
