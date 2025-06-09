# SRES Evaluation Framework

## Overview
This project provides a framework for evaluating Large Multi-modal Models (LMMs) using a self-reflective comparison and scoring system. It includes scripts for running a test model, comparing outputs via a self-reflective mechanism, and scoring performance. Results and detailed logs are saved for analysis.

## Project Structure
- **`lib/`**: Library modules for comparison and utilities.
  - `compare_llm.py`: Implements the self-reflective comparison model to analyze and refine LMM outputs.
- **`model/`**: Directory for test models and results.
  - `qwen.py`: The LMM to be tested (e.g., Qwen model implementation).
  - `result/`: Folder to store test model outputs and evaluation results.
- **`evaluator.py`**: Scoring model script to evaluate the test model's performance.

## Workflow
1. **Run the Test Model**:
   - Execute `model/qwen.py` to generate responses from the LMM under test.
   - Outputs are automatically saved to the `model/result/` folder.
2. **Evaluate the Model**:
   - In `evaluator.py`, specify the test model output file (ensure filenames match for consistency).
   - Run `evaluator.py` to score the model’s performance.
   - Evaluation results and detailed logs are saved to the `model/result/` folder.

## Setup and Usage
1. **Prerequisites**:
   - Python 3.8 or higher.
   - Install required dependencies: `pip install -r requirements.txt`.
2. **Steps**:
   - **Step 1: Generate Test Model Outputs**
     - Run: `python model/qwen.py`
     - Check `model/result/` for saved outputs.
   - **Step 2: Configure and Run Evaluation**
     - Edit `evaluator.py` to reference the output file from `model/result/`.
     - Run: `python evaluator.py`
     - Review scores and detailed logs in `model/result/`.

## Key Components
- **`lib/compare_llm.py`**:
  - Purpose: A self-reflective comparison model that analyzes LMM outputs, identifies inconsistencies, and refines results.
  - Usage: Called by `evaluator.py` for output validation and improvement.
- **`model/qwen.py`**:
  - Purpose: The LMM under test (e.g., Qwen model) to generate responses for evaluation.
  - Output: Saves results to `model/result/`.
- **`evaluator.py`**:
  - Purpose: Scores the test model’s outputs using predefined metrics.
  - Input: Requires the filename of the test model output (must match files in `model/result/`).
  - Output: Saves scores and detailed evaluation logs to `model/result/`.

## Output
- **Location**: All results, including test model outputs, scores, and detailed logs, are stored in `model/result/`.
- **Format**: Results are saved as text or structured files (e.g., JSON, CSV) for easy analysis.

## Notes
- Ensure filenames for test model outputs and evaluation inputs are consistent to avoid errors.
- Check `model/result/` for detailed logs to understand model performance and evaluation insights.
