Self-Reflective Evaluation System (SRES) for Large Multi-modal Models
Project Overview
The Self-Reflective Evaluation System (SRES) is a novel framework designed to comprehensively evaluate large multi-modal models (LMMs) across three core dimensions: Visual Comprehension, Linguistic Understanding, and Robustness. This system integrates a tri-channel assessment framework, a self-reflective mechanism, and a DeepEval scoring network to deliver holistic, reliable, and reproducible evaluations. SRES addresses limitations in traditional evaluation methods by capturing multi-modal interactions, mitigating output variability, and providing a scalable benchmark for 15 leading LMMs.
The implementation and benchmark dataset are publicly available at: https://anonymous.dopen.science/r/SRESB2B
Key Features

Tri-Channel Assessment: Simultaneously evaluates Visual (V), Linguistic (L), and Robustness (R) capabilities.
Self-Reflective Mechanism: Dynamically adjusts outputs to enhance stability and objectivity.
DeepEval Scoring Network: Uses DeepSeek-R1 for accurate, automated scoring aligned with human judgments.
Comprehensive Benchmark: Includes 352 subtasks across 181 task groups, covering diverse domains like medical imaging, mathematics, and humanities.

Configuration Parameters
The following parameters can be adjusted in the configuration files:



Parameter
Description
Default Value
Notes



max_eval_reflections
Max self-reflection cycles in evaluation phase
1
Adjustable for precision needs


max_score_reflections
Max self-reflection cycles in scoring phase
5
Range: 3-8 cycles per score


scoring_model
Primary scoring model
DeepSeek-R1
Selected for human-like accuracy


task_difficulty
Difficulty level of tasks (high, medium, low)
Mixed
Manually annotated, customizable


channel_weights
Weights for V, L, and R channels in scoring
Equal (1,1,1)
Adjustable for task prioritization


Code File Structure

sres/: Root directory for the project.
config/: Configuration files for parameters (e.g., config.yaml).
src/: Core source code.
tri_channel.py: Implements the tri-channel assessment framework.
self_reflect.py: Handles the self-reflective mechanism for output stability.
deepeval_score.py: DeepEval scoring network powered by DeepSeek-R1.
utils.py: Utility functions for data processing and evaluation.


scripts/: Scripts for running evaluations.
run_evaluation.py: Main script to execute the full pipeline.
analyze_results.py: Generates performance metrics and visualizations.


tests/: Unit tests for validation.
test_channels.py: Tests for tri-channel framework.
test_reflection.py: Tests for self-reflection mechanism.





Dataset Structure
The benchmark dataset comprises 181 task groups, totaling 352 fine-grained subtasks, structured as follows:

Task Group Composition: Each group includes:
1 Visual task (e.g., OCR, spatial awareness).
1 Linguistic task (e.g., text generation, logical inference).
3 Robustness evaluation items (e.g., stability across runs).


Domains Covered: Medical imaging, biological sciences, mathematics, humanities, social sciences, flowcharts, emoticons, and more.
Annotations: Tasks are supported by 1-8 images and labeled with difficulty levels (high, medium, low).
Formats: Includes multiple-choice, true/false, and open-ended questions.
Location: Available at https://anonymous.dopen.science/r/SRESB2B.

Evaluated Tasks
SRES evaluates 11 core capabilities across three dimensions:

Visual Comprehension (5 Tasks):
Optical Character Recognition (OCR)
Visual Recognition
Spatial Awareness
Motion Recognition
Environmental Understanding


Linguistic Understanding (4 Tasks):
Knowledge Utilization
Logical Inference
Mathematics
Text Generation


Robustness (2 Tasks):
Output Stability
Consistency Across Runs



Installation and Usage

Prerequisites:
Python 3.8+
Install dependencies: pip install -r requirements.txt
DeepSeek-R1 model access (or configure alternative scoring models)


Setup:
Clone the repository: git clone https://anonymous.dopen.science/r/SRESB2B
Configure parameters in config/config.yaml


Run Evaluation:
Execute: python scripts/run_evaluation.py --config config/config.yaml
Analyze results: python scripts/analyze_results.py


Output:
Results include composite scores, per-channel metrics, and stability analysis.



Evaluated Models

Open-Source (9): DeepSeek-VL2, InternVL2-26B, LLaMA-3.2-90B-vision-instruct, QVQ-72B-Preview, Qwen2-VL-72B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, Yi-vision-v2, etc.
Proprietary (6): Claude-3.5-sonnet, Doubao-1.5-vision-pro-32k, Gemini-2.0-flash-thinking-exp, ChatGPT-4o, ChatGPT-4o-all, Moonshot-v1-128k-vision-preview.

Limitations

Data Accuracy: Task difficulty validated only for 15 LMMs; broader validation ongoing.
Data Richness: Task quantity and variety to be expanded in future iterations.
Model Selection: Relies on DeepSeek-R1; future updates may incorporate new LLMs.
Prompt Engineering: Prompt responses vary by task type; customization in progress.

Citation
If you use SRES in your research, please cite:
Anonymous ACL submission. "Seeing is believing: Comprehensive Self-Reflective Evaluation System for Large Multi-modal Models."

Contact
For questions or contributions, refer to the repository at https://anonymous.dopen.science/r/SRESB2B.
