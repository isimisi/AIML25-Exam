# Exam for the course AI & ML 2025, CBS
This school project explores the development of an AI-powered agent that goes beyond the capabilities of general-purpose language models like ChatGPT by transforming flowchart diagrams into structured JSON representations. The core goal is to enable full operationalisation of visual platforms like Miro or Lucidchart — solely through AI.

While tools like ChatGPT can interpret text-based representations, they struggle with the precision and structure required to accurately extract, validate, and convert complex visual flowcharts into machine-usable formats. Our system combines object detection (YOLOv8) and large language models (LLMs) to bridge this gap.

The workflow involves:
- **YOLO-based segmentation**: Detect and isolate individual flowchart elements and sub-flowcharts from an input diagram.
- **Edge detection and validation**: Identify the logical flow between components using both visual and LLM-based techniques.
- **Structured conversion**: Convert the detected elements and connections into a structured JSON schema that can be used by downstream systems (e.g., to automate workflows, generate code, or populate BPM tools).

By combining computer vision with large language models, this project aims to create a scalable and flexible pipeline that can turn visual diagrams into operational logic.


## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/isimisi/AIML25-Exam.git
cd AIML25-Exam
```

### 2. Set Up Conda Environment
This project uses Conda to manage dependencies. All required packages are specified in environment.yml.

#### Create the environment:
```bash
conda env create -f environment.yml
```

#### Activate the environment:
```bash
conda activate AIML25-Exam
```

### 3. Run the Code

## Project Structure
```
├── datasets/ # Dataset for LLM and YOLO
├── models/ # YOLO models (large files, git-ignored)
├── notebooks/ # Notebooks for development and testing
│ ├── agent.ipynb # Final agent code
│ └── main.ipynb # Test code and experimentation
├── runs/ # Output directory (e.g., YOLO runs, logs)
├── src/ # Business logic and core modules
│ ├── utils/ # Utility functions (helpers)
│ ├── yolo/ # YOLO-related logic
│ ├── edge_detector_llm.py # Detects edges using LLM
│ ├── edge_validator_llm.py# Validates edges using LLM
│ ├── edge_validator.py # Traditional edge validation logic
│ ├── graph.py # Graph-related utilities and structures
│ ├── llm_caller.py # Handles calls to the LLM (taken from ma3)
│ ├── llm_detector.py # Detects elements using LLM
│ ├── mermaid_detector.py # Detects Mermaid diagram elements
│ └── mermaid_to_json.py # Converts Mermaid diagrams to JSON
```

## Results

| Test | Segmented | Nodes | Mermaid | Precision | Recall | F1 Score |
|------|-----------|-------|---------|-----------|--------|----------|
| 1    | No        | No    | No      | 0.622     | 0.575  | 0.596    |
| 2    | Yes       | No    | No      | 0.585     | 0.770  | 0.658    |
| 3    | No        | Yes   | No      | 0.613     | 0.532  | 0.567    |
| 4    | **Yes**   | **Yes**| **No** | **0.619** | **0.742** | **0.666** |
| 5    | No        | No    | Yes     | 0.441     | 0.396  | 0.417    |
| 6    | Yes       | No    | Yes     | 0.276     | 0.489  | 0.351    |
| 7    | No        | Yes   | Yes     | 0.532     | 0.499  | 0.512    |
| 8    | Yes       | Yes   | Yes     | 0.475     | 0.618  | 0.529    |
