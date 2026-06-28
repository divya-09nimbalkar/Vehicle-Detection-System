```markdown
# Vehicle Detection System

A structured deep learning project designed to classify images into Vehicle and Non-Vehicle categories using a ResNet18 architecture implemented in PyTorch.

---

##  Project Structure

```text
VEHICLE_DETECTION_SYSTEM/
│
├── data/
│   ├── processed/
│   │   ├── NonVehicle/       # Processed background, building, tree images
│   │   └── Vehicle/          # Processed car, truck images
│   └── raw/                  # Unprocessed original dataset
│
├── docs/                     # Project documentation and reports
│
├── models/
│   └── vehicle_resnet18.pth  # Trained PyTorch model weights
│
├── notebooks/
│   └── exploration.ipynb     # Jupyter notebook for EDA and prototyping
│
├── src/                      # Source code for the pipeline
│   ├── __init__.py
│   ├── data_pipeline.py      # Custom PyTorch Datasets and DataLoaders
│   ├── model.py              # ResNet18 model definition/modification
│   ├── train.py              # Script to train the model
│   ├── evaluate.py           # Script to evaluate model performance
│   └── main.py               # Main orchestration script
│
├── tests/                    # Unit tests for code reliability
│   ├── test_data_pipeline.py
│   ├── test_model.py
│   └── test_utils.py
│
├── .gitignore                # Specifies intentionally untracked files
├── README.md                 # Project overview and documentation
└── requirements.txt          # Python dependencies

```

---

##  Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 2. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt

```

---

##  Usage

### Data Preparation

Place your raw images into `data/raw/` and use the pipeline to preprocess and split them into `data/processed/Vehicle/` and `data/processed/NonVehicle/`.

### Training the Model

To train the ResNet18 model from scratch or fine-tune it:

```bash
python src/train.py

```

*The trained model weights will be saved automatically to `models/vehicle_resnet18.pth`.*

### Evaluation

To evaluate the performance of your saved model against the test set:

```bash
python src/evaluate.py

```

### End-to-End Execution

To run the full pipeline (Data preprocessing -> Training -> Evaluation) in one go:

```bash
python src/main.py

```

---

##  Running Tests

This project uses automated unit tests to ensure codebase stability. You can run all the tests using `pytest`:

```bash
pytest tests/

```

---
