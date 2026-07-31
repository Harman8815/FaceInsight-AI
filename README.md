# FaceInsight-AI

Facial character identification project using the CelebA dataset with PyTorch.

## Project Structure

```
FaceInsight-AI/
├── Facial character identfier/
│   ├── 01-Analysis.ipynb              # Exploratory data analysis
│   ├── 02_Frozen_Backbone_Training_Kaggle.ipynb  # Kaggle-compatible frozen backbone training
│   ├── 02-Frozen Backbone Training.ipynb         # Local frozen backbone training
│   ├── 03-evaluation.ipynb            # Model evaluation with training metrics
│   ├── 03-Fine Tuning.ipynb           # Fine-tuning notebook
│   ├── src/
│   │   ├── config.py                  # Project configuration
│   │   ├── data_pipeline.py           # Data loading and preprocessing
│   │   ├── model.py                   # Model definition
│   │   └── training.py                # Training loop utilities
│   └── archive/                       # Dataset, model weights, and checkpoints
├── .gitignore
└── README.md
```

## Features

- **Frozen backbone training** — Train a classification head on top of a pretrained backbone while keeping the backbone frozen.
- **Kaggle-compatible notebooks** — Training notebook designed to run on Kaggle with minimal setup.
- **Multi-task learning** — Predicts multiple facial attributes simultaneously using a multi-task model.
- **CelebA dataset** — Uses the CelebA dataset with aligned and cropped face images.

## Getting Started

1. Extract the CelebA dataset into the `Facial character identfier/archive/` directory.
2. Open and run the notebooks in order: `01-Analysis.ipynb` → `02_Frozen_Backbone_Training_Kaggle.ipynb` → `03-evaluation.ipynb`.

## Repository

- Remote: https://github.com/Harman8815/FaceInsight-AI.git