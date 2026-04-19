# SpikingDQN: Applying SNN to RL in Active Object Localization

This repository explores the intersection of Reinforcement Learning (RL) and Spiking Neural Networks (SNNs) to perform Active Object Localization. 

By formulating object localization as a Markov Decision Process (MDP), our RL agent learns to sequentially adjust a bounding box—moving, scaling, and altering its aspect ratio—until it accurately encapsulates the target object. The agent's policy and value networks are powered by Spiking Neural Networks, leveraging their innate temporal dynamics and efficiency.

## Repository Structure

The project has evolved through several iterations. Researchers should focus their efforts on `v2` for the most modular and up-to-date PyTorch implementation.

*   `v2/` **(Active Branch)**: The streamlined, modular PyTorch codebase. Features three interchangeable SNN backbones (`surrogate`, `ats`, and `stdp`), unified data loading, and standardized testing/evaluation pipelines. **You should work with this directory.**
*   `baseline/`: Houses the standard, non-spiking Artificial Neural Network implementations (Standard Deep Q-Network) for baseline comparisons against the SNN agents.
*   `v1/`: Legacy codebase containing earlier hybrid CNN-SNN experimentation.

## Dataset

This project is tuned to work with the **PASCAL VOC 2012** dataset. It expects the data to be placed in a directory named `VOC2012/` at the root of the project.

**Download the dataset here**: [PASCAL VOC 2012 (Google Drive)](https://drive.google.com/drive/folders/1ikKFR2nbdLw-W6cazaVYyYXCNUeXW6E7?usp=sharing)

Ensure your root structue looks like this:
```
PatternRecognition/
├── baseline/
├── v1/
├── v2/
└── VOC2012/
    ├── Annotations/
    └── JPEGImages/
```

## Quick Start (v2 Architecture)

The `v2` framework is designed for ease of use. It supports 3 primary SNN Methods:

1.  **Surrogate** (`--method surrogate`): Spiking Convolutional layers trained End-to-End via BPTT with surrogate gradients (`SuperSpike`).
2.  **ATS** (`--method ats`): ANN-to-SNN. Trains as a regular CNN and simulates discrete Integration-and-Fire neurons during evaluation.
3.  **STDP** (`--method stdp`): Unsupervised bio-plausible Feature Extraction using Difference of Gaussians (DoG) and Winner-Take-All lateral inhibition, followed by a spiking RL head.

*(Note: `surrogate` and `ats` optionally support a frozen VGG16 backbone via `--backbone vgg16` for abstracted feature extraction).*

### Training

Use `v2/train.py` to train an agent. You can specify a target class or use `mixing` to train on all objects.

```bash
# Train using Surrogate Gradients to localize aeroplanes
python v2/train.py --method surrogate --target aeroplane --epochs 20

# Train the ATS model using a VGG16 backbone
python v2/train.py --method ats --target mixing --backbone vgg16 --epochs 50

# Train using Biological STDP (Runs unsupervised pre-training automatically)
python v2/train.py --method stdp --target aeroplane
```

### Evaluation & Testing

Use `v2/test.py` to evaluate your trained saved models. The script calculates Localization Accuracy at multiple Intersection-over-Union (IoU) thresholds.

```bash
# Evaluate with visual Matplotlib playback of the bounding box search path
python v2/test.py --method surrogate --target aeroplane --render

# Evaluate quietly and export granular metrics to a CSV file
python v2/test.py --method surrogate --target aeroplane --logging
```

---
*Developed as part of research into Advanced Agentic Coding and Spiking Reinforcement Learning.*