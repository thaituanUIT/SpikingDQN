# Applying SNN to RL in Active Object Localization (v2)

This `v2` codebase provides a unified framework for training and testing Spiking Neural Networks (SNNs) as the policy/value network for an Active Object Localization agent via Reinforcement Learning (DQN).

## Overview

The reinforcement learning environment frames object localization as a Markov Decision Process (MDP). The agent observes a cropped region of the image and selects from 9 discrete actions to manipulate the bounding box:
1. Move Right
2. Move Left
3. Move Down
4. Move Up
5. Scale Larger
6. Scale Smaller
7. Decrease Vertical Aspect Ratio (Fatter)
8. Decrease Horizontal Aspect Ratio (Taller)
9. **Trigger/Terminate** (Indicates the object is found)

The agent receives a positive reward (+1) if the Intersection Over Union (IoU) with the ground truth improves, and a termination bonus (+3) if it halts with an IoU > 0.5.

## Available Methods

The framework supports three interchangeable SNN architectures:

1. **Surrogate (`--method surrogate`)**: 
   Standard Direct Training via Backpropagation Through Time (BPTT) using the `SuperSpike` surrogate gradient logic.
2. **ATS (`--method ats`)**: 
   ANN-To-SNN conversion. Pre-trains the RL agent as a standard Convolutional Neural Network with ReLUs, and discretizes the weights logically into Integrate-and-Fire neurons for inference/evaluation.
3. **STDP (`--method stdp`)**: 
   A biologically plausible Spiking Deep Convolutional Neural Network utilizing Spiking Timing-Dependent Plasticity (STDP). Employs a Difference of Gaussians (DoG) filter to simulate retinal latencies and applies unsupervised Winner-Take-All lateral inhibition. *Requires raw 2D input and does not support 1D VGG16 backbones.*

## VGG16 Backbone Support

By default, the `surrogate` and `ats` methods use a shallow, built-in Convolutional Neural Network layer stack to extract spatial features directly from the raw pixels. 

To improve convergence and feature abstraction during Reinforcement Learning, researchers can inject a frozen **VGG16 (`--backbone vgg16`)** model to extract 25,088 features from the image before feeding it directly into the Spiking fully-connected layers.

## Training Usage

Train an agent using the unified `train.py` script. The script automatically handles loading the VOC2012 dataset from the root directory.

```bash
# Basic Surrogate training isolating the "aeroplane" class
python v2/train.py --method surrogate --target aeroplane --epochs 20

# Surrogate training over the entire mixed dataset using a VGG16 extraction backbone
python v2/train.py --method surrogate --target mixing --backbone vgg16 --epochs 50

# ATS training with a 15-timestep simulation
python v2/train.py --method ats --target aeroplane --simulate 15

# Biological STDP training (NOTE: Automatically runs unsupervised STDP first, then DQN)
python v2/train.py --method stdp --target aeroplane
```

## Testing Usage

Test the saved agent policies with visual playback using `test.py`:

```bash
# Test the Surrogate model with the visualizer displaying the search path (Blue bounds -> Red bounds)
python v2/test.py --method surrogate --target aeroplane --render

# Evaluate but suppress matplotlib rendering for speed
python v2/test.py --method surrogate --target aeroplane

# Test the ATS model (Simulates the SNN conversion during inference) using the VGG16 backbone
python v2/test.py --method ats --target mixing --backbone vgg16
```

## Dataset 
Our dataset is the PASCAL VOC 2012 dataset, which is a collection of images of objects in different categories. Link to dataset: 