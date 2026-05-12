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

The framework supports two interchangeable SNN architectures:

1. **Surrogate (`--method surrogate`)**: 
   Standard Direct Training via Backpropagation Through Time (BPTT) using the `SuperSpike` surrogate gradient logic.
2. **ATS (`--method ats`)**: 
   ANN-To-SNN conversion. Pre-trains the RL agent as a standard Convolutional Neural Network with ReLUs, and discretizes the weights logically into Integrate-and-Fire neurons for inference/evaluation.

## VGG16 Backbone Support

By default, the `surrogate` and `ats` methods use a shallow, built-in Convolutional Neural Network layer stack to extract spatial features directly from the raw pixels. 

To improve convergence and feature abstraction during Reinforcement Learning, researchers can inject a frozen **VGG16 (`--extractor vgg16`)** model to extract 25,088 features from the image before feeding it directly into the Spiking fully-connected layers.

## Training Usage

Train an agent using the unified `train.py` script. The script automatically handles loading the VOC2012 dataset from the root directory.

### Training Parameters (`train.py`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--method` | string | (required) | SNN method to use: `surrogate` or `ats`. |
| `--extractor` | string | `conv` | Feature extractor: `conv`, `vgg16`, `resnet18`, `fusion`, `vit`, `efficientnet`, or `mobilenet`. |
| `--target` | string | `mixing` | Target class (e.g., `aeroplane`) or `mixing` for all classes. |
| `--num-samples`| int | `None` | Limit the number of samples loaded from VOC2012. |
| `--random` | flag | `False` | Random sample from dataset. |
| `--voc-dir` | string | `None` | Override default VOC2012 directory. |
| `--algo` | string | `dqn` | RL algorithm (`dqn`, `double`, `dueling`). |
| `--gamma` | float | `0.99` | Discount factor for future rewards. |
| `--epochs` | int | `10` | Number of RL epochs. |
| `--max-steps` | int | `20` | Max bounding box steps per image. |
| `--alpha` | float | `0.1` | Mask transformation rate. |
| `--nu` | float | `3.0` | Trigger reward weight. |
| `--threshold` | float | `0.5` | IoU threshold for trigger reward. |
| `--replay` | int | `10` | History size. |
| `--target-update`| int | `1` | Epochs between target network updates. |
| `--loss-fn` | string | `huber` | Loss function (`mse`, `huber`, `smooth_l1`). |
| `--simulate` | int | `10` | Simulation timesteps for the SNN. |
| `--optimizer` | string | `adam` | Optimizer (`adam`, `adamw`, `rmsprop`, `sgd`, `radam`). |
| `--lr` | float | `1e-4` | Learning rate. |
| `--weight-decay` | float | `0.0` | Weight decay for optimizer. |
| `--clip-grad` | float | `1.0` | Gradient clipping norm. |
| `--batch-size` | int | `20` | Batch size for training. |
| `--early-stop` | int | `0` | Early stopping if no improvement for N epochs. |
| `--validation` | string | `none` | Validation metric for saving the best model (`none`, `loss`, `iou`). |
| `--val-ratio`  | float | `0.2` | Ratio of samples to reserve for validation if enabled. |
| `--logging-dir` | string | `None` | Directory to save logs. If None, auto-creates 'logs'. |
| `--save` | string | `last` | Save model mode (`best`, `last`, `epoch`, `none`). |

Usage Examples:

```bash
# Basic Surrogate training isolating the "aeroplane" class
python v2/train.py --method surrogate --target aeroplane --epochs 20

# Surrogate training over the entire mixed dataset using a VGG16 extraction extractor
python v2/train.py --method surrogate --target mixing --extractor vgg16 --epochs 50

# ATS training with a 15-timestep simulation
python v2/train.py --method ats --target aeroplane --simulate 15
```

## Testing Usage

Test the saved agent policies using `test.py`. Evaluation is now strictly performed on the **VOC2007 test set** using `tensorflow_datasets` (TFDS) to ensure a clean train/test boundary from the VOC2012 training data. Please ensure you have `tensorflow` and `tensorflow_datasets` installed.

### Testing Parameters (`test.py`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--method` | string | (required) | SNN method to evaluate: `surrogate` or `ats`. |
| `--extractor` | string | `conv` | Feature extractor: `conv`, `vgg16`, `resnet18`, `fusion`, `vit`, `efficientnet`, or `mobilenet`. |
| `--target` | string | `mixing` | Target class for evaluation. |
| `--num-samples`| int | `10` | Number of samples to evaluate on from VOC2007 test. |
| `--replay` | int | `10` | History size. |
| `--max-steps` | int | `20` | Max steps per image. |
| `--simulate` | int | `10` | Number of simulation timesteps for the SNN. |
| `--weights` | string | `None` | Path to specific weights file. |
| `--logging-dir` | string | `None` | Directory to save logs. |

Usage Examples:

```bash
# Evaluate the Surrogate model on 50 random samples and log results
python v2/test.py --method surrogate --target mixing --num-samples 50 --random --logging-dir eval_logs

# Evaluate the ATS model using the VGG16 extractor
python v2/test.py --method ats --target mixing --extractor vgg16
```

## Visualization Usage

Visualize the agent's search path (Blue bounds -> Red bounds) using the `render.py` script.

### Rendering Parameters (`render.py`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--method` | string | (required) | SNN method to evaluate: `surrogate` or `ats`. |
| `--extractor` | string | `conv` | Feature extractor: `conv`, `vgg16`, `resnet18`, `fusion`, `vit`, `efficientnet`, or `mobilenet`. |
| `--target` | string | `mixing` | Target class for evaluation. |
| `--image-path` | string | `None` | Path to specific image file to render. |
| `--num-images`| int | `5` | Number of images to render if no path provided. |
| `--voc-dir` | string | `None` | Override default VOC2012 directory. |
| `--replay` | int | `10` | History size. |
| `--max-steps` | int | `20` | Max steps per image. |
| `--simulate` | int | `10` | Number of simulation timesteps for the SNN. |
| `--weights` | string | `None` | Path to specific weights file. |
| `--save` | flag | `False` | Save rendered images to disk. |
| `--save-dir` | string | `None` | Directory to save rendered images. |

Usage Examples:

```bash
# Render the Surrogate model search path for 5 images
python v2/render.py --method surrogate --target aeroplane --num-images 5

# Render the ATS model using the VGG16 extractor
python v2/render.py --method ats --target mixing --extractor vgg16
```

## Dataset 
Our dataset is the PASCAL VOC 2012 dataset, which is a collection of images of objects in different categories. Available at: https://drive.google.com/drive/folders/1ikKFR2nbdLw-W6cazaVYyYXCNUeXW6E7?usp=sharing