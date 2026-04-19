import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from dataclasses import dataclass, field
from RLSNN.convolutional.action import compute_iou, compute_mask, crop_image
import cv2
import matplotlib.pyplot as plt

@dataclass
class Parameters:
    history_size: int = 10
    action_option: int = 9
    max_steps: int = 20
    experience_sample_size: int = 20
    max_experience_size: int = 1000
    gamma: float = 0.1
    epsilon_change_steps: int = 10
    loss_arr: list = field(default_factory=list)

params = Parameters()

def _plot():
    plt.figure(figsize=(8, 5))
    plt.plot(params.loss_arr, label="Loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def _feature_extract(img, history):
    feat_hist = np.zeros(params.action_option * params.history_size)
    for i in range(params.history_size):
        if history[i] != -1:
            feat_hist[i * params.action_option + history[i]] = 1
    history_tensor = torch.tensor(feat_hist).float().unsqueeze(0)

    img_resized = cv2.resize(img, (224, 224))
    img_transposed = np.transpose(img_resized, (2, 0, 1)) 
    image_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float() 
    image_tensor = image_tensor / 255.0
    
    return image_tensor, history_tensor

def _compute_q(dqn, image_tensor, history_tensor):
    device = next(dqn.parameters()).device
    image_tensor = image_tensor.to(device)
    history_tensor = history_tensor.to(device)
    dqn.eval()
    with torch.no_grad():
        output = dqn(image_tensor, history_tensor)
    dqn.train()
    return output.cpu().numpy().flatten()

def _compute_reward(action, truth, curr_mask):
    new_mask = compute_mask(action, curr_mask)
    iou_new = compute_iou(new_mask, truth)
    iou_current = compute_iou(curr_mask, truth)

    if iou_current < iou_new:
        return 1
    else:
        return -1

def _compute_finish_reward(dqn, truth, curr_mask):
    if compute_iou(curr_mask, truth) > 0.5:
        return 3
    else:
        return -3

def _compute_target(dqn, reward, new_feature):
    return reward + params.gamma * np.amax(_compute_q(dqn, new_feature[0], new_feature[1]))

def _select_action(dqn, feature, truth, step, q_value, epsilon, curr_mask):
    if step == params.max_steps:
        action = 8
    else:
        if random.random() > epsilon:
            action = np.argmax(q_value)
        else:
            finish_reward = _compute_finish_reward(dqn, truth, curr_mask)
            if finish_reward > 0:
                action = 8
            else:
                rewards = []
                for i in range(params.action_option - 1):
                    reward = _compute_reward(i, truth, curr_mask)
                    rewards.append(reward)
                rewards = np.asarray(rewards)
                positive_reward_index = np.where(rewards >= 0)[0]

                if len(positive_reward_index) == 0:
                    positive_reward_index = np.asarray(range(9))

                action = np.random.choice(positive_reward_index)

    return action

def _execute_action(dqn, action, history, truth, curr_mask):
    if action == 8:
        new_mask = curr_mask
        reward = _compute_finish_reward(dqn, truth, curr_mask)
        end = True
    else:
        new_mask = compute_mask(action, curr_mask)
        reward = _compute_reward(action, truth, curr_mask)
        history = history[1:]
        history.append(action)
        end = False

    return new_mask, reward, end, history

def _experience_replay(dqn, optimizer, experience):
    sample = random.choices(experience, k=params.experience_sample_size)

    targets = np.zeros((params.experience_sample_size, params.action_option))

    for i in range(params.experience_sample_size):
        feature, action, new_feature, reward, end = sample[i]
        target = reward

        if not end:
            target = _compute_target(dqn, reward, new_feature)

        targets[i, :] = _compute_q(dqn, feature[0], feature[1])
        targets[i][action] = target

    # Combine all images and histories into respective batches
    x_images = torch.cat([each[0][0] for each in sample], dim=0)
    x_histories = torch.cat([each[0][1] for each in sample], dim=0)

    optimizer.zero_grad()
    device = next(dqn.parameters()).device
    x_images = x_images.to(device)
    x_histories = x_histories.to(device)
    targets_tensor = torch.FloatTensor(targets).to(device)
    
    outputs = dqn(x_images, x_histories)
    criterion = nn.HuberLoss()
    loss = criterion(outputs, targets_tensor)
    loss.backward()
    optimizer.step()

    params.loss_arr.append(loss.item())
    if len(params.loss_arr) == 100:
        print("loss %s" % str(sum(params.loss_arr) / len(params.loss_arr)))
        params.loss_arr = []

def train_deep_q(training_epoch, epsilon, image_list, bounding_box_list, dqn, optimizer):
    experience = []
    for current_epoch in range(1, training_epoch + 1):
        print("Now starting epoch %d" % current_epoch)
        training_set_size = np.shape(image_list)[0]
        for i in range(training_set_size):
            image = image_list[i]
            ground_truth_box = bounding_box_list[i]
            history = [-1] * params.history_size
            height, width, channel = np.shape(image)
            current_mask = np.asarray([0, 0, width, height])
            feature = _feature_extract(image, history)
            end = False
            step = 0
            total_reward = 0
            while not end:
                q_value = _compute_q(dqn, feature[0], feature[1])
                action = _select_action(dqn, feature, ground_truth_box, step, q_value, epsilon, current_mask)
                new_mask, reward, end, history = _execute_action(dqn, action, history, ground_truth_box, current_mask)
                cropped_image = crop_image(image, new_mask)
                new_feature = _feature_extract(cropped_image, history)
                if len(experience) > params.max_experience_size:
                    experience = experience[1:]
                    experience.append([feature, action, new_feature, reward, end])
                else:
                    experience.append([feature, action, new_feature, reward, end])

                _experience_replay(dqn, optimizer, experience)
                feature = new_feature
                current_mask = new_mask
                step += 1
                total_reward += reward

            print("Image %d, total reward %i" % (i, total_reward))

        if current_epoch < params.epsilon_change_steps:
            epsilon -= 0.1
            print("current epsilon is %f" % epsilon)

    return dqn

def test_deep_q(dqn, image_list, bounding_box_list, pretrained_weight=None):
    iou = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pretrained_weight:
        dqn.load_state_dict(torch.load(pretrained_weight, map_location=device))
    dqn.eval().to(device)

    for i in range(30,40):
        bounding_box = bounding_box_list[i]
        image = image_list[i]
        # History is a list of previous actions (-1 initially means no action)
        history = [-1] * params.history_size
        height, width, channel = np.shape(image)
        current_mask = np.asarray([0, 0, width, height])
        feature = _feature_extract(image, history)
        end = False
        masks = []
        step = 0
        
        while not end:
            q_value = _compute_q(dqn, feature[0], feature[1])
            action = np.argmax(q_value)
            history = history[1:]
            history.append(action)

            if action == 8 or step == 10: #steps should be changed to 40
                end = True
                plt.figure()
                new_mask = current_mask
                cv2.rectangle(image, (int(new_mask[0]), int(new_mask[1])),
                              (int(new_mask[2]), int(new_mask[3])), (255, 0, 0), 1)
                predicted_box = cv2.rectangle(image, (int(new_mask[0]), int(new_mask[1])),
                              (int(new_mask[2]), int(new_mask[3])), (0, 0, 255), 2)
                groundtruth= cv2.rectangle(image, (int(bounding_box[0]), int(bounding_box[1])),
                              (int(bounding_box[2]), int(bounding_box[3])), (0, 255, 0), 2)
                test_result = cv2.bitwise_and(image, image, groundtruth)
                
                plt.imshow(cv2.cvtColor(test_result, cv2.COLOR_BGR2RGB))
                plt.title('Predicted box is shown in red and ground truth box is shown in green \n Search path shown in blue')
                plt.show()

            else:
                new_mask = compute_mask(action, current_mask)

            cropped_image = crop_image(image, new_mask)
            feature = _feature_extract(cropped_image, history)

            masks.append(new_mask)
            current_mask = new_mask
            cv2.rectangle(image, (int(current_mask[0]), int(current_mask[1])),
                          (int(current_mask[2]), int(current_mask[3])), (255, 0, 0), 1)
            step += 1

        mask = masks[-1] if len(masks) > 0 else current_mask
        iou.append(compute_iou(mask,bounding_box))

    print(sum(iou)/len(iou) if len(iou) > 0 else 0)