import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
from RLSNN.vanilla.utility import load_data
from RLSNN.vanilla.vision import feature_extract, compute_mask, compute_iou, crop_image
import RLSNN.vanilla.parameters as parameters

def compute_q(feature, deep_q_model):
    device = next(deep_q_model.parameters()).device
    feature_tensor = torch.FloatTensor(feature).to(device)
    deep_q_model.eval()
    with torch.no_grad():
        output = deep_q_model(feature_tensor)
    deep_q_model.train()
    return output.cpu().numpy().flatten()

def compute_reward(action, ground_truth, current_mask):
    new_mask = compute_mask(action, current_mask)
    iou_new = compute_iou(new_mask, ground_truth)
    iou_current = compute_iou(current_mask, ground_truth)

    if iou_current < iou_new:
        return 1
    else:
        return -1

def compute_end_reward(current_mask, ground_truth):
    if compute_iou(current_mask, ground_truth) > 0.5:
        return 3
    else:
        return -3

def compute_target(reward, new_feature, model):
    return reward + parameters.gamma * np.amax(compute_q(new_feature, model))

def select_action(feature, ground_truth_box, step, q_value, epsilon, current_mask):
    if step == parameters.max_steps:
        action = 8 #select trigger if agent surpassed maximum number of steps

    else:
        if random.random() > epsilon:
            action = np.argmax(q_value)
        else:
            end_reward = compute_end_reward(current_mask, ground_truth_box)
            if end_reward > 0:
                action = 8
            else:
                rewards = []
                for i in range(parameters.action_option - 1):
                    reward = compute_reward(i, ground_truth_box, current_mask)
                    rewards.append(reward)
                rewards = np.asarray(rewards)
                positive_reward_index = np.where(rewards >= 0)[0]

                if len(positive_reward_index) == 0:
                    positive_reward_index = np.asarray(range(9))

                action = np.random.choice(positive_reward_index)

    return action


def execute_action(action, history, ground_truth_box, current_mask):
    if action == 8:
        new_mask = current_mask
        reward = compute_end_reward(current_mask, ground_truth_box)
        end = True
    else:
        new_mask = compute_mask(action, current_mask)
        reward = compute_reward(action, ground_truth_box, current_mask)
        history = history[1:]
        history.append(action)
        end = False

    return new_mask, reward, end, history

def experience_replay(deep_q_model, optimizer, experience):
    sample = random.choices(experience, k=parameters.experience_sample_size)

    targets = np.zeros((parameters.experience_sample_size, parameters.action_option))

    for i in range(parameters.experience_sample_size):
        feature, action, new_feature, reward, end = sample[i]
        target = reward

        if not end:
            target = compute_target(reward, new_feature, deep_q_model)

        targets[i, :] = compute_q(feature, deep_q_model)
        targets[i][action] = target

    x = np.concatenate([each[0] for each in sample])

    optimizer.zero_grad()
    device = next(deep_q_model.parameters()).device
    x_tensor = torch.FloatTensor(x).to(device)
    targets_tensor = torch.FloatTensor(targets).to(device)
    
    outputs = deep_q_model(x_tensor)
    criterion = nn.HuberLoss()
    loss = criterion(outputs, targets_tensor)
    loss.backward()
    optimizer.step()

    parameters.loss_arr.append(loss.item())
    if len(parameters.loss_arr) == 100:
        print("loss %s" % str(sum(parameters.loss_arr) / len(parameters.loss_arr)))
        parameters.loss_arr = []


def train_deep_q(training_epoch, epsilon, image_list, bounding_box_list, deep_q_model, vgg16):
    experience = []
    optimizer = torch.optim.Adam(deep_q_model.parameters(), lr=1e-5)

    for current_epoch in range(1, training_epoch + 1):

        print("Now starting epoch %d" % current_epoch)
        training_set_size = np.shape(image_list)[0]

        for i in range(30):
            image = image_list[i]
            ground_truth_box = bounding_box_list[i]
            history = [-1] * parameters.history_size
            height, width, channel = np.shape(image)
            current_mask = np.asarray([0, 0, width, height])
            feature = feature_extract(image, history, vgg16)
            end = False
            step = 0
            total_reward = 0

            while not end:
                q_value = compute_q(feature, deep_q_model)
                action = select_action(feature, ground_truth_box, step, q_value, epsilon, current_mask)
                new_mask, reward, end, history = execute_action(action, history, ground_truth_box, current_mask)
                cropped_image = crop_image(image, new_mask)
                new_feature = feature_extract(cropped_image, history, vgg16)
                if len(experience) > parameters.max_experience_size:
                    experience = experience[1:]
                    experience.append([feature, action, new_feature, reward, end])
                else:
                    experience.append([feature, action, new_feature, reward, end])

                experience_replay(deep_q_model, optimizer, experience)
                feature = new_feature
                current_mask = new_mask
                step += 1
                total_reward += reward

            print("Image %d, total reward %i" % (i, total_reward))

        if current_epoch < parameters.epsilon_change_steps:
            epsilon -= 0.1
            print("current epsilon is %f" % epsilon)

    return deep_q_model

def pre_test(vgg16_model, dqn_weight, spiking=False):
    object_number = 1
    image_list, bounding_box_list = load_data(object_number , test=True)
    iou = []
    if spiking:
        deep_q = SpikingDQN()
    else:
        deep_q = DQN()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deep_q.load_state_dict(torch.load(dqn_weight, map_location=device, weights_only=True))
    deep_q.to(device)

    for i in range(0,100):
        bounding_box = bounding_box_list[i]
        image = image_list[i]
        history = [-1] * parameters.history_size
        height, width, channel = np.shape(image)
        current_mask = np.asarray([0, 0, width, height])
        feature = feature_extract(image, history, vgg16_model)
        end = False
        masks = []
        step = 0

        while not end:

            q_value = compute_q(feature, deep_q)

            action = np.argmax(q_value)

            history = history[1:]
            history.append(action)

            if action == 8 or step == 10:
                end = True
                print("end")
                new_mask = current_mask
                cv2.rectangle(image, (int(bounding_box[0]), int(bounding_box[1])),
                              (int(bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 1)
                import os
                if not os.path.exists("./result"):
                    os.makedirs("./result")
                cv2.imwrite("./result/plane_result%d.jpg" % i, image)
            else:
                new_mask = compute_mask(action, current_mask)

            cropped_image = crop_image(image, new_mask)
            feature = feature_extract(cropped_image, history, vgg16_model)

            masks.append(new_mask)
            current_mask = new_mask
            cv2.rectangle(image, (int(current_mask[0]), int(current_mask[1])),
                          (int(current_mask[2]), int(current_mask[3])), (0, 255, 0), 1)
            step += 1

        mask = masks[-1] if len(masks) > 0 else current_mask
        iou.append(compute_iou(mask,bounding_box))


    print(sum(iou)/len(iou) if len(iou) > 0 else 0)
    cv2.rectangle(image, (int(mask[0]), int(mask[1])),
                   (int(mask[2]),int(mask[3])),(0, 255, 0), 2)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def post_test(vgg16_model, dqn_weight, spiking=False):
    object_number = 1
    image_list, bounding_box_list = load_data(object_number , test=True)
    iou = []
    if spiking:
        deep_q = SpikingDQN()
    else:
        deep_q = DQN()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deep_q.load_state_dict(torch.load(dqn_weight, map_location=device, weights_only=True))
    deep_q.to(device)

    for i in range(30,40):
        bounding_box = bounding_box_list[i]
        image = image_list[i]
        history = [-1] * parameters.history_size
        height, width, channel = np.shape(image)
        current_mask = np.asarray([0, 0, width, height])
        feature = feature_extract(image, history, vgg16_model)
        end = False
        masks = []
        step = 0
        
        while not end:
            q_value = compute_q(feature, deep_q)
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
                plt.title('predicted box is shown in red and ground truth box is shown in green \n Search path shown in blue')
                plt.show()

            else:
                new_mask = compute_mask(action, current_mask)

            cropped_image = crop_image(image, new_mask)
            feature = feature_extract(cropped_image, history, vgg16_model)

            masks.append(new_mask)
            current_mask = new_mask
            cv2.rectangle(image, (int(current_mask[0]), int(current_mask[1])),
                          (int(current_mask[2]), int(current_mask[3])), (255, 0, 0), 1)
            step += 1

        mask = masks[-1] if len(masks) > 0 else current_mask
        iou.append(compute_iou(mask,bounding_box))

    print(sum(iou)/len(iou) if len(iou) > 0 else 0)