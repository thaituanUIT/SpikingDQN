import torch
import torch.optim as optim
from torchvision.models import vgg16, VGG16_Weights
from RLSNN.convolutional.models import ConvDQN
from RLSNN.convolutional.utility import load_data
from RLSNN.convolutional.learning import train_deep_q
import os

def main():
    obj_num = 1
    epoch = 10
    eps = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_list, bounding_box_list = load_data(obj_num)
    
    deep_q_model = ConvDQN().to(device)
    optimizer = optim.Adam(deep_q_model.parameters(), lr=1e-5)

    train_model = train_deep_q(epoch, eps, image_list, bounding_box_list, deep_q_model, optimizer)

    save_dir = "pretrained_model"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_model.state_dict(), os.path.join(save_dir, "conv_deep_q_model.pth"))

if __name__ == "__main__":
    main()
