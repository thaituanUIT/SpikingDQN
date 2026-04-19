import torch
import torch.optim as optim
from torchvision.models import vgg16, VGG16_Weights
from RLSNN.convolutional.utility import load_data
from RLSNN.convolutional.learning import test_deep_q, train_deep_q
from RLSNN.convolutional.models import ConvDQN, SpikingWrapperV1, SpikingWrapperV2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_num = 1
    epoch = 10
    eps = 0.5
    
    image_list, bounding_box_list = load_data(obj_num)
    img_list, bb_list = load_data(obj_num, test=True)
    
    deep_q_model = ConvDQN().to(device)
    optimizer = optim.Adam(deep_q_model.parameters(), lr=1e-5)

    train_model = train_deep_q(epoch, eps, image_list, bounding_box_list, deep_q_model, optimizer)
    spiking_model = SpikingWrapperV2(train_model).to(device)
    test_deep_q(spiking_model, img_list, bb_list)

if __name__ == "__main__":
    main()