import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

batch_size = 32
PATH="./datasets/"


def read_voc_dataset(download=True, year='2007'):
    T = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                             #CustomRotation()
                            ])
    voc_data =  torchvision.datasets.VOCDetection(PATH, year=year, image_set='train', 
                        download=download, transform=T)
    train_loader = DataLoader(voc_data,shuffle=True)

    voc_val =  torchvision.datasets.VOCDetection(PATH, year=year, image_set='val', 
                        download=download, transform=T)
    val_loader = DataLoader(voc_val,shuffle=False)

    return voc_data, voc_val


