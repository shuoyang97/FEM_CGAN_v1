from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from FEM_Dataset import FEM_Dataset

def dataloader(dataset_path, input_size, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset = FEM_Dataset(dataset_path, input_size, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

