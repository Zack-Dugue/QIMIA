import torchvision as tv
import torch as th
from torch.utils import data as data_utils
from torchvision import transforms, datasets

# def get_ImageNet_dataloader():

def get_CIFAR10_dataloader(image_size, batch_size,num_dataloader_workers=0):

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load data
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

# Package it up in batches
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_dataloader_workers,pin_memory=True)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_dataloader_workers,pin_memory=True)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_dataloader_workers,pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

PATH = "./ImageNetData"
def get_imageNet_dataloader(image_size, batch_size,num_dataloader_workers=0):

    train_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.AutoAugment(interpolation = transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load data
    train_dataset = datasets.ImageNet(root=PATH, split="train",transform=train_transform)
    val_dataset = datasets.ImageNet(root=PATH, split="val",transform=val_transform)
    test_dataset = datasets.ImageNet(root=PATH,split="val",transform=val_transform)
    print(f"length of train_dataset is {len(train_dataset)}")
# Package it up in batches
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_dataloader_workers,pin_memory=True)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_dataloader_workers,pin_memory=True)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_dataloader_workers,pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader