import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def download_and_extract_kaggle_dataset(dataset_name, download_path, extract_path):
    # Authenticate and initialize Kaggle API
    api = KaggleApi()

    # Download the dataset
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

    # Extract the dataset if it is in zip format
    for file in os.listdir(download_path):
        if file.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(download_path, file), 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            os.remove(os.path.join(download_path, file))

def prepare_cifar10_64_loader(batch_size=64):
    download_path='data/kaggle'
    extract_path='data/cifar10_64'
    # Download and extract the dataset
    dataset_name = 'joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution'
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
        download_and_extract_kaggle_dataset(dataset_name, download_path, extract_path)

    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(download_path, 'cifar10-64/train'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(download_path, 'cifar10-64/test'), transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    test_size = len(test_dataset) // 4
    indices = torch.randperm(len(test_dataset)).tolist()
    test_subset = Subset(test_dataset, indices[:test_size])

    # Create a dataloader for the test subset
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)

    return train_loader, test_loader