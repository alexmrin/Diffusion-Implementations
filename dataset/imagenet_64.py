import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

def prepare_imagenet64_loader(batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    def transform_features(example, transform):
        example['image'] = transform(example['image'])
        return example

    def load_dataset_split(split, transform):
        dataset = load_dataset("zh-plus/tiny-imagenet", split=split)
        dataset = dataset.map(lambda x: transform_features(x, transform), remove_columns=['image'], load_from_cache_file=False)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=6, drop_last=True)
    
    train_loader = load_dataset_split('train', train_transform)
    validation_loader = load_dataset_split('validation', test_transform)
    
    return train_loader, validation_loader