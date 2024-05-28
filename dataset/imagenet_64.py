import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data.dataloader import default_collate

class ImagenetDataset(Dataset):
    def __init__(self, dataset_name, split, transform=None):
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        # Apply the transform to the image if one is provided
        if self.transform:
            image = self.transform(image)
        
        # Check if the image has the desired size [3, 64, 64]
        if image.size() == torch.Size([3, 64, 64]):
            return image, label
        else:
            return None
        
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def prepare_imagenet64_loader(batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = ImagenetDataset('zh-plus/tiny-imagenet', 'train', train_transform)
    test_data = ImagenetDataset('zh-plus/tiny-imagenet', 'valid', test_transform)
    test_size = len(test_data) // 10
    indices = torch.randperm(len(test_data)).tolist()
    test_subset = Subset(test_data, indices[:test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=custom_collate_fn)
    validation_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, collate_fn=custom_collate_fn)
    return train_loader, validation_loader