from torch.utils import data
import torchvision

class Dataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.train_dataset = torchvision.datasets.MNIST(
            root='./',
            train=True,
            # transform=transform,
            download=True
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root='./',
            train=False,
            # transform=test_transform,
            download=True
        )

    def __getitem__(self, index):
        return super().__getitem__(index)
    def __len__(self):
        pass


if __name__ == '__main__':
    
    pass