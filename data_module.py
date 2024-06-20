from d2l import torch as d2l
from torchvision import datasets, transforms
import torch


class MNISTDataModule(d2l.DataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.save_hyperparameters()
        self.X, self.y = self.load_data()
        self.num_train = 50000
        self.num_val = 10000

    def get_transforms(self):
        return transforms.Compose([transforms.ToTensor()])

    def load_data(self):
        dataset = datasets.MNIST(root="./datasets", train=True, transform=self.get_transforms())
        data_loader_all = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset))
        return next(iter(data_loader_all))

    def test_dataloader(self):
        return datasets.MNIST(root="./datasets", train=False, transform=self.get_transforms())

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=train)

