from torch import nn
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.utils.data import TensorDataset, DataLoader
from time import time
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomHorizontalFlip
import csv
from torchvision.models import resnet18
from torchmetrics.functional.classification import binary_auroc, binary_precision_recall_curve
from torch.utils.data import Dataset

class TwoImageFolder(Dataset):
    def __init__(self, root, ctor, *args, **kwargs):
        super().__init__()
        self.ds1 = ctor(root=root, *args, **kwargs, is_valid_file=lambda fname: fname.endswith("0.png"))
        self.ds2 = ctor(root=root, *args, **kwargs, is_valid_file=lambda fname: fname.endswith("1.png"))
        assert len(self.ds1) == len(self.ds2)

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, item):
        x1, y1 = self.ds1[item]
        x2, y2 = self.ds1[item]
        assert y1 == y2
        return torch.min(x1, x2), y1



class MyImageFolder(ImageFolder):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        with open(f"{root}/train_data_labels.csv") as f:
            self.y = f.readlines()
        self.y = torch.tensor([int(y.replace("\n", "")) for y in self.y])


    def __getitem__(self, idx):
        x, _ = super().__getitem__(idx)
        return x, self.y[idx]


def main():
    out = Path("out") / str(int(time()))
    print(out.as_posix())
    out.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda")
    train_transform = Compose([RandomCrop(size=200, padding=10), RandomHorizontalFlip(), ToTensor()])
    train_ds = TwoImageFolder(ctor=MyImageFolder, root="level_03/train_data", transform=train_transform)
    test_ds = TwoImageFolder(ctor=ImageFolder, root="level_03/test_data", transform=ToTensor())
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = resnet18(num_classes=1)
    #model.conv1 = nn.Conv2d(6, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)
    optim = SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(100):
        print(f"epoch: {epoch + 1}")
        model.train()
        losses = []
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = model(x)
            optim.zero_grad()
            loss = F.mse_loss(y_hat.squeeze(1), y.float())
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu())
        print(f"train loss: {torch.stack(losses).mean().item():.4f}")

        if epoch % 5 != 0 or epoch == 0:
            continue

        #model.eval()
        with torch.no_grad():
            y_hats = []
            for x, _ in test_dl:
                x = x.to(device, non_blocking=True)
                y_hat = model(x)
                y_hats.append(y_hat.cpu())
            y_hats = torch.concat(y_hats)
            y_hats = y_hats.round()

            # write
            lines = [y_hat.item() for y_hat in y_hats]
            with open(out / f"epoch{epoch + 1}.csv", "w") as f:
                lines = [f"{str(line)}\n" for line in lines[:-1]] + [str(lines[-1])]
                f.writelines(lines)


if __name__ == "__main__":
    main()
