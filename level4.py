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
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm
from torchvision.ops.focal_loss import sigmoid_focal_loss

from gaussian_blur import get_gaussian_blur
class TwoImageFolder(Dataset):
    def __init__(self, root, ctor, transform, *args, **kwargs):
        super().__init__()
        self.ds1 = ctor(root=root, *args, **kwargs, is_valid_file=lambda fname: fname.endswith("0.png"))
        self.ds2 = ctor(root=root, *args, **kwargs, is_valid_file=lambda fname: fname.endswith("1.png"))
        assert len(self.ds1) == len(self.ds2)
        self.transform = transform

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, item):
        x1, y1 = self.ds1[item]
        x2, y2 = self.ds2[item]
        x1 = to_tensor(x1)
        x2 = to_tensor(x2)
        img = torch.min(x1, x2)
        xcoord = torch.arange(img.size(1)).unsqueeze(1).repeat(1, img.size(1))
        # ycoord = torch.arange(img.size(2)).unsqueeze(0).repeat(img.size(2), 1)
        # img = torch.concat([img, xcoord.unsqueeze(0), ycoord.unsqueeze(0)])

        y = torch.zeros_like(xcoord)

        if isinstance(y1, int):
            y = 0
        else:
            padding = 7
            y[y1[1] - padding:y1[1] + padding, y1[0] - padding:y1[0] + padding] = 1

        return self.transform(img), y, y1



class MyImageFolder(ImageFolder):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        with open(f"{root}/train_data_labels.csv") as f:
            self.y = f.readlines()
        y = []
        for yy in self.y:
            split = yy.replace("\n", "").split(",")
            y.append([int(split[0]), int(split[1])])
        self.y = torch.tensor(y)


    def __getitem__(self, idx):
        x, _ = super().__getitem__(idx)
        return x, self.y[idx]


def y_hat_to_coord(y_hat, blur):
    with torch.no_grad():
        blurred = blur(y_hat.unsqueeze(1)).squeeze(1)
        max_of_blurred = blurred.flatten(start_dim=1).max(dim=1).values
        coords = (blurred == max_of_blurred[:, None, None]).nonzero()[:, 1:]
    return coords

def main():
    out = Path("out") / str(int(time()))
    print(out.as_posix())
    out.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda")
    train_transform = nn.Identity()
    train_ds = TwoImageFolder(ctor=MyImageFolder, root="level_04/train_data", transform=train_transform)
    test_ds = TwoImageFolder(ctor=ImageFolder, root="level_04/test_data", transform=nn.Identity())
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)
    blur = get_gaussian_blur().to(device)
    # model = resnet18(num_classes=2)
    # model.conv1 = nn.Conv2d(5, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    # model.maxpool = nn.Identity()
    # model.layer2[0].downsample = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    # model.layer3[0].downsample = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    #model.avgpool = nn.Conv2d(512, 2, kernel_size=13)
    from unet import Unet
    model = Unet(dim=32, depth=2)
    model = model.to(device)
    optim = AdamW(model.parameters(), lr=0.001, weight_decay=0)

    for epoch in range(100):
        print(f"epoch: {epoch + 1}")
        model.train()
        losses = []
        cdists = []
        for x, y, coords in tqdm(train_dl):
            # ys.append(y.clone())
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)
            y_hat = model(x).squeeze(1)
            optim.zero_grad()
            unreduced_loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction="none")
            neg_loss = unreduced_loss[y == 0].mean()
            pos_loss = unreduced_loss[y == 1].mean()
            loss = neg_loss + pos_loss * 100
            #loss = sigmoid_focal_loss(y_hat, y.float(), reduction="mean")
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu())
            # get prediction
            coords_hat = y_hat_to_coord(y_hat=y_hat, blur=blur)
            cdist = ((coords - coords_hat) ** 2).sum(dim=1).float().mean()
            cdists.append(cdist.cpu())
        print(f"train loss: {torch.stack(losses).mean().item():.4f}")
        print(f"cdists: {torch.stack(cdists).mean().item():.4f}")
        # y_hats = torch.concat(y_hats)
        # ys = torch.concat(ys)
        # cdist = ((y_hats - ys.float()) ** 2).sum(dim=1).mean()
        # print(f"cdist: {cdist.item():.4f}")


        if (epoch + 1) % 5 != 0 or epoch == 0:
            continue

        #model.eval()
        with torch.no_grad():
            y_hats = []
            for x, _ in test_dl:
                x = x.to(device, non_blocking=True)
                y_hat = model(x).squeeze(1)
                coords_hat = y_hat_to_coord(y_hat=y_hat, blur=blur)
                y_hats.append(coords_hat.cpu())
            y_hats = torch.concat(y_hats)
            y_hats = y_hats.round().long()

            # write
            lines = [f"{y_hat[1].item()},{y_hat[0].item()}" for y_hat in y_hats]
            with open(out / f"epoch{epoch + 1}.csv", "w") as f:
                lines = [f"{str(line)}\n" for line in lines[:-1]] + [str(lines[-1])]
                f.writelines(lines)


if __name__ == "__main__":
    main()
