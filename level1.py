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

class MyImageFolder(ImageFolder):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        with open(f"{root}/train_data_labels.csv") as f:
            self.y = f.readlines()
        self.y = torch.tensor([int(y.replace("\n", "")) for y in self.y])


    def __getitem__(self, idx):
        x, _ = super().__getitem__(idx)
        return x, self.y[idx]

def best_threshold(preds, target):
    # calculate f1
    precisions, recalls, thresholds = binary_precision_recall_curve(preds=preds, target=target)
    f1_scores = 2 * precisions * recalls / (precisions + recalls)
    # filter nan values
    not_nan_idxs = torch.logical_not(torch.isnan(f1_scores))
    f1_scores = f1_scores[not_nan_idxs]
    thresholds = thresholds[not_nan_idxs[:-1]]
    # get threshold of max f1 score
    # NOTE: argmax returns the first maximum, there might be an edge case where multiple optimal f1 scores exist
    best_f1_idx = f1_scores.argmax()
    assert len(thresholds) > 0, "only 1 f1_score found (no threshold)"
    best_threshold = thresholds[best_f1_idx].item()
    return best_threshold

def main():
    out = Path("out") / str(int(time()))
    print(out.as_posix())
    out.mkdir(exist_ok=True)
    device = torch.device("cuda")
    train_transform = Compose([RandomCrop(size=200, padding=10), RandomHorizontalFlip(), ToTensor()])
    train_ds = MyImageFolder("level_01/train_data", transform=train_transform)
    test_ds = ImageFolder("level_01/test_data", transform=ToTensor())
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = resnet18(num_classes=1).to(device)
    optim = SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(100):
        print(f"epoch: {epoch + 1}")
        model.train()
        losses = []
        y_hats = []
        ys = []
        for x, y in train_dl:
            ys.append(y.clone())
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = model(x)
            optim.zero_grad()
            loss = F.binary_cross_entropy(y_hat.squeeze(1).sigmoid(), y.float())
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu())
            y_hats.append(y_hat.squeeze(1).detach().cpu())
        print(f"train loss: {torch.stack(losses).mean().item():.4f}")
        y_hats = torch.concat(y_hats)
        ys = torch.concat(ys)
        train_auroc = binary_auroc(preds=y_hats, target=ys)
        print(f"train_auroc: {train_auroc.item():.4f}")
        best_thresh = best_threshold(preds=y_hats, target=ys)
        print(f"best thresh: {best_thresh:.4f}")
        acc_at_best_thresh = (y_hats.sigmoid() > best_thresh).sum() / len(y_hats)
        print(f"acc at best thresh: {acc_at_best_thresh.item():.4f}")

        if epoch % 5 != 0 or epoch == 0:
            continue

        #model.eval()
        with torch.no_grad():
            y_hats = []
            for batch in test_dl:
                x = batch[0].to(device, non_blocking=True)
                y_hat = model(x)
                y_hats.append(y_hat.cpu())
            y_hats = torch.concat(y_hats)
            y_hats_raw = y_hats
            y_hats = (y_hats.sigmoid() > best_thresh).long()
            print(f"zero preds: {(y_hats == 0).sum()}")

            # write
            lines = [y_hat.item() for y_hat in y_hats]
            with open(out / f"epoch{epoch + 1}.csv", "w") as f:
                lines = [f"{str(line)}\n" for line in lines[:-1]] + [str(lines[-1])]
                f.writelines(lines)


if __name__ == "__main__":
    main()
