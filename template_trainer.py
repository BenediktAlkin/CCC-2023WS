from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from time import time

def main():
    out = Path("out") / str(int(time()))
    print(out.as_posix())
    out.mkdir(exist_ok=True)
    device = torch.device("cuda")
    x = torch.randn(100, 16)
    y = torch.randint(5, size=(len(x),))
    train_ds = TensorDataset(x, y)
    test_ds = TensorDataset(x)
    train_dl = DataLoader(train_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=32)

    model = torch.nn.Linear(16, 5).to(device)
    optim = AdamW(model.parameters(), lr=1e-3)

    for epoch in range(10):
        print(f"epoch: {epoch + 1}")
        model.train()
        losses = []
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = model(x)
            optim.zero_grad()
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu())
        print(f"train loss: {torch.stack(losses).mean().item():.4f}")

        model.eval()
        with torch.no_grad():
            y_hats = []
            for batch in test_dl:
                x = batch[0].to(device, non_blocking=True)
                y_hat = model(x)
                y_hats.append(y_hat.cpu())
            y_hats = torch.concat(y_hats)

            # write
            lines = [y_hat.item() for y_hat in y_hats.argmax(dim=1)]
            with open(out / f"epoch{epoch + 1}.csv", "w") as f:
                lines = [f"{str(line)}\n" for line in lines[:-1]] + [str(lines[-1])]
                f.writelines(lines)


if __name__ == "__main__":
    main()
