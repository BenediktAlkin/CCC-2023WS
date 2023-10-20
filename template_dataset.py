import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CCCDataset(Dataset):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = Path(path).expanduser().with_suffix(".csv")
        assert self.path.exists(), f"'{self.path}' doesn't exist"

        # read csv
        x = []
        y = []
        with open(self.path) as f:
            reader = csv.reader(f, delimiter=",")

            for j, row in enumerate(reader):
                if j == 0:
                    continue
                x_rows.append(values)

        # to tensor
        self.x = torch.tensor(x_rows)
        print("initialized dataset")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if len(self.targets) == len(self.x):
            return self.x[item], self.targets[item]
        return self.x[item]
