import torch
import os
import json
from PIL import Image

import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MushokuTenseiDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, 'train_mt' if train else 'test_mt')
        self.transform = transform

        with open(os.path.join(self.path, 'format_mt.json'), 'r') as fp:
            self.format = json.load(fp)

        self.length = len(self.format)
        self.files = tuple(self.format.keys())
        self.targets = tuple(self.format.values())

    def __getitem__(self, item):
        path_file = os.path.join(self.path, self.files[item])
        img = Image.open(path_file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.targets[item], dtype=torch.long)

    def __len__(self):
        return self.length


model = nn.Sequential(
    # Блок 1
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   # 128 → 64

    # Блок 2
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   # 64 → 32

    # Блок 3
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   # 32 → 16

    # Блок 4
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),  # → 256x1x1

    nn.Flatten(),
    nn.Dropout(0.4),
    nn.Linear(256, 3)
)


transforms = tfs.Compose([tfs.Resize((128, 128)),tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
d_train = MushokuTenseiDataset('/MT_NN', transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

optimizer = optim.Adam(params=model.parameters(), lr=0.0005, weight_decay=1e-4)
loss_function = nn.CrossEntropyLoss()
epochs = 25
model.train()

for epoch in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 -1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{epoch+1}/{epochs}], loss_mean={loss_mean:.3f}")


st = model.state_dict()
torch.save(st, 'model_MT.tar')

d_test = MushokuTenseiDataset('/MT_NN', train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

Q = 0
count = 0
model.eval()

test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)
        Q += loss_function(p, y_test).item()
        count += 1

Q /= count
print(Q)