import torch
import torch.nn as nn
from torchvision.transforms import v2 as tfs
from PIL import Image
import os


# путь к уже обученной модели
MODEL_PATH = '/MT_NN/model_MT.tar'  # путь к сохранённым весам



class_names = {
    0: 'Rudeus',
    1: 'Eris',
    2: 'Sylphiette'
}


model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),

    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),

    nn.Flatten(),
    nn.Dropout(0.4),
    nn.Linear(256, 3)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

transform = tfs.Compose([
    tfs.Resize((128, 128)),
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True),
])

def determinant(photo):
    img = Image.open(photo).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    return class_names[pred_class]