import torch
from D2 import Yolov4

import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if device != 'cpu':
    usingcuda = True
else:
    usingcuda = False

# create a model
model = Yolov4()
model.to(device)
print("YOLOv4 is Ready")

n = 1
print(f"input size: {n}")
input = torch.randn(n, 3, 608, 608).to(device)
d1 = torch.load('d1.pt')
model.forward(d1)
model.forward(d1)
model.forward(d1)

t0 = time.time()
model.forward(d1)
t1 = time.time()

print(f"D2 = {t1 - t0}s")