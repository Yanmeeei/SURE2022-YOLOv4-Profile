import torch
from D5 import Yolov4

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
d4 = torch.load('d4.pt')
model.forward(d4)
model.forward(d4)
model.forward(d4)

t0 = time.time()
model.forward(d4)
t1 = time.time()

print(f"D5 = {t1 - t0}s")
