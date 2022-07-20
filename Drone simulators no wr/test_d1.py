import torch
from D1 import Yolov4

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

d1 = model.forward(input)
model.forward(input)
model.forward(input)

t0 = time.time()
model.forward(input)
t1 = time.time()
torch.save(d1, 'd1.pt')
print(f"D1 = {t1 - t0}s")
