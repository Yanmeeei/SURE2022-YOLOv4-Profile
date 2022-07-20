import torch
from entire_model import Yolov4

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
model.forward(input)
model.forward(input)

t0 = time.time()
model.forward(input)
t1 = time.time()

print(f"Entire model on agx: {t1 - t0}s")
