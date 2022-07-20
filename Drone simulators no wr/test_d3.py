import torch
from D3 import Yolov4

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
d2 = torch.load('d2.pt')
d3 = model.forward(d2)
model.forward(d2)
model.forward(d2)
t0 = time.time()
model.forward(d2)
t1 = time.time()
torch.save(d3, 'd3.pt')
print(f"D3 = {t1 - t0}s")