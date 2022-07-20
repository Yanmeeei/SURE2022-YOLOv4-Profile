import torch
from D4 import Yolov4

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
d3 = torch.load('d3.pt')
d4 = model.forward(d3)
model.forward(d3)
model.forward(d3)
t0 = time.time()
model.forward(d3)
t1 = time.time()
torch.save(d4, 'd4.pt')
print(f"D4 = {t1 - t0}s")
