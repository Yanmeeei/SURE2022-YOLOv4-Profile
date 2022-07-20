import torch
from Neck import Yolov4

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
d5 = torch.load('d5.pt')
d4 = torch.load('d4.pt')
d3 = torch.load('d3.pt')

x20, x13, x6 = model.forward(d5, d4, d3)
model.forward(d5, d4, d3)
model.forward(d5, d4, d3)

t0 = time.time()
model.forward(d5, d4, d3)
t1 = time.time()

torch.save(x20, 'x20.pt')
torch.save(x13, 'x13.pt')
torch.save(x6, 'x6.pt')
print(f"Neck on agx: {t1 - t0}s")
