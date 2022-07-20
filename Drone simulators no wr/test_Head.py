import torch
from Head import Yolov4

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

x20 = torch.load('x20.pt')
x13 = torch.load('x13.pt')
x6 = torch.load('x6.pt')
output = model.forward(x20, x13, x6)
model.forward(x20, x13, x6)
model.forward(x20, x13, x6)

t0 = time.time()
model.forward(x20, x13, x6)
t1 = time.time()

torch.save(output, 'output.pt')

print(f"Head on agx: {t1 - t0}s")
