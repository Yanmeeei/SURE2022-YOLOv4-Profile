import torch
from models import Yolov4, YoloLayer, Yolov4Head

from timer import Clock
from memorizer import MemRec

tt = Clock()
mr = MemRec()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


# create a model
model = Yolov4()
model.to(device)
print("YOLOv4 is Ready")

n = 15
print(f"input size: {n}")
input = torch.randn(n, 3, 244, 244)
model.forward(input, tt, mr)

tt.report(sample=False)
mr.report(sample=False)