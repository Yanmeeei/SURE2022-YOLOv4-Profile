
import torch
from drive/My Drive/ColabNotebooks/SURE2022/pytorch-YOLOv4-master/models import Yolov4, YoloLayer, Yolov4Head

from drive/My Drive/ColabNotebooks/SURE2022/pytorch-YOLOv4-master/timer import Clock
from drive/My Drive/ColabNotebooks/SURE2022/pytorch-YOLOv4-master/memorizer import MemRec

tt = Clock()
mr = MemRec()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
if device != 'cpu':
  usingcuda = True
else:
  usingcuda = False

# create a model
model = Yolov4()
model.to(device)
print("YOLOv4 is Ready")

n = 15
print(f"input size: {n}")
input = torch.randn(n, 3, 244, 244).to(device)
model.forward(input, tt, mr, usingcuda)

tt.report(sample=False)
mr.report(sample=False)