import torch
from models import Yolov4, YoloLayer, Yolov4Head
from ptflops import get_model_complexity_info

from profilerwrapper import ProfilerWrapper

prof_wrapper = ProfilerWrapper()

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
model.forward(input, prof_wrapper, usingcuda)

prof_wrapper.report(sample=False)
