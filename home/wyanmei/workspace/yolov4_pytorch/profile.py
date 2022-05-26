import torch
from torch.profiler import profile, record_function, ProfilerActivity

import models

model = models.Yolov4()
inputs = torch.randn(50, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

# print("==>> cpu_time_total")
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print("==>> self_cpu_memory_usage")
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# print("==>> cpu_memory_usage")
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

