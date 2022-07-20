import torch
import io
from models import Yolov4, YoloLayer, Yolov4Head

m = torch.jit.script(Yolov4())

# Save to file
torch.jit.save(m, 'scriptmodule.pt')

# Save to io.BytesIO buffer
buffer = io.BytesIO()
torch.jit.save(m, buffer)

# Save with extra files
extra_files = {'foo.txt': b'bar'}
torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)