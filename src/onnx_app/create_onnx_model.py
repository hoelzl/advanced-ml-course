# %%
import torch
import torchvision.models as models
from torchvision.models import mobilenet
from torchvision.models.mobilenet import mobilenet_v2

# %%
model = mobilenet
torch.onnx.export(model,  torch.randn((1, 8, 512, 512)), 'seg_model.onnx')

# %%
