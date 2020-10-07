# %%
import torch
import onnxruntime
from PIL import Image
import sys
import numpy as np

# %%
session = onnxruntime.InferenceSession('mobilenet_v2.onnx')
input_name = session.get_inputs()[0].name
input_tensor = torch.tensor(np.array(Image.open(sys.argv[1]).resize((224, 224))), dtype=torch.float)
input_tensor = input_tensor.transpose(0, 2).unsqueeze(0).repeat(32, 1, 1, 1)
pred_onnx = session.run(None, {input_name: np.array(input_tensor)})
print(f"Prdicted {np.argmax(pred_onnx[0], axis=1)}")

# %%
