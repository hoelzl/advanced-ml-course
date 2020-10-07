# %%
import io
import json
from os import O_TRUNC
from flask import Flask, jsonify, request

import torchvision.transforms as transforms
from torchvision import models

from PIL import Image

app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet-class-index.json'))
model = models.densenet121(pretrained=True)
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    value, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return y_hat, imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
    print(f"Request: {request}, files: {dir(request)}")
    if request.files:
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})
    return "No files?"

@app.route("/hello")
def hello():
    return "Hello, world!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

