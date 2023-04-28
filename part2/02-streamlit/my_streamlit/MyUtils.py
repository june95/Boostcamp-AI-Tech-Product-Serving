import io
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


def transform_image(image):
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    tensor = transform(image).unsqueeze(0)
    return tensor