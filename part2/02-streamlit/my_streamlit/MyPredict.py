import torch
import streamlit as st
import clip
from MyUtils import transform_image
import yaml
from typing import Tuple
import math

@st.cache
def load_model():
    base_model, preprocess = clip.load('ViT-B/32', 'cpu')
    state_dict = torch.load('./aug_t_1_seed34_0_epoch20.pt', map_location=torch.device('cpu'))
    model = get_model_from_sd(state_dict, base_model)
    return model

def get_model_from_sd(state_dict, base_model):
    
    if not 'classification_head.weight' in state_dict : 
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
            state_dict = new_state_dict

    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    model = model.cpu()
    device = torch.device('cpu')
    return torch.nn.DataParallel(model,  device_ids=device)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits
    
def get_prediction(model, image) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = transform_image(image=image).to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return tensor, y_hat