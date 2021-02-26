import torch

from torch import nn
from torch.nn import functional as F
import math


import torchvision.models as models
from . import classifier, semseg

def get_model(train_loader, exp_dict, device):
    model_name = exp_dict.get('model', 'classifier')
    if model_name == 'classifier':
        return classifier.Classifier(train_loader, exp_dict, device)
    if model_name == 'semseg':
        return semseg.SemSeg(train_loader, exp_dict, device)

    



