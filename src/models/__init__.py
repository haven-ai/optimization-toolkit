import torch

from torch import nn
from torch.nn import functional as F
import math


import torchvision.models as models
from . import classifier

def get_model(train_loader, exp_dict, device):
    return classifier.Classifier(train_loader, exp_dict, device)

    



