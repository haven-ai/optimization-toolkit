import torch

from torch import nn
from torch.nn import functional as F
import math


import torchvision.models as models
from . import classifier, semseg

def get_model(train_loader, exp_dict, device):
    model_name = exp_dict['model_base']['name'] if type(exp_dict['model_base']) is dict else exp_dict['model_base']
    if model_name in ['semseg']:
        model =  semseg.SemSeg(train_loader, exp_dict, device)

        # load pretrained
        if 'pretrained' in exp_dict:
            model.load_state_dict(torch.load(exp_dict['pretrained']))
        return model
    else:
        return classifier.Classifier(train_loader, exp_dict, device)

    
def text_on_image(text, image, color=None):
    """Adds test on the image
    
    Parameters
    ----------
    text : [type]
        [description]
    image : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 0.8
    if color is None:
        fontColor              = (1,1,1)
    else:
        fontColor              = color
    lineType               = 1
    # img_mask = skimage.transform.rescale(np.array(img_mask), 1.0)
    # img_np = skimage.transform.rescale(np.array(img_points), 1.0)
    img_np = cv2.putText(image, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness=2
        # lineType
        )
    return img_np


