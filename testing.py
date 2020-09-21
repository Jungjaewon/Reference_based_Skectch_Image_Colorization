from torchvision.models import vgg19
import torch
import numpy as np
from utils import get_outline_image
from utils import elastic_transform
from PIL import Image
from model import Encoder
from model import SCFT_Moudle
import cv2
import torch.nn as nn
vgg_activation = dict()

def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output.detach()

    return hook

if __name__ == '__main__':
    pass
    """
    model = vgg19()
    model.features[3].register_forward_hook(get_activation(str('relu_3')))
    model.features[8].register_forward_hook(get_activation(str('relu_8')))
    model.features[17].register_forward_hook(get_activation(str('relu_17')))
    model.features[26].register_forward_hook(get_activation(str('relu_26')))
    model.features[35].register_forward_hook(get_activation(str('relu_35')))

    print(model.features)
    
    input = torch.rand((1,3,244,244))
    result = model(input)
    print(vgg_activation)
    """

    #result = get_outline_image(r"C:\Users\woodc\Desktop\deep_paint_split\train\1020_color.png").convert('RGB')
    #result = np.array(result)
    #result = Image.open(r"C:\Users\woodc\Desktop\deep_paint_split\train\1038_color.png").convert("RGB")
    """
    result = np.asarray(result)
    elastic_result = elastic_transform(result, 5000, 8, random_state=None)
    cv2.namedWindow('elastic', 0)
    cv2.imshow('elastic', elastic_result)
    cv2.waitKey(0)
    elastic_result = Image.fromarray(elastic_result)
    elastic_result.save('./elastic_result.jpg')
    result = np.array(result)
    uniform_result = np.random.uniform(-50, 50, np.shape(result))
    print(uniform_result)
    print(np.shape(result))
    print(np.shape(uniform_result))
    sum = result + uniform_result
    sum = sum.astype('uint8')
    noised = Image.fromarray(sum)
    noised.save('./noised_result.jpg')

    uniform = (-50. - 50.) * torch.randn(10)
    uniform = torch.FloatTensor(10).uniform_(-50, 50)
    print(uniform)
    """

    model_r = Encoder()
    model_s = Encoder(in_channels=1)
    tensor1 = torch.randn((2,3,256,256))
    tensor2 = torch.randn((2,1,256,256))
    v_r, _ = model_r(tensor1)
    v_s, _ = model_s(tensor2)

    scft_module = SCFT_Moudle()
    scft_module(v_s, v_r)




