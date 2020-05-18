import io 
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

def get_model():
    checkpoint = torch.load(r'whiteclassifier_squeezenet_m2.pth', map_location='cpu')
    
    model_ft = models.squeezenet1_0(pretrained=True)
    for parameter in model_ft.parameters():
        parameter.requires_grad = False
    model_ft.classifier[1]= nn.Conv2d(512,11, kernel_size=(1,1))
    model_ft.num_classes = 11
    model_ft.load_state_dict(checkpoint['state_dict'])
    model_ft.eval()
    return model_ft

def get_tensor(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize([224,224]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    #print('Size:   ',my_transforms(image_bytes).unsqueeze(0).shape)
    #image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image_bytes).unsqueeze(0)
        