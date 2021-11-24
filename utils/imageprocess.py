import numpy as np
import cv2
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
#from utils import augmentations
import IPython

class RotateTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        if isinstance(self.angles, list):
            angle = random.choice(self.angles)
        else:
            angle = self.angles
        return TF.rotate(x, angle)

    

def image_transformer(input_image=None, train=True):
    """
    Using torchvision.transforms, make PIL image to tensor image
    with normalizing and flipping augmentations
    """
    
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    if train:
        transformer = transforms.Compose([        
            RotateTransform([0, 0, 0, -90, 90, 180]),
            #transforms.RandomAffine(degrees=0, shear=(0, 10, 0, 10)),
            #transforms.RandomRotation(degrees=[-30, 30], expand=True),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            #transforms.Resize([256, 256]),
            #transforms.RandomHorizontalFlip(p=0.3),
            #transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    #transformed_image = transforms.functional.rotate(transformer(input_image), angle)
    transformed_image = transformer(input_image)
    
    return transformed_image



def tta_transformer(input_image, angle):
    """
    Test Time Augmentation for creating final test labels.
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transformer = transforms.Compose([        
        RotateTransform(angle),
        #transforms.RandomPerspective(distortion_scale=0.25, p=1.0),
#         transforms.RandomHorizontalFlip(p=0.3),
#         transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        normalize,
    ])

    transformed_image = transformer(input_image)
    
    return transformed_image



def image_processor(input_image):
    """
    This function is to process given image using opencv.
    """
    blur = cv2.GaussianBlur(input_image, (3, 3), 0)

    denoised =  cv2.fastNlMeansDenoising(blur, None, 15, 7, 49)
    denoised = cv2.fastNlMeansDenoising(denoised, None, 15, 7, 21)
    denoised = cv2.fastNlMeansDenoising(denoised, None, 15, 7, 21)
    
    kernel_sharpen = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) 
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
    
    #_, thr = cv2.threshold(sharpened, 150, 255, 0)
    img_clip = np.where(sharpened < 150, 0, sharpened)
    
    return img_clip


#     if train:
#         transformer = transforms.Compose([
#             transforms.RandomRotation(30),
#             transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
#             transforms.RandomHorizontalFlip(p=0.3),
#             transforms.RandomVerticalFlip(p=0.3),
#             transforms.ToTensor(),
#             normalize,
#         ])



# for index 10
#         transformer = transforms.Compose([
            
#             transforms.RandomApply([transforms.RandomRotation([-30, 30])], p=0.3), #add(02.02)
#             transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.3)], p=0.3), #add(02.02)
#             transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(0, 10, 0, 10))], p=0.1),
#             transforms.ToTensor(),
#             normalize,
#         ])