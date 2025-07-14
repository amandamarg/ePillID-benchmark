from imgaug import augmenters as iaa
import torch
import torchvision.transforms.v2 as transforms
import random
from typing import Any, Callable, Literal, Optional, Union, Dict, List
from PIL import Image

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# class AddPerChannel(transforms.Transform):
#     def __init__(self):
#         super().__init__()

#     def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
#         apply_transform = (torch.rand(size=(1,)) < self.p).item()
#         params = dict(apply_transform=apply_transform)
#         return params

#     def transform(self, inpt: Any, params: Dict[str, Any]):
#         if not params["apply_transform"]:
#             print("Not transforming anything!")
#             return inpt
#         else:
#             return super().transform(inpt, params)

# def add_per_channel(image, transform):
#     ch_r, ch_b, ch_g = transform(image[])

# def randomize_parameters(params_to_randomize, params={}):
#     for (key, value) in (params_to_randomize.items()):
#         func, a, b = value
#         params[key] = func(a,b)
#     return params



def get_imgaug_sequences(low_gblur = 1.0, 
high_gblur = 3.0, addgn_base_ref = 0.01, 
addgn_base_cons = 0.001, rot_angle = 180, 
max_scale = 1.0, add_perspective = False
):
    affine_seq = iaa.Sequential([
            iaa.Affine(
                rotate=(-rot_angle, rot_angle),
                scale=(0.8, max_scale),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            ),
            sometimes(iaa.Affine(
                shear=(-4, 4),
            ))
        ])


    affine_list = [affine_seq]

    contrast_list = [
            iaa.Sequential([
                iaa.LinearContrast((0.7, 1.0), per_channel=False), # change contrast
                iaa.Add((-30, 30), per_channel=False), # change brightness
            ]),
            iaa.Sequential([
                iaa.LinearContrast((0.4, 1.0), per_channel=False), # change contrast
                iaa.Add((-80, 80), per_channel=False), # change brightness
            ])            
        ]

    if add_perspective:
        print("Adding perspective transform to augmentation")
        affine_list =  affine_list + [
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ]

        contrast_list = contrast_list + [ 
            iaa.GammaContrast((0.5, 1.7), per_channel=True),
            iaa.SigmoidContrast(gain=(8, 12), cutoff=(0.2,0.8), per_channel=False)
             ]
        

    ref_seq = iaa.Sequential(affine_list + [
        iaa.OneOf(contrast_list),
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 3*addgn_base_ref*255), per_channel=0.5),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, addgn_base_ref*255), per_channel=0.5),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, high_gblur)),
            iaa.GaussianBlur(sigma=(0, low_gblur)),
        ])
    ])


    cons_seq = iaa.Sequential(affine_list + [
        iaa.LinearContrast((0.9, 1.1), per_channel=False),
        iaa.Add((-10, 10), per_channel=False),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 5*addgn_base_cons*255), per_channel=0.5),
        iaa.GaussianBlur(sigma=(0, low_gblur)),
    ])
    # cons_seq = transforms.Compose(affine_list + [
    #     transforms.ColorJitter(brightness=(.9, 1.1), contrast=(.9, 1.1)),
    #     # transforms.GaussianNoise(sigma=),
    #     transforms.GaussianBlur(kernel_size=5, sigma=(0, low_gblur))
    # ])
    
    return affine_seq, ref_seq, cons_seq

def split_channels(img):
    if isinstance(img, Image.Image):
        img = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(img)
    r,g,b = torch.unbind(img, dim=-3)
    return torch.unsqueeze(r,-3), torch.unsqueeze(g,-3), torch.unsqueeze(b,-3)

def transform_channels(tr, img):
    r,g,b = split_channels(img)
    return torch.cat([tr(r), tr(g), tr(b)], dim=-3)

transform_half_per_channel = lambda aug: transforms.Lambda(lambda x: random.choice([aug(x), transform_channels(aug,x)]))

'''
SigmoidContrast does not accept PIL objects
'''
class SigmoidContrast(transforms.Transform):
    def __init__(self, gain, cutoff):
        super().__init__()
        self.gain = gain
        self.cutoff = cutoff
    
    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if isinstance(self.gain, tuple):
            min_gain, max_gain = self.gain
            gain = random.uniform(float(min_gain), float(max_gain))
        else:
            gain = float(self.gain)
    
        if isinstance(self.cutoff, tuple):
            min_cutoff, max_cutoff = self.cutoff
            cutoff = random.uniform(float(min_cutoff), float(max_cutoff))
        else:
            cutoff = float(self.cutoff)
        params = dict(gain=gain, cutoff=cutoff)
        return params
            
    def transform(self, inpt: Any, params: Dict[str, Any]):
        gain = params["gain"]
        cutoff = params["cutoff"]
        if inpt.dtype == torch.int8:
            inpt = transforms.ToDtype(torch.float32, scale=True)(inpt)
            return 255*torch.nn.Sigmoid()(gain*(inpt - cutoff))
        return torch.nn.Sigmoid()(gain*(inpt - cutoff))

def get_imgaug_sequences(low_gblur = 1.0, 
high_gblur = 3.0, addgn_base_ref = 0.01, 
addgn_base_cons = 0.001, rot_angle = 180, 
max_scale = 1.0, add_perspective = False
):
    affine_seq = transforms.RandomChoice([transforms.RandomAffine(degrees=(-rot_angle, rot_angle), scale=(0.8, max_scale), translate=(.05, .05), shear=(-4,4)), transforms.RandomAffine(degrees=(-rot_angle, rot_angle), scale=(0.8, max_scale), translate=(.05, .05))])
    affine_list = [affine_seq]
    contrast_list = [transforms.ColorJitter(brightness=(.75, 1.25), contrast=(0.7, 1.0)), transforms.ColorJitter(brightness=(.4, 1.6),contrast=(0.4, 1.0))]

    if add_perspective:
        print("Adding perspective transform to augmentation")
        affine_list.append(transforms.RandomPerspective(distortion_scale=(0.01, 0.1)))
        gamma_contrast = transforms.Lambda(lambda x: transforms.functional.adjust_gamma(x, random.uniform(.5, 1.7)))
        contrast_list.extend([transform_half_per_channel(gamma_contrast), SigmoidContrast(gain=(8,12), cutoff=(0.2,0.8))])

    gauss_noise = lambda a,b: transform_half_per_channel(transforms.GaussianNoise(sigma=random.uniform(a,b)))


    ref_seq = transforms.Compose(affine_list + [transforms.RandomChoice(contrast_list),    
                                                transforms.RandomChoice([gauss_noise(0.0, 3*addgn_base_ref), gauss_noise(0.0, addgn_base_ref)]),
                                                transforms.RandomChoice([transforms.GaussianBlur(kernel_size=5, sigma=(0.001,high_gblur)), transforms.GaussianBlur(kernel_size=5, sigma=(0.001,low_gblur))])
                                ])

    cons_seq = transforms.Compose(affine_list + [
        transforms.ColorJitter(brightness=(.9, 1.1), contrast=(.9, 1.1)),
        gauss_noise(0.0, 5*addgn_base_cons),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.001, low_gblur))
    ])
    return affine_seq, ref_seq, cons_seq
