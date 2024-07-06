import numbers
import random
from torchvision.transforms import functional as F 
import numpy as np
import torch
import math
from imresize.py import random_resize_train, fix_resize_test
from torch.nn.functional import interpolate
import pdb
import copy

class compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        
        for t in self.transforms:
           img = t(img) 
        
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class compose_two(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, rgb):
        
        for t in self.transforms:
           img = t(img)
           rgb = t(rgb)       
        return img, rgb

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string       
        
class toTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
        
    @staticmethod
    def to_tensor(img):
    	
        return torch.from_numpy(img.transpose((2, 0, 1)))                 
        	
    def __call__(self, img):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """

        if isinstance(img, torch.FloatTensor):           
            return img 
        else:       	
            return self.to_tensor(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class toTensor_two(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
        
    @staticmethod
    def to_tensor(img):
    	
        return torch.from_numpy(img.transpose((2, 0, 1)))                 
        	
    def __call__(self, img):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """

        if isinstance(img, torch.FloatTensor):           
            return img 
        else:       	
            return self.to_tensor(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std
              
        
    @staticmethod
    def normal(img, mean, std):
    	        
        for t, m, s in zip(img, mean, std):
            t.sub_(m).div_(s)
            
        return img
                
    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        return self.normal(img, self.img_mean, self.img_std) 


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
                        
class randomCrop(object):
    """Crop the given  Image at a random location."""
    

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        w, h, _ = img.shape
        tw,th = output_size
        if w == tw and h == th:
            return 0, 0, w, h

        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)


        return i, j, tw, th
    
    @staticmethod
    def crop(img, i, j, w, h):
           
        img = img[i: i+w, j:j+h, :]
       
        return img

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        i, j, h, w = self.get_params(img, self.size)


        return self.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class randomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    @staticmethod
    def hflip(img):
#        print(img.shape)
        source = np.array(img, dtype=np.float32)
        for i in range(img.shape[1]):
            source[:,i,:] = img[:,img.shape[1]-i-1,:]   

        return source
        	
    def __call__(self, img):
        """
        Args:
            Image to be flipped.

        Returns:
            Randomly flipped image.
        """
        if random.random() < self.p:
            return self.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class randomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    @staticmethod
    def vflip(img):
#        print(img.shape)
        source = np.array(img, dtype=np.float32)
        for i in range(img.shape[1]):
            source[i,:,:] = img[img.shape[1]-i-1,:,:]   

        return source
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return self.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class randomRoll(object):
    """Roll the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being rolled. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p
        	
    def __call__(self, img):
        """
        Args:
            Image to be flipped.

        Returns:
            Randomly flipped image.
        """
        if random.random() < self.p:
            img = copy.deepcopy(np.roll(img, int(img.shape[0] / 2), 0))      	
        	
            return img
        
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
                
class randomRotation(object):
    """Rotate the image by angle.
    """

    def __init__(self, degrees, p = 0.5):

        self.degrees = degrees
        self.p = p
        
    def __call__(self, img):

        if random.random() < self.p:
        	
            if self.degrees == '90':        	
                img = copy.deepcopy(np.rot90(img, 1, [0, 1]))
            elif self.degrees == '-90': 
                img = copy.deepcopy(np.rot90(img, -1, [0, 1]))            
            
        return img 
       
class Resize_train(object):

    def __init__(self, upscale_factor, interpolation):
        self.sacle = upscale_factor
        self.interpolation = interpolation
  
    @staticmethod
    def resize(img, scale, interpolation, typeKernel= 'cubic'):

        if interpolation == 'Bicubic':  
            img = img.unsqueeze(0)
            resize_img = interpolate(img, (int(img.shape[2]/scale), int(img.shape[3]/scale)), mode='bicubic',  align_corners=False)
            resize_img = resize_img.squeeze(0)
        
        else:

             resize_img = random_resize_train(img, scale)  

        return resize_img
                
    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """            

        return self.resize(img, self.sacle, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '()'  
        
        
class Resize_test(object):

    def __init__(self, upscale_factor, interpolation):
        self.sacle = upscale_factor
        self.interpolation = interpolation
  
    @staticmethod
    def resize(img, scale, interpolation, typeKernel= 'cubic'):

        if interpolation == 'Bicubic':  
            img = img.unsqueeze(0)
            resize_img = interpolate(img, (int(img.shape[2]/scale), int(img.shape[3]/scale)), mode='bicubic',  align_corners=False)
            resize_img = resize_img.squeeze(0)
        
        else:

             resize_img = fix_resize_test(img, scale)  

        return resize_img
                
    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """            

        return self.resize(img, self.sacle, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '()'                           
        