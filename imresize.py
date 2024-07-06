import numpy as np
import torch
from scipy.ndimage import filters, measurements, interpolation
import pdb
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import random

   
def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)
    
def downscale(im, kernel):

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    for channel in range(im.shape[0]):
        out_im[channel, :, :] = filters.correlate(im[channel, :, :], kernel)

    # Then subsample and return
    return torch.from_numpy(out_im)  #.permute(2,0,1)
    
def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)

    
def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
    
                     	       
def anisotropic_gaussian_kernel(k_size, scale_factor, lambda_1, lambda_2, theta):
        
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]
    
    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2  + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]
    
    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]
    
    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) #* (1 + noise)
    
    # shift the kernel so it will be centered
#    raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)
    
    # Normalize the kernel and return
    kernel = raw_kernel / np.sum(raw_kernel)
    
    return kernel 

def isotropic_gaussian_kernel(k_size, scale_factor, sigma):
        
    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
        
    raw_kernel = np.exp(-(X ** 2 + Y ** 2) / (2. * sigma ** 2)) #* (1 + noise)
       
    # shift the kernel so it will be centered
#    raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)
    
    # Normalize the kernel and return
    kernel = raw_kernel / np.sum(raw_kernel)
    
    return kernel
        
def random_anisotropic_gaussian_kernel(k_size, scale_factor, min_var, max_var):
	
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi
    
    return anisotropic_gaussian_kernel(k_size, scale_factor, lambda_1, lambda_2, theta)	
    
def random_isotropic_gaussian_kernel(k_size, scale_factor, min_sigma, max_sigma):
    sigma = min_sigma + np.random.rand() * (max_sigma - min_sigma)
    
    return isotropic_gaussian_kernel(k_size, scale_factor, sigma)	    
    

def random_gaussian_noise(img_lr, sigma=5):

    if np.random.random() > 0.8:
       return img_lr
     
    noise_std = np.random.randint(1, sigma) 
    gaussian_noise = np.random.randn(*img_lr.shape)*noise_std / 255   
    img_lr = img_lr + gaussian_noise   

    img_lr[img_lr<0] = 0
    img_lr[img_lr>1] = 1                        
    
    return img_lr


def fix_gaussian_noise(img_lr, sigma=9):
    
    gaussian_noise = np.random.randn(*img_lr.shape)*sigma / 255  
    img_lr = img_lr + gaussian_noise
    img_lr[img_lr<0] = 0
    img_lr[img_lr>1] = 1     
                    
    return img_lr      
        
        
        
        	        	                       
def random_resize_train(img, scale_factor):

    scale = np.array([scale_factor, scale_factor])  # choose scale-factor
    
    sizeName = [7, 9, 13, 15, 21]        
    k_size = sizeName[np.random.randint(0,5,1)[0]]  
    sizeType = np.array([k_size, k_size])               

    kernelName = ['isotropic', 'anisotropic']        
    kernelType = kernelName[np.random.randint(0,2,1)[0]]

    
    if kernelType == 'anisotropic':

        min_var = 0.2  # variance of the gaussian kernel will be sampled between min_var and max_var
        max_var = scale_factor 
        kernel = random_anisotropic_gaussian_kernel(sizeType, scale, min_var, max_var)    
 
    elif kernelType == 'isotropic':
       
        min_sigma = 0.2
        max_sigma = scale_factor
        kernel = random_isotropic_gaussian_kernel(sizeType, scale, min_sigma, max_sigma)

    downsName = ['bicubic', 'bilinear', 'direction']        
    downsType = downsName[np.random.randint(0,3,1)[0]]
        
    
    # Choose interpolation method, each method has the matching kernel size
    
    if downsType == 'direction':

        kernel = torch.from_numpy(kernel).type_as(img).unsqueeze(0).unsqueeze(0)
#        kernel = torch.repeat_interleave(kernel, img.shape[0], dim=1)

        img = img.unsqueeze(0)
  
        if kernel.shape[2] % 2 == 1:
            pad = int((kernel.shape[2] - 1) / 2.)
        else:
            pad = int((kernel.shape[2] - scale_factor) / 2.)            

        padding = nn.ReplicationPad2d(pad)               
        img = padding(img)
        
        img = img.permute(1,0,2,3)
        downs_img = F.conv2d(img, kernel, stride=scale_factor)    
        downs_img = downs_img.squeeze(1)       
         	
    else:
        blur_img = downscale(img, kernel)    
        downs_img = F.interpolate(blur_img.unsqueeze(0), scale_factor=1/scale_factor, mode=downsType, align_corners=False).type(torch.FloatTensor)    	
        downs_img = downs_img.squeeze(0) 
     

    downs_img = random_gaussian_noise(downs_img)
    
    return downs_img.type(torch.FloatTensor) 

def fix_resize_test(img, scale_factor):

    k_size = 13 
      
    kernel = anisotropic_gaussian_kernel(np.array([k_size, k_size]), np.array([scale_factor, scale_factor]), random.uniform(2,5), random.uniform(8,12), random.random()*np.pi)
    blur_img = downscale(img, kernel)  	
    downs_img = F.interpolate(blur_img.unsqueeze(0), scale_factor=1/scale_factor, mode='bicubic', align_corners=False).type(torch.FloatTensor) 
    downs_img = downs_img.squeeze(0) 
      
    downs_img = fix_gaussian_noise(downs_img, sigma=3)
    
    return downs_img.type(torch.FloatTensor) 