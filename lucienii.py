import os
import cv2
import numpy as np
import SimpleITK as sitk

def files_indir(suffix='', path='.', abspath=True, deep=False, hidden=False, sort_level=-1):
    """find files of certain type in specified directory and all subdirectories

    suffix='': suffix of files for searching (dafault all types)
    path='': root to start searching (default current directory)
    abspath=True: show absolute path (default True)
    deep=False: search in subdirectories (dafault False)
    hidden=False: show hidden files (default False)
    sort_level: sorting condition (-1 -> file name, -2 -> the last directory, -3 -> upper directory,....)
    """
    all_files = []
    abs_files = os.listdir(path)
    for i in range(len(abs_files)):
        abs_files[i] = os.path.abspath(path)+'/'+abs_files[i]
    if hidden == True:
        for name in abs_files:
            if os.path.isfile(name):
                if name.endswith(suffix):
                    all_files.append(name)
            elif os.path.isdir(name) and deep == True:
                all_files = all_files + (files_indir(suffix=suffix, path=name, abspath=abspath, deep=deep, hidden=hidden, sort_level=sort_level))
    else:
        for name in abs_files:
            if os.path.isfile(name):
                if name.endswith(suffix) and not name.startswith('.'):
                    all_files.append(name)
            elif os.path.isdir(name) and deep == True:
                all_files = all_files + (files_indir(suffix=suffix, path=name, abspath=abspath, deep=deep, hidden=hidden, sort_level=sort_level))
    all_files.sort(key= lambda x:x.split('/')[sort_level])
    if abspath == False:
        for i in range(len(all_files)):
            all_files[i] = all_files[i].split('/')[-1]
    return all_files


def rescale_slices_cxy(slices, shape):
    """resize all slices of an array nii file, channels first"""
    resized = np.empty((slices.shape[0], shape[0], shape[1]))
    for i in range(slices.shape[0]):
        resized[i] = cv2.resize(slices[i], shape)
    return resized

def rescale_slices_xyc(slices, shape):
    """resize all slices of an array nii file, channels last"""
    resized = np.empty((shape[0], shape[1], slices.shape[2]))
    for i in range(slices.shape[2]):
        resized[:,:,i] = cv2.resize(slices[:,:,i], shape)
    return resized

def pad(img, shape):
    """pad an image to transorm it into a specified shape, does not cut the image 
    if output size is smaller"""
    delta_h = shape[0]-img.shape[0]
    delta_w = shape[1]-img.shape[1]
    if delta_h > 0:
        up = delta_h//2
        down = delta_h-up
        img = np.vstack((np.zeros((up,img.shape[1])), img))
        img = np.vstack((img, np.zeros((down,img.shape[1]))))
    if delta_w > 0:
        left = delta_w//2
        right = delta_w-left
        img = np.hstack((np.zeros((img.shape[0],left)), img))
        img = np.hstack((img, np.zeros((img.shape[0], right))))
    return img

def crop(img, shape):
    """pad an image to transorm it into a specified shape, does not cut the image 
    if output size is smaller"""
    delta_h = img.shape[0]-shape[0]
    delta_w = img.shape[1]-shape[1]
    if delta_h > 0:
        up = delta_h//2
        down = delta_h-up
        img = img[up:-down,:]
    if delta_w > 0:
        left = delta_w//2
        right = delta_w-left
        img = img[:,left:-right]
    return img

def pad_crop(img, shape):
    """aplly padding and cropping to resize the current image without rescaling"""
    img = pad(img, shape)
    img = crop(img, shape)
    return img

def resize_slices_cxy(slices, shape):
    resized = np.empty((slices.shape[0], shape[0], shape[1]))
    for i in range(slices.shape[0]):
        resized[i] = pad_crop(slices[i], shape)
    return resized

def resize_slices_xyc(slices, shape):
    resized = np.empty((shape[0], shape[1], slices.shape[2]))
    for i in range(slices.shape[2]):
        resized[:,:,i] = pad_crop(slices[:,:,i], shape)
    return resized
