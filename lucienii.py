import os
import cv2
import numpy as np
import SimpleITK as sitk
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
import matplotlib.pyplot as plt

def files_indir(suffix='', path='.', include='', abspath=True, deep=False, hidden=False, sort_level=-1):
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
                if name.endswith(suffix) and name.split('/')[-1].find(include)>=0:
                    all_files.append(name)
            elif os.path.isdir(name) and deep == True:
                all_files = all_files + (files_indir(suffix=suffix, path=name, include=include, abspath=abspath, deep=deep, hidden=hidden, sort_level=sort_level))
    else:
        for name in abs_files:
            if os.path.isfile(name):
                if name.endswith(suffix) and name.split('/')[-1].find(include)>=0 and not name.startswith('.'):
                    all_files.append(name)
            elif os.path.isdir(name) and deep == True:
                all_files = all_files + (files_indir(suffix=suffix, path=name, include=include, abspath=abspath, deep=deep, hidden=hidden, sort_level=sort_level))
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

def nii_to_png(nii_file, dst_path=None, name_idx=-2):
    itkimg = sitk.ReadImage(nii_file)
    img = sitk.GetArrayFromImage(itkimg)
    if dst_path == None:
        dst_path = nii_file.rstrip(nii_file.split('/')[-1]).rstrip('/')
    for i in range(len(img)):
        plt.imsave(dst_path+'/slice'+str(i)+'.png',img[i], cmap='gray')
    print("File #"+nii_file.split('/')[-2]+" saved "+str(i)+" slices")
    return dst_path

def niis_to_png(nii_files,dst_path=None):
    print("start converting "+str(len(nii_files))+" nii files")
    if dst_path!=None:
        dst = dst_path
    for x in nii_files:
        nii_to_png(x, dst_path=dst_path)
    print(str(len(nii_files))+" files converted")
    return 

def correct_bias(in_file, out_file=None, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """
    if out_file == None:
        out_file = in_file.rstrip('.nii')+"_bias_corrected.nii"
    #使用N4BiasFieldCorrection校正MRI图像的偏置场
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)

def correct_bias_itk(in_file, out_file=None, image_type=sitk.sitkFloat64):
    if out_file == None:
        out_file = in_file.rstrip('.nii')+"_bias_corrected.nii"
    inputImage = sitk.ReadImage(in_file)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    maskImage = sitk.OtsuThreshold( inputImage, 0, 1, 200 )
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    output = corrector.Execute(inputImage, maskImage)
    sitk.WriteImage(output, out_file)
    return out_file

def change_name(files,a="",b=""):
    for i in range(len(files)):
        tmp = files[i]
        if tmp.find(a) >=0:
            print(tmp)
            print(a+" found in path, changing it to "+b)
            tmp=tmp.replace(a, b)
            os.rename(files[i], tmp)
            files[i] = tmp
            print("new name: "+tmp)
    return files

def normalization(x):
    mean = np.mean(x)
    std = np.std(x)
    out = (x-mean)/std
    print("Single normalization finished: mean="+str(mean)+", std="+str(std))
    return out

def nii_to_array(nii_path):
    itkimage = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(itkimage)
    print("data type: "+str(np.dtype(img[0][0][0])))
    return img

def save_as_nii(img, path):
    itkimage = sitk.GetImageFromArray(img)
    sitk.WriteImage(itkimage, path)
    print("Images saved as nii file.")
    return path
