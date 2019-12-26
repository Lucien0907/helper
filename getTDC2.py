from lupy import *
from lufil import *
import os
import time
import sys
import math
import shutil
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


# find files
files = fsearch(path = '../data/ISLES2018/TRAINING/',
                suffix='.nii',
                include='4DPWI',
                sort_level=-3)
print('\n'.join(files))
print(len(files),"files found!")

# select certain files
if len(sys.argv) == 2:
    if not sys.argv[1] == "all":
        case = int(sys.argv[1])
        for x in files:
            if x.find('case_'+str(case)) >= 0:
                files = [x]
                print('selected: '+files[0].split('/')[-3])
                break
            else:
                files=[]
                print('case does not exist, please enter a valid case number')

for fin in files:
    if not os.path.exists('../av/aif_vof_'+fin.split('/')[-3]+'.png'):
        print('\n',fin.split('/')[-3])
        img = sitk.ReadImage(fin)
        arr = sitk.GetArrayFromImage(img)
        print('shape: ', arr.shape)
        print('spacing: ', img.GetSpacing())
        print('origin: ', img.GetOrigin())
        print('direction: ', img.GetDirection())
        print("original: max="+str(np.max(arr))+", min="+str(np.min(arr))+", dtype="+str(arr.dtype))

        # normalization
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        print("normalized: max="+str(np.max(arr))+", min="+str(np.min(arr))+", dtype="+str(arr.dtype))

        # specify the location and size of the window to search for aif and vof
        xroi = arr.shape[2]//2
        yroi = arr.shape[3]//2
        w = 200
        h = 200

        # create roi segments from original data
        #arr_roi = arr[:,:,xroi-h//2:xroi+h//2, yroi-w//2:yroi+w//2]
        arr_roi = arr
        print('arr_roi shape:', arr_roi.shape)

        # segments to store max and max-min functions
        arr_max = np.amax(arr_roi, axis=0)
        arr_diff = np.amax(arr_roi, axis=0) - np.amin(arr_roi, axis=0)
        print('arr_max shape:', arr_max.shape, 'arr_diff shape:', arr_diff.shape)

        n_slice = arr_roi.shape[1]
        switcher = {
            1: 2,
            2: 3,
            3: 4,
            4: 4,
            5: 4,
            6: 4,
            7: 4,
            8: 4,
        }

        step = switcher.get(n_slice, 5)
        shape1 = range(n_slice)
        shape2 = range(0, arr_roi.shape[2], step)
        shape3 = range(0, arr_roi.shape[3], step)
        total = len(shape1) * len(shape2) * len(shape3)
        c = 0
        q = 0

        ttp = np.empty((0,5), dtype=np.int16)
        imax = 0
        imin = len(ttp) - 1
        for z in shape1:
            for i in shape2:
                for j in shape3:
                    c += 1
                    loc = (z,i,j)
                    print('\n'+fin.split('/')[-3]+': '+str(n_slice)+' slices ')
                    print(str(c)+'/'+str(total)+' '+str(loc))

                    sig = arr_roi[:,z,i,j]
                    sig = sig - np.min(sig)
                    if np.max(sig) == 0:
                        q += 1
                        print('blank',q)
                    else:
                        plt.subplot(141)
                        plt.plot(sig)

                        # MedianAerageFitering
                        sig = MyMedianAverage(sig, 5)
                        sig = MyMedianAverage(sig, 5)
                        plt.subplot(142)
                        plt.plot(sig)

                        sig = sig - np.min(sig)
                        idx_max = np.argmax(sig)
                        if 10 <= idx_max < 30:
                            idx1 = idx_max - 7
                            idx2 = idx_max + 7
                            ot1 = sig[idx_max-10:idx_max]
                            ot2 = sig[idx_max:idx_max+10]
                            ot = np.hstack((sig[0:idx1],sig[idx2:-1]))
                            bo = sig[idx1:idx2]
                            if np.sum(ot1) < np.sum(bo):
                                if np.sum(ot2) <np.sum(bo):
                                    if np.sum(bo) > np.sum(sig)*0.5:
                                        if idx_max < imin:
                                            imin = idx_max
                                            aif = sig
                                        if idx_max > imax:
                                            imax = idx_max
                                            vof = sig
                                        ttp[idx_max] += 1
                                        plt.subplot(143)
                                        plt.plot(sig)

        plt.subplot(144)
        plt.plot(ttp)
        plt.savefig('../av/aif_vof_'+fin.split('/')[-3]+'.png')
        plt.cla()
        plt.clf()
