#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:57:17 2018

@author: timothylucas
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## Preprocessing and visualising the dataset

def loadFaces(path):
    # Number of faces
    os.chdir(path)
    files = [f for f in os.listdir(path) if ('jpg' in f)]
    N = len(files)
    
    # Use a dict to store everything in the end
    all_faces = dict()
    S = []

    for i in range(N):
        im = Image.open(files[i])
        # convert to grayscale
        im = im.convert('L')
        # make sure it's the right size
        if im.size != (300, 300):
            size = (300, 300)
            im.thumbnail(size, Image.ANTIALIAS)
        
        all_faces[i] = im
        
        im_raw = np.asarray(im)
        irow, icol = im_raw.shape
        
        # Reshape and add to the main matrix
        
        temp = np.reshape(im_raw, irow*icol, 1)
        S.append(temp)
    
    return S, all_faces

def normalizeImages(S):
    for i in range(len(S)):
        temp = S[i]
        m = np.mean(temp)
        st = np.std(temp)
        norm = (temp-m)*st/(st+m)
        S[i] = norm
        
    return S
        
def showNormImages(S):
    for i in range(len(S)):
        img_norm = np.reshape(S[i], (300,300))
        # Show the image
        plt.imshow(img_norm.T)
        
def computeAverageFace(S):
    # Convert S to appropriate matrix form
    
    S_f = np.array(S)
    m = np.mean(S_f, axis = 0)
    
    # don't know yet how to do the 'convert to u8bit'
    
    img_avg = np.reshape(m, (300,300))
    plt.imshow(img_avg.T)
    
    return m

def computeEigenFaces()
    
        

if __name__ == '__main__':
    
    # set correct directory
    curr_path = os.getcwd()
    faces_path = curr_path+'/Instructors'
    
    # Now load the faces
    
    S, all_faces = loadFaces(faces_path)