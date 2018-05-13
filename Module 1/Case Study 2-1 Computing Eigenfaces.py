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
    
    # for the plotting setup
    fig=plt.figure(figsize=(16, 16))
    l = np.round(np.sqrt(N))
    columns, rows = l, l
    
    for i in range(N):
        im = Image.open(files[i])
        # convert to grayscale
        im = im.convert('LA')
        # make sure it's the right size
        if im.size != (300, 300):
            size = (300, 300)
            im.thumbnail(size, Image.ANTIALIAS)
        
        all_faces[i] = im
        
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        
    
    

if __name__ == '__main__':
    
    # set correct directory
    curr_path = os.getcwd()
    faces_path = curr_path+'/Instructors'
    
    # Now load the faces
    