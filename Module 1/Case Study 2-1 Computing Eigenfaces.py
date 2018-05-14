#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:57:17 2018

@author: timothylucas
"""

## Disclaimer: this script is pretty much a direct port of the M file
## and so quite unpythonic, I'll work on that hopefully later

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

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
    # Scale the image
    img_avg -= np.min(img_avg)
    img_avg *= (255.0/img_avg.max())
    # plot 
    im = Image.fromarray(img_avg.T)
    im.convert('RGB')
    im.show()
    
    return m
    
def computeEigenFaces(S, show_images = False):
    dbx = np.array(S)
    A = dbx.T # Note that the dbx here is transposed as compared to the 
              # original M script
    
    A_t = copy.deepcopy(A)
    A_t = A_t.T
    
    L = np.matmul(A_t, A) #Is this the correct order?
    
    dd, vv = np.linalg.eig(L) #Order here reversed from MATLAB
    # dd : eigenvalues
    # vv : eigenvectors
    # from the docs:
    # The normalized (unit “length”) eigenvectors, such that the column 
    # v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    
    # Sort and eliminate those whose eigenvalue is zero
    
    nonzero_eigenvals = dd > 1e-4
    v = vv[:,nonzero_eigenvals]
    d = dd[nonzero_eigenvals]
    
    # Sort
    # eigenvalues already sorted, but in descending order, so let's
    # change this into ascending
    
    d = d[::-1]
    v = v[:,::-1]
    
    # Normalization also does not need to happen anymore, already done
    # by numpy
    
    # Now eigenvectors of the C matrix
    u = []
    for i in range(len(v)):
        temp = np.sqrt(d[i])
        u.append(np.matmul(dbx.T, v[:,i])/temp)
    
    # are the eigenvectors transposed again?    
    
    # Normalization of the C matrix, looks like it's already normalized 
    # but just for the hell of it
    for i in range(len(u)):
        kk = u[i]
        temp = np.sqrt(np.sum(np.square(kk)))
        u[i] = u[i]/temp
    
    # Now render the eigenfaces
    EigenFaces = []
    
    for i in range(len(u)):
        img = np.reshape(u[i], (300,300))
        img = img.T
        # Scale the image
        img -= np.min(img)
        img *= (255.0/img.max())
        # convert and save
        im = Image.fromarray(img)
        im.convert('RGB').save('eigen_{}.jpeg'.format(i+1))
        if show_images:
            im.show()
        
        EigenFaces.append(img)
    
    return EigenFaces

def classifyNewFaces():
    
    return None 

if __name__ == '__main__':
    
    # set correct directory
    curr_path = os.getcwd()
    faces_path = curr_path+'/Instructors'
    
    # Now load the faces
    
    S, all_faces = loadFaces(faces_path)

    S = normalizeImages(S)
    avg_face = computeAverageFace(S)
    eigen_faces = computeEigenFaces(S, show_images = False)