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
import copy

## Preprocessing and visualising the dataset

def convertImageToU8bit(input_img):
    # assumes input array that can be represented 
    # as an image, needs to be scaled to a [0, 255]
    # range, which is what this function returns
    input_img -= np.min(input_img)
    input_img *= (255.0/input_img.max())
    return input_img

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
    img_avg = convertImageToU8bit(img_avg)
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
        img = convertImageToU8bit(img)
        # convert and save
        im = Image.fromarray(img)
        im.convert('RGB').save('eigen_{}.jpeg'.format(i+1))
        if show_images:
            im.show()
        
        EigenFaces.append(img)
    
    return u, EigenFaces

def classifyNewFaces(S, u, input_image = '1.jpg', show_images = True):
    # Find the weight of each face for each image in the training set.
    # omega will store this information for the training set.
    omega = []
    dbx = np.array(S)
    
    for h in range(len(dbx)):
        WW = []
        for i in range(len(u)):
            t = u[i].T
            WeightOfImage = np.dot(t, dbx.T[:,h]).T
            WW.append(WeightOfImage)
        omega.append(WW)
        
    im = Image.open(input_image)
    InputImage = im.convert('L')
    if show_images:
        InputImage.show()
    
    im_raw = np.asarray(InputImage)
    InImage = np.reshape(im_raw.T,im_raw.shape[0]*im_raw.shape[1],1)
    temp = InImage
    me=np.mean(temp)
    st=np.std(temp)
    temp=(temp-me)*st/(st+me)
    Difference = temp
    NormImage = temp
    
    p = []
    aa = len(u)
    for i in range(aa):
        pare = np.dot(NormImage, u[i])
        p.append(pare)
        
    #m is the mean image, u is the eigenvector
    ReshapedImage = me + np.matmul(np.array(u).T, p)
    ReshapedImage = np.reshape(ReshapedImage,im_raw.shape)
    ReshapedImage = ReshapedImage.T
    # Show the reconstructed image.
    if show_images:
        ReshapedImage_s = convertImageToU8bit(ReshapedImage)
        ReshapedImage_s = Image.fromarray(ReshapedImage_s)
        ReshapedImage_s.show()
    # Compute the weights of the eigenfaces in the new image
    InImWeight = [];
    for i in range(len(u)):
        t = u[i]
        WeightOfInputImage = np.dot(t,Difference.T)
        InImWeight.append(WeightOfInputImage)
    
    # Find distance
    e=[]
    for i in range(len(omega)):
        q = omega[i]
        DiffWeight = InImWeight-q
        mag = np.linalg.norm(DiffWeight)
        e.append(mag)

    kk = list(range(len(e)))
    
#    ll = 1:M;
#    stem(ll,InImWeight)
    
    return None
        

if __name__ == '__main__':
    
    # set correct directory
    curr_path = os.getcwd()
    faces_path = curr_path+'/Instructors'
    
    # Now load the faces
    
    S, all_faces = loadFaces(faces_path)
    # Normalize
    S = normalizeImages(S)
    # Compute average face
    avg_face = computeAverageFace(S)
    # Compute eigen faces
    u, eigen_faces = computeEigenFaces(S, show_images = False)
    # Now compute the reconstructed face
    classifyNewFaces(S, u, input_image = '1.jpg', show_images = True)
    
    