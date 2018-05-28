#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:08:57 2018

@author: timothylucas
"""



###
# Load data (copy paste from document)
###

import matplotlib.pyplot as plt 
import numpy as np

def plot_dataset(x, y, legend_loc='lower left'): 
    fig, ax = plt.subplots()
    ax.scatter(x[y==1, 0], x[y==1, 1], c='r', s=100, alpha=0.7, marker='*', label='Sea Bass', linewidth=0)
    ax.scatter(x[y==- 1, 0], x[y==-1, 1], c='b', s=100, alpha=0.7, marker='o', label='Salmon', linewidth=0)
    ax.axhline(y=0, color='k') 
    ax.axvline(x=0, color='k') 
    ax.set_xlabel('Length')
    ax.set_ylabel('Lightness') 
    ax.set_aspect('equal')
    if legend_loc: 
        ax.legend(loc=legend_loc,fancybox=True).get_frame().set_alpha(0.5)
        ax.grid('on')

# First datasets in part 1

x = np.array([[2, 1], [0, -1], [1.5, 0], [0, 1], [-1, 1], [-3, 0], [1, -1], [2, - 1], [3, -2], [3, 1], [-2, 1.5], [-3, 0.5], [-1, 2]]) 
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, - 1, -1]) 
plot_dataset(x, y)
x2 = np.vstack([x, np.array([0, -0.2])])
y2 = np.hstack([y, np.array([-1])]) 
plot_dataset(x2, y2)
x3 = np.array([[4, 1], [-2, 1], [ -2, - 4], [-1, -1], [2, -1], [-1, -3], [3, 2], [1, 2.5], [-3, -1], [-3, 3], [0,-2], [4, -2], [3, -4]])
y3 = np.array([1, 1, 1, -1, - 1, -1, 1, 1, 1, 1, -1, -1, -1])
plot_dataset(x3, y3, legend_loc='lower right')

# For the sigmoid network in part 2 
def sigmoid(inputs):
    return 1.0 / (1.0 + np.exp(-inputs))

def nn_2layer(inputs):
    return np.sign(sigmoid(inputs[:, 0]) + sigmoid(-inputs[:, 1]) - 1.5)

def plot_decision_boundary(network):
    x0v, x1v = np.meshgrid(np.linspace(-2, 8, 20), np.linspace(-8, 2, 20)) 
    x4 = np.hstack([x0v.reshape((-1,1)), x1v.reshape((-1,1))])
    y4 = network(x4)
    plot_dataset(x4, y4, legend_loc=None)

plot_decision_boundary(nn_2layer)

# For the ReLU network in Part 2 
def relu(inputs):
    return np.maximum(0, inputs)

def nn_2layer_relu(inputs):
    return np.sign(relu(-inputs[:, 0]) + relu(inputs[:, 1]) - 0.1)

plot_decision_boundary(nn_2layer_relu)

###
# Helper function to plot the perceptron boundary layers
###

def plot_perceptrons(X, Y, clf, h = 0.02, legend_loc='lower left'):
    # Source: https://stats.stackexchange.com/questions/71335/decision-boundary-plot-for-a-perceptron
    # Plus some extra things to correctly plot the labels etc...
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#    ax.axis('off')
    
    # Plot also the training points
    ax.scatter(X[Y==1, 0], X[Y==1, 1], c='w', s=100, alpha=0.7, marker='*', label='Sea Bass', linewidth=0)
    ax.scatter(X[Y==- 1, 0], X[Y==-1, 1], c='r', s=100, alpha=0.7, marker='o', label='Salmon', linewidth=0)
    
    ax.axhline(y=0, color='k') 
    ax.axvline(x=0, color='k') 
    ax.set_xlabel('Length')
    ax.set_ylabel('Lightness') 
    ax.set_aspect('equal')
    ax.set_title('Perceptron')
    if legend_loc: 
        ax.legend(loc=legend_loc,fancybox=True).get_frame().set_alpha(0.5)
        ax.grid('on')

###
# Individual questions
###

# Now for the training of the perceptrons, inspiration taken from:
# https://stats.stackexchange.com/questions/71335/decision-boundary-plot-for-a-perceptron

from sklearn.linear_model import Perceptron

# we create an instance of SVM and fit our data. We do not scale our
# data since we want to plot the support vectors

###
# Question 1
###

# So we should get a line y = x, which divides the salmons and the sea basses 
# nicely, so for the weights this should mean that since all of the sb's
# are on a positive x, and all the salmons on a negative x, we should just 
# correct for the two points which have x = 0.  So something like w_x = 2, and 
# w_y = -1 should work...

#Check
x_c = x
x_c = x_c*[2,-1]
x_s = [np.sign(x+y) for x,y in x_c]
x_s == y # So this works

# Now let's train the perceptron and see what we get

clf = Perceptron(max_iter=100).fit(x, y)
plot_perceptrons(x, y, clf)

# So the perceptron doesn't actually goes through the origin, but that's not too
# much of a problem here (sorry I'm lazy to see if this is fixable in the 
# algorithm somehow)

# The weights of this perceptron are [3,-2], which is kind of close I guess 
clf.coef_

###
# Question 2
###

# Here we can't use a line through the origin, but we could use one that is 
# just below the origin
# Not sure about the threshold for the perceptron here, since the line through
# the origin is not the same as a perceptron, but I think we should use an 
# offset because of the salmon point slightly below the x axis, so we need
# to correct for that. We could try something like -0.5+[3,-2]

#Check
x_c = x2
x_c = x_c*[3,-2]
x_s = [np.sign(x+y-0.5) for x,y in x_c]
x_s == y2 # So this works

# To actually check it:
clf2 = Perceptron(max_iter=100).fit(x2, y2)
plot_perceptrons(x2, y2, clf2)

# Coefficients, so aparently we do get away with out the offset...
clf2.coef_

###
# Question 3
###

# Getting lazy here, but it seems hard to get the correct weights here since 
# there is a seabass at (-2, -4), so if that one is excluded a line can be found
# and including this will always have some measure of error...

# training the dataset to see what is happening

clf3 = Perceptron(max_iter=100).fit(x3, y3)
plot_perceptrons(x3, y3, clf3)

# Coefficients, so aparently we again do get away without an offset
clf3.coef_

#######
# Part 2
#######

###
# Question 4
###

# So I read on and got the answer :(, but it's a RELU Rectangle

######
# Part 3
######

# Let's train a two layer perceptron network, and let's make it work for the
# third example

from sklearn.neural_network import MLPClassifier

# I don't really understand how the hidden_layer_sizes param works just yet
# but it works... 
clf_ml = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(40,), random_state=1).fit(x3, y3)
plot_perceptrons(x3, y3, clf_ml)


# NEXT STEPS
# Do this tutorial: https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/