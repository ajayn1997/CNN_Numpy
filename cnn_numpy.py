# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:02:39 2018

@author: theaj
"""

from im2col import *
import numpy as np

class ConvolutionLayer():

    def __init__(self, X_dim, num_filter, height_filter, width_filter, stride, padding):

        self.d_X, self.h_X, self.w_X = X_dim

        self.num_filter, self.height_filter, self.width_filter = num_filter, height_filter, width_filter
        self.stride, self.padding = stride, padding

        self.W = np.random.randn(
            num_filter, self.d_X, height_filter, width_filter) / np.sqrt(num_filter / 2.)
        self.b = np.zeros((self.num_filter, 1))
        self.params = [self.W, self.b]

        self.h_out = (self.h_X - height_filter + 2 * padding) / stride + 1
        self.w_out = (self.w_X - width_filter + 2 * padding) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.num_filter, self.h_out, self.w_out)

    def forward(self, X):

        self.n_X = X.shape[0]

        self.X_col = im2col_indices(
            X, self.height_filter, self.width_filter, stride=self.stride, padding=self.padding)
        W_row = self.W.reshape(self.num_filter, -1)

        out = W_row @ self.X_col + self.b
        out = out.reshape(self.num_filter, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):

        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.num_filter, -1)

        dW = dout_flat @ self.X_col.T
        dW = dW.reshape(self.W.shape)

        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.num_filter, -1)

        W_flat = self.W.reshape(self.num_filter, -1)

        dX_col = W_flat.T @ dout_flat
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dX = col2im_indices(dX_col, shape, self.height_filter,
                            self.width_filter, self.padding, self.stride)

        return dX, [dW, db]


class FullyConnectedLayer():

    def __init__(self, in_size, out_size):

        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size / 2.)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        self.X = X
        out = self.X @ self.W + self.b
        return out

    def backward(self, dout):
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0)
        dX = dout @ self.W.T
        return dX, [dW, db]

class ReLU():
    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, dout):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, []


