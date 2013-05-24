#!/usr/bin/env python
'''sklearn.decomposition container class for Convolutional dictionary learning

CREATED:2013-05-03 21:57:00 by Brian McFee <brm2132@columbia.edu>
'''

import numpy as np
import functools
import random
from sklearn.base import BaseEstimator, TransformerMixin

from joblib import Parallel, delayed
import _cdl


class ConvolutionalDictionaryLearning(BaseEstimator, TransformerMixin):
    '''Convolutional mini-batch dictionary learning'''

    def __init__(self, n_atoms, alpha=1, penalty='l1_space', nonneg=True, 
                                pad_data=False, n_iter=100, 
                                n_jobs=1, chunk_size=32, shuffle=True,
                                random_state=None, verbose=False):
        '''Mini-batch convolutional dictionary learning.

        (D, A) = argmin 0.5 sum || X^i - sum_k D^i_k (*) A^i_k||^2 + alpha * g(A^i_k)
                  (D, A)     i

        Arguments
        ---------
        n_atoms : int    
            Number of dictionary elements to extract

        alpha : float  
            Regularization penalty

        penalty : {'l1_space', 'l1', 'l2_group'}
            Sparsity penalty on activations.

        nonneg : bool
            Constrain activations to be non-negative

        pad_data : bool   
            Should the data be zero-padded before transforming?

        n_iter   : int    
            Total number of iterations to perform

        n_jobs   : int
            Number of parallel jobs to run

        chunk_size : int    
            Number of data points per batch

        shuffle : bool   
            Whether to shuffle the data before forming each batch

        random_state : int or RandomState
            Pseudo random number generator used for sampling

        verbose : bool
            Whether to show progress during training
        '''

        self.n_atoms        = n_atoms
        self.alpha          = alpha
        self.penalty        = penalty
        self.nonneg         = nonneg
        self.pad_data       = pad_data
        self.n_iter         = n_iter
        self.n_jobs         = n_jobs
        self.chunk_size     = chunk_size
        self.shuffle        = shuffle
        self.random_state   = random_state
        self.verbose        = verbose

    def data_generator(self, X_full):
        '''Make a CDL data generator from an input array
        
        Arguments
        ---------
            X_full      --  (ndarray) n-by-h-by-w data array

        Returns
        -------
            batch_gen   --  (generator) batches of CDL-transformed input data
                                        the generator will loop infinitely
        '''


        # 1. initialize the RNG
        if type(self.random_state) is int:
            random.seed(self.random_state)
        elif self.random_state is not None:
            random.setstate(self.random_state)

        n = X_full.shape[0]

        indices = range(n)

        while True:
            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, n, self.chunk_size):
                if i + self.chunk_size > n:
                    break

                X = X_full[i:i+self.chunk_size]

                # Swap the axes around
                X = X.swapaxes(0, 1).swapaxes(1, 2)

                # X is now h-*-w-by-n
                yield _cdl.patches_to_vectors(X, pad_data=self.pad_data)


    def fit(self, X):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features_y, n_features_x]
            Training data patches.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''

        width = X.shape[2]
        extra_args = {}
        if self.penalty == 'l1_space':
            extra_args['height']    = X.shape[1]
            extra_args['width']     = X.shape[2]
            extra_args['pad_data']  = self.pad_data
            extra_args['nonneg']    = self.nonneg


        encoder, D, diagnostics = _cdl.learn_dictionary(
                                        self.data_generator(X),
                                        self.n_atoms,
                                        reg         = self.penalty,
                                        alpha       = self.alpha,
                                        max_steps   = self.n_iter,
                                        verbose     = self.verbose,
                                        **extra_args)

        self.fft_components_ = D
        D                   = _cdl.diags_to_columns(D)
        D                   = _cdl.vectors_to_patches(D, width, 
                                    pad_data=self.pad_data)
        self.components_    = D.swapaxes(1, 2).swapaxes(0, 1)
        self.encoder_       = encoder
        self.diagnostics_   = diagnostics
        return self

    def set_codebook(self, D):
        '''Clobber the existing codebook with a new one.'''

        self.components_ = D
        D = D.swapaxes(0, 1).swapaxes(1, 2)
        D = _cdl.patches_to_vectors(D, pad_data=self.pad_data)
        D = _cdl.columns_to_diags(D)
        self.fft_components_ = D
        self.encoder_ = functools.partial(self.encoder_, D=D)

        return self

    def _transform(self, X):
        '''Encode the data as a sparse convolution of the dictionary atoms.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_y, n_features_x)

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms, n_features_y, n_features_x)
        '''

        # Swap axes
        X = X.swapaxes(0, 1).swapaxes(1, 2)

        h, w, n = X.shape

        # Fourier transform
        X_new = _cdl.patches_to_vectors(X, pad_data=self.pad_data)

        # Encode
        X_new = self.encoder_(X_new)
        
        X_new = _cdl.real2_to_complex(X_new)
        X_new = X_new.reshape( (-1, X_new.shape[1] * self.n_atoms), order='F')
        X_new = _cdl.complex_to_real2(X_new)   

        X_new = _cdl.vectors_to_patches(X_new, w, 
                            pad_data=self.pad_data, real=True)

        X_new = X_new.reshape( (X_new.shape[0], X_new.shape[1], 
                            self.n_atoms, n), order='F')

        X_new = X_new.swapaxes(3, 0).swapaxes(3, 2).swapaxes(3, 1)

        if self.nonneg:
            X_new = np.maximum(X_new, 0.0)

        return X_new

    def transform(self, X, chunk_size=512):
        '''Encode the data as a sparse convolution of the dictionary atoms.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_y, n_features_x)

        Returns
        -------
        X_new : array, shape (n_samples, n_atoms, n_features_y, n_features_x)
        '''


        def chunker(c):
            '''Data chunk generator'''
            n = X.shape[0]
            c = min(n, c)
            for i in range(0, n, c):
                end = min(i + c, n)
                yield X[i:end]

        X_new = Parallel(n_jobs=self.n_jobs)(
                    delayed(global_transform)(B, self.pad_data, self.encoder_,
                    self.n_atoms, self.nonneg) for B in chunker(chunk_size)
                )

        return np.vstack(X_new)

def global_transform(X_batch, pad_data, encoder_, n_atoms, nonneg):
    X_batch = X_batch.swapaxes(0, 1).swapaxes(1, 2)

    h, w, n = X_batch.shape

    # Fourier transform
    X_new = _cdl.patches_to_vectors(X_batch, pad_data=pad_data)

    # Encode
    X_new = encoder_(X_new)

    X_new = _cdl.real2_to_complex(X_new)
    X_new = X_new.reshape( (-1, X_new.shape[1] * n_atoms), order='F')
    X_new = _cdl.complex_to_real2(X_new)   

    X_new = _cdl.vectors_to_patches(X_new, w, 
                                    pad_data=pad_data, 
                                    real=True)

    X_new = X_new.reshape( (X_new.shape[0], X_new.shape[1], 
                                n_atoms, n), 
                            order='F')

    X_new = X_new.swapaxes(3, 0).swapaxes(3, 2).swapaxes(3, 1)

    if nonneg:
        X_new = np.maximum(X_new, 0.0)

    return X_new
