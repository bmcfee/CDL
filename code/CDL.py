#!/usr/bin/env python
# CREATED:2013-05-03 21:57:00 by Brian McFee <brm2132@columbia.edu>
# sklearn.decomposition container class for CDL 

import cdl
from sklearn.base import BaseEstimator, TransformerMixin

class ConvolutionalDictionaryLearning(BaseEstimator, TransformerMixin):

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
        pass

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

        extra_args = {}
        if self.penalty == 'l1_space':
            extra_args['height']    = X.shape[1]
            extra_args['width']     = X.shape[2]
            extra_args['pad_data']  = self.pad_data
            extra_args['nonneg']    = self.nonneg

        # Swap the axes around
        X = X.swapaxes(0,1).swapaxes(1,2)

        # X is now h-*-w-by-n
        X = cdl.patches_to_vectors(X, pad_data=self.pad_data)

        encoder, D, diagnostics = cdl.learn_dictionary(X, self.n_atoms,
                                                    reg         = self.penalty,
                                                    alpha       = self.alpha,
                                                    max_steps   = self.n_iter,
                                                    n_threads   = self.n_jobs,
                                                    batch_size  = self.chunk_size,
                                                    verbose     = self.verbose,
                                                    shuffle     = self.shuffle,
                                                    **extra_args)
        self.components_ = D
        self.encoder_    = encoder
        return self

    def transform(self, X):
        '''Encode the data as a sparse convolution of the dictionary atoms.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_y, n_features_x)

        Returns
        -------
        X_new : array, shape (n_samples, n_components, n_features_y, n_features_x)
        '''

        # Swap axes
        X = X.swapaxes(0,1).swapaxes(1,2)

        h, w, n = X.shape

        # FIXME:  2013-05-03 22:52:30 by Brian McFee <brm2132@columbia.edu>
        # can we make this more memory-efficient?         
        # Fourier transform
        X_new = cdl.patches_to_vectors(X, pad_data=self.pad_data)

        # Encode
        X_new = self.encoder_(X_new)

        # Reshape each activation into its own frame
        X_new = X_new.reshape( (X_new.shape[0] / self.n_atoms, n * self.n_atoms),
                                order='F')

        # Transform back
        X_new = cdl.vectors_to_patches(X_new, w, pad_data=self.pad_data, real=True)
        # X_new is now h-by-w-by-n*n_atoms

        X_new = X_new.reshape( (h, w, n, self.n_atoms), order='F')

        # Now X_new is h-by-w-by-n-by-n_atoms.  Permute into the shape we want
        X_new = X_new.swapaxes(1,2).swapaxes(2,3).swapaxes(0,1).swapaxes(1,2)

        return X_new
