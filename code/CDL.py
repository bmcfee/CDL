#!/usr/bin/env python
'''
CREATED:2013-03-01 08:19:26 by Brian McFee <brm2132@columbia.edu>

Convolutional Dictionary Learning

'''

import numpy
import scipy.sparse

#--- Utility functions          ---#

def separateComplex(X):
    '''
    Separate the real and imaginary components of a matrix

    See also: combineComplex()

    Input:
        complex d-by-n matrix X

    Output:
        real 2d-by-n matrix Y = [ real(X) ; imag(X) ]
    '''
    return numpy.vstack((X.real, X.imag))

def combineComplex(Y):
    '''
    Combine the real and imaginary components of a matrix

    See also: separateComplex()

    Input:
        real 2d-by-n matrix Y = [ real(X) ; imag(X) ]

    Output:
        complex d-by-n matrix X
    '''
    d = Y.shape[0] / 2
    return Y[:d] + 1.j * Y[d:]

def sparseDiagonalBlock(D):
    '''
    Rearrange a d-by-m matrix D into a sparse d-by-dm matrix Q
    The i'th d-by-d block of Q = diag(D[:,i])
    '''

    (d, m)  = D.shape
    Q       = scipy.sparse.spdiags(D.T, range(0, - d * m, -d), d * m, d)
    return Q.T

#---                            ---#


#--- Regression function        ---#
#---                            ---#


#--- Regularization functions   ---#
#---                            ---#


#--- Encoder loop               ---#
#---                            ---#

#--- Dictionary loop            ---#
#---                            ---#

