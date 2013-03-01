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
    A       = scipy.sparse.spdiags(D.T, range(0, - d * m, -d), d * m, d)
    return A.T.tocsr()

def columnsFromDiags(Q):
    '''
    Input:  2d-by-2dm sparse matrix Q
    Output: 2d-by-m dense matrix D of diagonals 
            from the upper and lower block of Q
    '''
    # Q = [A, -B ; B A]
    # cut to the first half of columns
    # then break vertically

    (d2, d2m)  = Q.shape

    d   = d2    / 2
    dm  = d2m   / 2
    
    
    D = numpy.empty( (d2, dm / d) )

    for k in xrange(0, dm, d):
        D[:d,k/d] = Q[range(d), range(k, k + d)]
        D[d:,k/d] = Q[range(d, d2), range(k, k + d)]
        pass

    return D

def diagonalBlockRI(D):
    '''
    Input:
        D:  2d-by-m matrix of real+imaginary vectors

    Output:
        Q:  2d-by-2dm sparse diagonal block matrix [A, -B ; B, A]
            where A and B are derived from the real and imaginary components
    '''

    # Get the size of each codeword
    d = D.shape[0] / 2

    # Block the real component
    A = sparseDiagonalBlock(D[:d,:])

    # Block the imaginary component
    B = sparseDiagonalBlock(D[d:,:])

    # Stack horizontally
    Q1 = scipy.sparse.hstack([A, -B])
    Q2 = scipy.sparse.hstack([B, A])

    return scipy.sparse.vstack([Q1, Q2]).tocsr()

def vectorize(X):
    '''
    Input:  X 2d-by-m array
    Output: Y 2dm-by-1 array

    If X = [A ; B], then Y = [vec(A) ; vec(B)]
    '''

    (d2, m) = X.shape

    d = d2 / 2

    A = numpy.reshape(X[:d,:], (d * m, 1), order='F')
    B = numpy.reshape(X[d:,:], (d * m, 1), order='F')
    return numpy.vstack( (A, B) )

def blockify(AB, m):
    '''
    Input:  AB  2dm-by-1 array
            m   number of columns
    Output: X   2d-by-m array
    '''

    d2m = AB.shape[0]

    d = d2m / (2 * m)

    A = numpy.reshape(AB[:(d*m)], (d, m), order='F')
    B = numpy.reshape(AB[(d*m):], (d, m), order='F')

    return numpy.vstack( (A, B) )


def __ridge(A, rho, b, Z):
    '''
    Specialized ridge regression solver for hadamard products.

        X <- 1/rho  *  (I + 1/rho * A' * Z * A) * b

        A:      2d-by-2dm
        rho:    scalar > 0
        b:      2dm
        Z:      2d > 0,  == diag(inv(I + 1/rho * A * A.T))

    '''
    # b is an ndarray
    # Q is a sparse matrix

    return (b + (A.T * (Z**(-1) * (A * b)) / rho)) / rho
#---                            ---#


#--- Regression function        ---#
#---                            ---#


#--- Regularization functions   ---#
#---                            ---#


#--- Encoder loop               ---#
#---                            ---#

#--- Dictionary loop            ---#
#---                            ---#

