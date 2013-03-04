#!/usr/bin/env python
'''
CREATED:2013-03-01 08:19:26 by Brian McFee <brm2132@columbia.edu>

Convolutional Dictionary Learning

'''

import numpy
import scipy.linalg, scipy.sparse, scipy.sparse.linalg

#--- magic numbers              ---#
RHO_MIN     =   1e-2
RHO_MAX     =   1e4
ABSTOL      =   1e-3
RELTOL      =   1e-2
MU          =   10.0
TAU         =   2
#---                            ---#

#--- Utility functions          ---#

def complexToReal2(X):
    '''
    Separate the real and imaginary components of a matrix

    See also: real2ToComplex()

    Input:
        complex d-by-n matrix X

    Output:
        real 2d-by-n matrix Y = [ real(X) ; imag(X) ]
    '''
    return numpy.vstack((X.real, X.imag))

def real2ToComplex(Y):
    '''
    Combine the real and imaginary components of a matrix

    See also: complexToReal2()

    Input:
        real 2d-by-n matrix Y = [ real(X) ; imag(X) ]

    Output:
        complex d-by-n matrix X
    '''
    d = Y.shape[0] / 2
    return Y[:d] + 1.j * Y[d:]


def diagsToColumns(Q):
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

def columnsToDiags(D):
    '''
    Input:
        D:  2d-by-m matrix of real+imaginary vectors

    Output:
        Q:  2d-by-2dm sparse diagonal block matrix [A, -B ; B, A]
            where A and B are derived from the real and imaginary components
    '''

    def __sparseDiagonalBlock(_D):
        '''
        Rearrange a d-by-m matrix D into a sparse d-by-dm matrix Q
        The i'th d-by-d block of Q = diag(D[:,i])
        '''

        (_d, _m)  = _D.shape
        _A       = scipy.sparse.spdiags(_D.T, range(0, - _d * _m, -_d), _d * _m, _d)
        return _A.T

    # Get the size of each codeword
    d = D.shape[0] / 2

    # Block the real component
    A = __sparseDiagonalBlock(D[:d,:])

    # Block the imaginary component
    B = __sparseDiagonalBlock(D[d:,:])

    # Block up everything in csr format
    return scipy.sparse.bmat([ [ A, -B], [B, A] ], format='csr')

def columnsToVector(X):
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

def vectorToColumns(AB, m):
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

def normalizeDictionary(D):
    '''
    Normalize a codebook to have all unit-length bases.

    Input is assumed to be in diagonal-block format.

    '''
    D = diagsToColumns(D)
    D = D / (numpy.sum(D**2, axis=0) ** 0.5)
    D = columnsToDiags(D)
    return D
#---                            ---#


#--- Regression function        ---#
def __ridge(A, rho, b, Z):
    '''
    Specialized ridge regression solver for Hadamard products.

    Not for external use.

    Input:
        A:      2d-by-2dm
        rho:    scalar > 0
        b:      2dm
        Z:      2d > 0,  == diag(inv(I + 1/rho * A * A.T))

    Output:
        X = 1/rho  *  (I - 1/rho * A' * Z * A) * b
    '''
    return (b - (A.T * (Z * (A * b)) / rho)) / rho
#---                            ---#


#--- Regularization functions   ---#
def reg_time_l1(X, rho, lam):
    '''
        Temporal L1 sparsity: assumes each column of X is (DFT) of a time-series.

        sum_i lam / rho * |FFTinv(X[:,i])|_1

        The proper way to use these is as follows:

        from functools import partial

        g = functools.partial(CDL.reg_time_l1, lam=0.5)
        A = CDL.encode(X, D, g)
    '''
    raise Exception('not yet implemented')
    pass

def reg_space_l1(X, rho, lam, w, h):
    '''
        Spatial L1 sparsity: assumes each column of X is a columnsToVectord 2d-DFT of a 2d-signal
    '''
    #   TODO:   2013-03-04 08:29:21 by Brian McFee <brm2132@columbia.edu>
    #   need to do iffts on each block independently

    raise Exception('not yet implemented')
    pass

def reg_group_l2(X, rho, lam, m):
    '''
    For each column of X, break the rows into m groups
    shrink each group by soft-thresholding
    '''

    (d2m, n)    = X.shape

    # Compute norm-squared of real + imaginary
    dm          = d2m / 2
    V           = X[:dm,:]**2 + X[dm:,:]**2

    # Group by codeword
    d           = dm / m
    Z           = numpy.zeros( (m, n) )
    for k in xrange(m):
        Z[k,:]  = numpy.sum(V[(k * d):(k * d + d),:], axis=0)**0.5
        pass

    Z[Z < rho / lam] = rho / lam
    # Compute the soft-thresholding mask by group
    mask        = numpy.maximum(0, 1 - (lam / rho) / Z)

    # Duplicate each row of the mask, then tile it to catch the complex region
    mask        = numpy.tile(numpy.repeat(mask, d, axis=0), (2, 1))

    # Apply the soft-thresholding operator
    return mask * X

def proj_l2_ball(X, m):
    '''
        Input:      X 2*d*m-by-1 vector  (ndarray) of real and imaginary codewords
                    m >0    number of codewords

        Output:     X where each codeword is projected onto the unit l2 ball
    '''
    d2m     = X.shape[0]
    d       = d2m / (2 * m)

    #         Real part        Imaginary part
    Xnorm   = X[:(d2m/2)]**2 + X[(d2m/2):]**2   

    # Group by codewords
    Z = numpy.empty(m)
    for k in xrange(0, m * d, d):
        Z[k/d] = min(1.0, numpy.sum(Xnorm[k:(k+d)])**-0.5)
        pass

    # Repeat and tile each norm
    Z = numpy.tile(numpy.repeat(Z, d), (1, 2))

    # Project
    Xp = numpy.zeros(d2m)
    Xp[:] = Z * X
    return Xp
#---                            ---#



#--- Encoder                    ---#
def encoder(X, D, reg, max_iter=500, dynamic_rho=True):
    '''
    Encoder

    Input:
        X:          2d-by-n     data
        D:          2d-by-2dm   codebook
        reg:        regularization function.

                    Example:
                    reg = functools.partial(CDL.reg_group_l2, lam=0.5, m=num_codewords)

        max_iter:   # of iterations to run the encoder  (Default: 30)

        dynamic_rho: re-scale the augmented lagrangian term?    (Default: False)

    Output:
        A:          2dm-by-n    encoding matrix
    '''

    (d, dm) = D.shape
    n       = X.shape[1]

    # Initialize split parameter
    Z   = numpy.zeros( (dm, n) )
    O   = numpy.zeros( (dm, n) )

    # Initialize augmented lagrangian weight
    rho = 1.0

    # Precompute D'X
    DX  = D.T * X

    # Precompute dictionary normalization
    Dnorm   = (D * D.T).diagonal()
    Dinv    = scipy.sparse.spdiags( (1.0 + Dnorm / rho)**-1, 0, d, d)

    # ADMM loop
    for t in xrange(max_iter):
        # Encode all the data
        A = __ridge(D, rho, DX + rho * (Z - O), Dinv)

        # Apply the regularizer
        Zold = Z
        Z = reg(A + O, rho)

        # Update residual
        O = O + A - Z

        #  compute stopping criteria
        ERR_primal  = scipy.linalg.norm(A - Z)
        ERR_dual    = rho * scipy.linalg.norm(Z - Zold)

        eps_primal  = (dm**0.5) * ABSTOL + RELTOL * max(scipy.linalg.norm(A), scipy.linalg.norm(Z))
        eps_dual    = (dm**0.5) * ABSTOL + RELTOL * scipy.linalg.norm(O)

        if t % 50 == 0:
            print '%04d| Encoder: [%.2e < %.2e]\tand\t[%.2e < %.2e]?' % (t, ERR_primal, eps_primal, ERR_dual, eps_dual)
        if ERR_primal < eps_primal and ERR_dual <= eps_dual:
            break

        if not dynamic_rho:
            continue

        rho_changed = False

        if ERR_primal > MU * ERR_dual and rho < RHO_MAX:
            rho         = rho   * TAU
            O           = O     / TAU
            rho_changed = True
        elif ERR_dual > MU * ERR_primal and rho > RHO_MIN:
            rho         = rho   / TAU
            O           = O     * TAU
            rho_changed = True
            pass

        # Update Dinv
        if rho_changed:
            Dinv = scipy.sparse.spdiags( (1 + rho * Dnorm)**-1.0, 0, d, d)
            pass
        pass
    return Z
#---                            ---#

#--- Dictionary                 ---#
def dictionary(X, A, max_iter=500, dynamic_rho=True):

    (d2, n) = X.shape
    d2m     = A.shape[0]

    d       = d2  / 2
    m       = d2m / d2

    # Initialize ADMM variables
    rho     = 1.0

    D       = numpy.zeros( d2m )        # Unconstrained codebook
    E       = numpy.zeros_like(D)       # l2-constrained codebook
    W       = numpy.zeros_like(E)       # Scaled dual variables


    # Aggregate the scatter and target matrices
    def __aggregator():
        Si      = columnsToDiags(vectorToColumns(A[:,0], m))
        StX     = Si.T * X[:,0]
        StS     = Si.T * Si
        for i in xrange(1, n):
            Si          = columnsToDiags(vectorToColumns(A[:,i], m))
            StX         = StX + Si.T * X[:,i]
            StS         = StS + Si.T * Si
            pass
        return (StS, StX)

    (StS, StX) = __aggregator()

    # We need to solve:
    #   D <- (rho * I + StS) \ (StX + rho * (E - W) )
    #   Use the sparse factorization solver to pre-compute cholesky factors

    SOLVER  = scipy.sparse.linalg.factorized( rho * scipy.sparse.eye(d2m, d2m) + StS)

    for t in xrange(max_iter):
        # Solve for the unconstrained codebook
        D   = SOLVER( StX + rho * (E - W) )

        # Project each basis element onto the l2 ball
        Eold = E
        E   = proj_l2_ball(D + W, m)

        # Update the residual
        W   = W + D - E

        #  compute stopping criteria
        ERR_primal  = scipy.linalg.norm(D - E)
        ERR_dual    = rho * scipy.linalg.norm(E - Eold)

        eps_primal  = (d2m**0.5) * ABSTOL + RELTOL * max(scipy.linalg.norm(D), scipy.linalg.norm(E))
        eps_dual    = (d2m**0.5) * ABSTOL + RELTOL * scipy.linalg.norm(W)
        
        if ERR_primal < eps_primal and ERR_dual <= eps_dual:
            break

        if not dynamic_rho:
            continue

        rho_changed = False
        if ERR_primal > MU * ERR_dual and rho < RHO_MAX:
            rho = rho   * TAU
            W   = W     / TAU
            rho_changed = True
        elif ERR_dual > MU * ERR_primal and rho > RHO_MIN:
            rho = rho   / TAU
            W   = W     * TAU
            rho_changed = True
            pass

        if rho_changed:
            SOLVER  = scipy.sparse.linalg.factorized( rho * scipy.sparse.eye(d2m, d2m) + StS)
            pass
        pass

    return columnsToDiags(vectorToColumns(E, m))
#---                            ---#

#--- Alternating minimization   ---#
def learn_dictionary(X, m, reg, max_steps=50, max_admm_steps=30, D=None):
    '''
    Alternating minimization to learn convolutional dictionary

    Input:
        X:      2d-by-n     data matrix, real/imaginary-separated
        m:      number of filters to learn
        reg:    regularization function handle
        max_steps:      number of outer-loop steps
        max_admm_steps: number of inner loop steps
        D:      initial codebook

    Output:
        (D, A) where X ~= D * A
    '''

    (d2, n) = X.shape

    if D is None:
        # Initialize a random dictionary
        D = numpy.random.randn(d2, m)
#         D = X[:, numpy.random.randint(0, X.shape[1], m)]
        # Pick m random columns from the input
        D = normalizeDictionary(columnsToDiags(D))
        pass


    for T in xrange(max_steps):
        # Encode the data
        A = encoder(X, D, reg, max_iter=max_admm_steps)
        print '%2d| A-step MSE=%.3f' % (T, numpy.mean((D * A - X)**2))
        # Optimize the codebook
        D = dictionary(X, A, max_iter=max_admm_steps)
        print '__| D-step MSE=%.3f' %  numpy.mean((D * A - X)**2)

        # Normalize the codebook
        D = normalizeDictionary(D)
        pass

    # Re-encode the data with the final codebook
    A = encoder(X, D, reg, max_iter=max_admm_steps)
    return (D, A)
#---                            ---#
