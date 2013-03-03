#!/usr/bin/env python
'''
CREATED:2013-03-01 08:19:26 by Brian McFee <brm2132@columbia.edu>

Convolutional Dictionary Learning

'''

import numpy
import scipy.sparse, scipy.sparse.linalg
import functools

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
        Spatial L1 sparsity: assumes each column of X is a vectorized 2d-DFT of a 2d-signal
    '''
    raise Exception('not yet implemented')
    pass

def reg_group_l2(X, rho, m, lam):
    '''
        sum_i sum_k lam / rho * |X[group,i]|)|_2
    '''

    (d2m, n)    = X.shape

    # Compute norm-squared of real + imaginary
    dm          = d2m / 2
    V           = X[:dm,:]**2 + X[dm:,:]**2

    # Group by codeword
    d           = dm / m
    Z           = numpy.zeros( (m, n) )
    for k in xrange(m):
        Z[k,:]  = numpy.sum(V[(k * d):(k * d + d),:], axis=0)
        pass

    # Compute the soft-thresholding mask by group
    mask        = numpy.maximum(0, 1 - (lam / rho) * Z ** -0.5)

    # Duplicate each row of the mask, then tile it to catch the complex region
    mask        = numpy.tile(numpy.repeat(mask, d, axis=0), (2, 1))

    # Apply the soft-thresholding operator
    return mask * X

def reg_l2_ball(X, m):
    '''
        Input:      X 2*d*m-by-1 vector  (ndarray) of real and imaginary codewords
                    m >0    number of codewords

        Output:     X where each codeword is scaled to at most unit length
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
def encoder(X, D, reg, max_iter=30, dynamic_rho=False):
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

    #   FIXME:  2013-03-01 16:12:27 by Brian McFee <brm2132@columbia.edu>
    #   could be more efficient here, but this is the simplest to code
    #   also a one-off computation, NBD.

    # Precompute dictionary normalization
    Dnorm   = (D * D.T).diagonal()
    Dinv    = scipy.sparse.spdiags( (1.0 + Dnorm / rho)**-1, 0, d, d)

    # ADMM loop
    for t in xrange(max_iter):
        # TODO:   2013-03-02 08:38:09 by Brian McFee <brm2132@columbia.edu>
        #   parallelize me block-wise

        # Encode all the data
        A = __ridge(D, rho, DX + rho * (Z - O), Dinv)

        # Apply the regularizer
        Z = reg(A + O, rho)

        # Update residual
        O = O + A - Z

        if not dynamic_rho:
            continue

        # TODO:   2013-03-01 21:15:09 by Brian McFee <brm2132@columbia.edu>
        #  rescale rho by primal and dual gap

        # Update Dinv
        Dinv = scipy.sparse.spdiags( (1 + rho * Dnorm)**-1, 0, d, d)
        pass
    return Z
#---                            ---#

#--- Dictionary                 ---#
def dictionary_block_parallel(X, A, max_iter=30, dynamic_rho=False):

    # Get the shape of the data
    (d2, n) = X.shape
    d2m     = A.shape[0]

    d       = d2 / 2
    m       = d2m / d2

    # Initialize ADMM variables
    rho     = 1.0

    D       = numpy.zeros(d2m)      # The global codebook
    Di      = numpy.zeros(d2m * n)  # Point-wise codebooks
    Ei      = numpy.zeros(d2m * n)  # Point-wise residuals

    # Pre-compute targets

    def __rearrange():
        for i in xrange(n):
            Si      = diagonalBlockRI(blockify(A[:,i], m))  # 2d-by-2dm encoding, sparse
            SXi     = Si.T * X[:,i]                         # 2dm, dense
            Snormi  = (Si * Si.T).diagonal()                # 2d, dense
            
            if i == 0:
                S       = Si
                SX      = SXi
                Snorm   = Snormi
            else:
                # Build up block-diagonals
                S       = scipy.sparse.bmat( [[S, None], [None, Si]])   
                # Stack the data and normalizers horizontally
                SX      = numpy.hstack( (SX, SXi) )
                Snorm   = numpy.hstack( (Snorm, Snormi) )
                pass
            pass
        # Pre-compute Sinv: 2dn-by-2dn diagonal sparse
        Sinv    = scipy.sparse.spdiags( (1.0 + Snorm / rho)**-1, 0, len(Snorm), len(Snorm) )
        return (S, SX, Sinv)

    # FIXME:  2013-03-03 11:17:45 by Brian McFee <brm2132@columbia.edu>
    #   rearrange for dynamic rho: we'll need Snorm separately

    (S, SX, Sinv) = __rearrange()

    # Tile the global dictionary to compute residuals
    Dtile = numpy.tile(D, (1, n))[0]

    for t in xrange(max_iter):
        
        # Solve for pointwise codebooks
        Di  = __ridge(S, rho, SX + rho * (Dtile - Ei), Sinv)

        # Combine pointwise codebooks
        RES = numpy.sum(numpy.reshape(Di + Ei, (d2m, n), order='F'), axis=1)

        D   = reg_l2_ball(RES, m)[:,0]

        # Tile the global dictionary to compute residuals
        Dtile = numpy.tile(D, (1, n))[0]

        # Update residuals
        Ei  = (Ei + Di) - Dtile

        if not dynamic_rho:
            continue

        # TODO:   2013-03-03 10:53:03 by Brian McFee <brm2132@columbia.edu>
        #  update rho

        pass

    # Recombobulate the dictionary
    return diagonalBlockRI(blockify(D, m))

# TODO:   2013-03-03 11:51:52 by Brian McFee <brm2132@columbia.edu>
# redo using scipy.sparse.linalg.spsolve, or related method
#   probably try a few to see which is fastest
#   screw this parallel, pointwise encoding business

def dictionary(X, A, max_iter=30, dynamic_rho=False):

    (d2, n) = X.shape
    d2m     = A.shape[0]

    d       = d2 / 2
    m       = d2m / d2

    # Initialize ADMM variables
    rho     = 1.0

    D       = numpy.zeros( d2m )   # Unconstrained codebook
    E       = numpy.zeros_like(D)       # Constrained codebook
    W       = numpy.zeros_like(E)       # Scaled dual variables

    StX     = numpy.zeros_like(D)

    # Aggregate the scatter and target matrices
    for i in xrange(n):
        Si  = diagonalBlockRI(blockify(A[:,i], m))
        StX     = StX + Si.T * X[:,i]
        if i == 0:
            StS     = Si.T * Si
        else:
            StS     = StS + Si.T * Si
        pass

    # We need to solve:
    #   D <- (rho * I + StS) \ (StX + rho * (E - W) )
    #   pre-cache the cholesky factor (L * L') = (rho * I + StS), then use back-substitution:
    #   z <- L \ (StX + rho * (E - W))
    #   D <- L.T \ z
    #   looks like scipy.sparse.linalg.factorized does this?

    SOLVER  = scipy.sparse.linalg.factorized( rho * scipy.sparse.eye(d2m, d2m) + StS)

    for t in xrange(max_iter):
        # Solve for the unconstrained codebook
        D   = SOLVER( StX + rho * (E - W) )

        # Project onto the feasible set
        E   = reg_l2_ball(D + W, m)

        # Update the residual
        W   = W + D - E

        if not dynamic_rho:
            continue

        # TODO:   2013-03-03 14:53:57 by Brian McFee <brm2132@columbia.edu>
        # update rho? 
        SOLVER  = scipy.sparse.linalg.factorized( rho * scipy.sparse.eye(d2m, d2m) + StS)
        pass

    return diagonalBlockRI(blockify(E, m))

def dictionary_old(X, A, max_iter=30, dynamic_rho=False):
    '''
    Optimize a dictionary

    Input:
        X:  2*d-by-n        data matrix
        A:  2*d*m-by-n      encoding matrix
        max_iter:           maximum iterations of ADMM
    '''

    # Get the shapes
    (d2, n) = X.shape
    d2m     = A.shape[0]

    d       = d2 / 2
    m       = d2m / d2

    # Initialize ADMM variables
    rho     = 1.0

    D       = numpy.zeros((d2m, 1))     # The global codebook
    Di      = numpy.zeros((d2m, n))     # Point-wise codebooks
    Ei      = numpy.zeros((d2m, n))     # Point-wise residuals

    # Pre-compute targets

    # FIXME:  2013-03-01 21:41:27 by Brian McFee <brm2132@columbia.edu>
    #   this is frickin horrendous...
    #   can scipy.sparse do 3d arrays/tensors??

    #     TODO:   2013-03-02 21:04:25 by Brian McFee <brm2132@columbia.edu>
    # can we block these guys up vertically and make one call to ridge? 

    S       = []
    SX      = []
    Snorm   = []
    Sinv    = []
    for i in xrange(n):
        S.append(diagonalBlockRI(blockify(A[:,i], m)))
        SX.append( S[-1].T * X[:,i])
        Snorm.append((S[-1] * S[-1].T).diagonal())
        Sinv.append(scipy.sparse.spdiags( (1.0 + Snorm[-1] / rho)**-1, 0, 2*d, 2*d))
        pass

    for t in xrange(max_iter):

        # Optimize all the codebooks
        for i in xrange(n):
            # TODO:   2013-03-01 21:42:54 by Brian McFee <brm2132@columbia.edu>
            # parallelize me         
            Di[:, i] = __ridge(S[i], rho, SX[i] + rho * (D[:,0] - Ei[:,i]), Sinv[i])
            pass

        # Combine point-wise solutions and project
        D = reg_l2_ball(numpy.sum(Di, axis=1) + numpy.sum(Ei, axis=1), m)

        # Update residuals.  Using array broadcast over all examples here.
        Ei = (Ei + Di) - D

        if not dynamic_rho:
            continue

        # TODO:   2013-03-01 21:46:21 by Brian McFee <brm2132@columbia.edu>
        # update rho

        # Re-compute the inverses
        for i in xrange(n):
            Sinv[i] = scipy.sparse.spdiags( (1.0 + Snorm[-1] / rho)**-1, 0, 2*d, 2*d)
            pass
        pass

    # Reshape D
    return diagonalBlockRI(blockify(D, m))
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
        # FIXME:  2013-03-02 19:20:10 by Brian McFee <brm2132@columbia.edu>
        #   probably better to initialize with columns of X, not random noise

        D = numpy.random.randn( d2, m )
        # Normalize the codebook
        D = D / (numpy.sum(D ** 2, axis=0))**0.5
        # Reshape into diagonalized size
        D = diagonalBlockRI(D)
        pass

    for T in xrange(max_steps):
        # TODO:   2013-03-02 19:08:30 by Brian McFee <brm2132@columbia.edu>
        # add diagonostics, progress-bar output

        A = encoder(X, D, reg, max_iter=max_admm_steps)
        D = dictionary(X, A)
        pass

    A = encoder(X, D, reg, max_iter=max_admm_steps)
    return (D, A)
#---                            ---#
