#!/usr/bin/env python
'''
CREATED:2013-03-01 08:19:26 by Brian McFee <brm2132@columbia.edu>

Convolutional Dictionary Learning

'''

import numpy
import scipy.linalg, scipy.sparse, scipy.sparse.linalg
import scipy.weave

#--- magic numbers              ---#
RHO_MIN     =   1e-4        # Minimum allowed scale for augmenting term rho
RHO_MAX     =   1e4         # Maximum allowed scale for rho
ABSTOL      =   1e-4        # absolute tolerance for convergence criteria
RELTOL      =   1e-3        # relative tolerance
MU          =   10.0        # maximum ratio between primal and dual residuals
TAU         =   2           # scaling factor for rho when primal/dual is off by more than MU
T_CHECKUP   =   10          # number of steps between convergence tests and rho-rescaling
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
    if Y.ndim > 1:
        return Y[:d,:] + 1.j * Y[d:,:]
    else:
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
    m   = dm / d
    
    D = numpy.empty( (d2, m) )

    for k in xrange(0, d * m, d):
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

def reg_space_l1(A, rho, lam, w, h):
    '''
        Spatial L1 sparsity: assumes each column of X is a columnsToVectord 2d-DFT of a 2d-signal

        Input: 
                A   = 2*d*m-by-n
                rho > 0
                lam > 0
                w, h: d = w * h

    '''

    (d2m, n) = A.shape
    d       = w * h
    m       = d2m / (2 * d)

    # Reshape activations, transform each one back into image space
    Aspace  = numpy.fft.ifft2(numpy.reshape(real2ToComplex(A), (w, h, m, n), order='F'), axes=(0, 1)).real

    # Apply shrinkage
    Ashrunk           = Aspace - (lam / rho) * numpy.sign(Aspace)
    Ashrunk[numpy.abs(Aspace) < (lam / rho)]    = 0

    # Transform back, reshape, and separate real from imaginary
    Atime               = numpy.reshape(numpy.fft.fft2(Ashrunk, axes=(0,1)), (w * h * m, n), order='F')

    return complexToReal2(Atime)


def reg_group_l2(X, rho, lam, m):
    '''
    For each column of X, break the rows into m groups
    shrink each group by soft-thresholding
    '''
    # TODO:   2013-03-06 14:00:34 by Brian McFee <brm2132@columbia.edu>
    # weave this function

    (d2m, n)    = X.shape
    dm          = d2m / 2
    d           = dm / m
    
    # Group 2-norm by codeword
    Vd          = numpy.reshape(X[:(d * m),:]**2 + X[(d * m):,:]**2, (d, m * n), order='F')
    Z           = numpy.reshape(numpy.sum(Vd, axis=0)**0.5, (m, n), order='F')

    # Avoid numerical underflow: these entries will get squashed to 0 in the mask anyway
    Z[Z < (lam / rho)]  = lam / rho        

    # Compute the soft-thresholding mask by group
    mask                = numpy.maximum(0, 1 - (lam / rho) / Z)

    # Duplicate each row of the mask, then tile it to catch the complex region
    mask                = numpy.tile(numpy.repeat(mask, d, axis=0), (2, 1))

    # Apply the soft-thresholding operator
    return mask * X

def reg_group_l2_weave(X, rho, lam, m):
    '''
    For each column of X, break the rows into m groups
    shrink each group by soft-thresholding
    '''

    (d2m, n)    = X.shape
    dm          = d2m / 2
    d           = dm / m
    
    #   1.  compute sub-vector l2 norms
    #   2.  apply soft-thresholding group-wise

    # Group 2-norm by codeword
    Z           = numpy.zeros( (m, n) )

    l2_subvectors = r"""

        for (int i = 0; i < n; i++) {
            // loop over data points

            for (int k = 0; k < m; k++) {
                // loop over codewords

                for (int j = 0; j < d; j++) {
                    // accumulate over codeword coordinates (real and imaginary)
                    Z[(k*n) + i]   
                                +=      X[(k * d + j) * n       +   i]   
                                    *   X[(k * d + j) * n       +   i] 
                                +       X[((k + m) * d + j) * n +   i]   
                                    *   X[((k + m) * d + j) * n +   i];
                }
                Z[(k * n) + i] = sqrt(Z[(k * n) +i]);
            }
        }

    """

    # Execute the inline code
    scipy.weave.inline(l2_subvectors, ['n', 'm', 'd', 'X', 'Z'])

    ### 
    # soft-thresholding

    threshold   = lam / rho
    Xshrunk     = numpy.zeros_like(X)

    group_shrinkage =   r"""
        for (int i = 0; i < n; i++) {
            // loop over data points
            for (int k = 0; k < m; k++) {
                // loop over codewords
                float scale = 0.0;
                if (Z[(k*n) + i] > threshold) {
                    scale = 1.0 - threshold / Z[(k*n) + i];
                }
                for (int j = 0; scale > 0.0 && j < d; j++) {
                    // loop over coordinates
                    Xshrunk[(k * d + j) * n       + i]  = scale *   X[(k * d + j) * n       +   i];
                    Xshrunk[((k + m) * d + j) * n + i]  = scale *   X[((k + m) * d + j) * n +   i];
                }
            }
        }
    """
    scipy.weave.inline(group_shrinkage, ['n', 'm', 'd', 'threshold', 'X', 'Z', 'Xshrunk'])

    # Apply the soft-thresholding operator
    return Xshrunk


def proj_l2_ball(X, m):
    '''
        Input:      X 2*d*m-by-1 vector  (ndarray) of real and imaginary codewords
                    m >0    number of codewords

        Output:     X where each codeword is projected onto the unit l2 ball
    '''
    # TODO:   2013-03-06 14:00:34 by Brian McFee <brm2132@columbia.edu>
    # weave this function
    d2m     = X.shape[0]
    d       = d2m / (2 * m)

    #         Real part        Imaginary part
    Xnorm   = X[:(d*m)]**2 + X[(d*m):]**2   

    # Group by codewords
    Z = numpy.empty(m)
    for k in xrange(0, d*m, d):
        Z[k/d] = max(1.0, numpy.sum(Xnorm[k:(k+d)])**0.5)
        pass
    
    # Repeat and tile each norm
    Z       = numpy.tile(numpy.repeat(Z, d), (1, 2))

    # Project
    Xp      = numpy.zeros(2 * d * m)
    Xp[:]   = X / Z
    return Xp
#---                            ---#



#--- Encoder                    ---#
def encoder(X, D, reg, max_iter=2000, dynamic_rho=True):
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
    m       = dm / d
    n       = X.shape[1]

    # Initialize split parameter
    Z   = numpy.zeros( (d*m, n) )
    O   = numpy.zeros( (d*m, n) )

    # Initialize augmented lagrangian weight
    rho = 1.0

    # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
    #  sparse multiply ~= hadamard multiply
    # Precompute D'X
    DX  = D.T * X   

    # Precompute dictionary normalization
    # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
    #  sparse multiply ~= hadamard multiply
    Dnorm   = (D * D.T).diagonal()
    # XXX:    2013-03-06 14:48:36 by Brian McFee <brm2132@columbia.edu>
    #  sparse multiply ~= axis-scaling
    Dinv    = scipy.sparse.spdiags( (1.0 + Dnorm / rho)**-1, 0, d, d)

    #--- Regression function        ---#
    def __ridge(_D, _b, _Z):
        '''
        Specialized ridge regression solver for Hadamard products.
    
        Not for external use.
    
        Input:
            A:      2d-by-2dm
            b:      2dm
            Z:      2d > 0,  == diag(inv(I + 1/rho * A * A.T))

        Output:
            X = 1/rho  *  (I - 1/rho * A' * Z * A) * b
        '''
        # FIXME:  2013-03-05 16:26:55 by Brian McFee <brm2132@columbia.edu>
        # profile this on real data: make sure all strides and row/column-majorness is optimal     
    
        # FIXME:  2013-03-06 14:01:12 by Brian McFee <brm2132@columbia.edu>
        # Use a different data packing to get rid of sparsity?
        # or implement our own broadcasty-hadamard product via weave and eliminate sparsity?

        # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
        #  sparse multiply ~= hadamard multiply
        return (_b - (_D.T * (_Z * (_D * _b)) / rho)) / rho
    #---                            ---#

    # ADMM loop
    for t in xrange(max_iter):
        # Encode all the data
        #         FIXME:  2013-03-05 17:28:21 by Brian McFee <brm2132@columbia.edu>
        #         move ridge down here 

        A       = __ridge(D, DX + rho * (Z - O), Dinv)

        # Apply the regularizer
        Zold    = Z
        Z       = reg(A + O, rho)

        # Update residual
        O       = O + A - Z

        #   only compute the rest of this loop every T_CHECKUP iterations 
        if t % T_CHECKUP != 0:
            continue
    
        #  compute stopping criteria
        ERR_primal  = scipy.linalg.norm(A - Z)
        ERR_dual    = rho * scipy.linalg.norm(Z - Zold)

        eps_primal  = A.size**0.5 * ABSTOL + RELTOL * max(scipy.linalg.norm(A), scipy.linalg.norm(Z))
        eps_dual    = O.size**0.5 * ABSTOL + RELTOL * scipy.linalg.norm(O)

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
            # XXX:    2013-03-06 14:48:36 by Brian McFee <brm2132@columbia.edu>
            #  sparse multiply ~= axis-scaling
            Dinv = scipy.sparse.spdiags( (1.0 + Dnorm / rho)**-1.0, 0, d, d)
            pass
        pass
    return Z
#---                            ---#

#--- Dictionary                 ---#
def dictionary(X, A, max_iter=2000, dynamic_rho=True, Dinitial=None):

    (d2, n) = X.shape
    d2m     = A.shape[0]

    d       = d2  / 2
    m       = d2m / d2

    # Initialize ADMM variables
    rho     = 1.0

    D       = numpy.zeros( 2 * d * m )  # Unconstrained codebook
    E       = numpy.zeros_like(D)       # l2-constrained codebook
    W       = numpy.zeros_like(E)       # Scaled dual variables

    if Dinitial is not None:
        E   = columnsToVector(diagsToColumns(Dinitial))[:,0]
        pass

    # Aggregate the scatter and target matrices
    def __aggregator():
        Si      = columnsToDiags(vectorToColumns(A[:,0], m))
        # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
        #  sparse multiply ~= hadamard multiply
        StX     = Si.T * X[:,0]
        # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
        #  sparse multiply ~= hadamard multiply
        StS     = Si.T * Si
        for i in xrange(1, n):
            Si          = columnsToDiags(vectorToColumns(A[:,i], m))
            # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
            #  sparse multiply ~= hadamard multiply
            StX         = StX + Si.T * X[:,i]
            # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
            #  sparse multiply ~= hadamard multiply
            StS         = StS + Si.T * Si
            pass
        return (StS, StX)

    (StS, StX) = __aggregator()

    # We need to solve:
    #   D <- (rho * I + StS) \ (StX + rho * (E - W) )
    #   Use the sparse factorization solver to pre-compute cholesky factors

    SOLVER  = scipy.sparse.linalg.factorized( rho * scipy.sparse.eye(2 * d * m, 2 * d * m) + StS)

    for t in xrange(max_iter):
        # Solve for the unconstrained codebook
        D   = SOLVER( StX + rho * (E - W) )

        # Project each basis element onto the l2 ball
        Eold = E
        E   = proj_l2_ball(D + W, m)

        # Update the residual
        W   = W + D - E

        #   only compute the rest of this loop every T_CHECKUP iterations
        if t % T_CHECKUP != 0:
            continue

        #  compute stopping criteria
        ERR_primal  = scipy.linalg.norm(D - E)
        ERR_dual    = rho * scipy.linalg.norm(E - Eold)

        eps_primal  = (D.size**0.5) * ABSTOL + RELTOL * max(scipy.linalg.norm(D), scipy.linalg.norm(E))
        eps_dual    = (W.size**0.5) * ABSTOL + RELTOL * scipy.linalg.norm(W)
        
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
            SOLVER  = scipy.sparse.linalg.factorized( rho * scipy.sparse.eye(2 * d * m, 2 * d * m) + StS)
            pass
        pass

    # XXX:    2013-03-06 14:49:59 by Brian McFee <brm2132@columbia.edu>
    #  diags construction
    return columnsToDiags(vectorToColumns(E, m))
#---                            ---#

#--- Alternating minimization   ---#
def learn_dictionary(X, m, reg, max_steps=20, max_admm_steps=2000, D=None):
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
    d = d2 / 2

    if D is None:
        # Pick m random columns from the input
        D = X[:, numpy.random.randint(0, n, m)]
        D = normalizeDictionary(columnsToDiags(D))
        pass


    for T in xrange(max_steps):
        # Encode the data
        A = encoder(X, D, reg, max_iter=max_admm_steps)
        print '%2d| A-step MSE=%.3e' % (T, numpy.mean((D * A - X)**2))

        # Optimize the codebook
        D = dictionary(X, A, max_iter=max_admm_steps)
        print '__| D-step MSE=%.3e' %  numpy.mean((D * A - X)**2)

        # Normalize the codebook
        #         D = normalizeDictionary(D)
        pass

    # Re-encode the data with the final codebook
    A = encoder(X, D, reg, max_iter=max_admm_steps)
    return (D, A)
#---                            ---#
