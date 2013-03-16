#!/usr/bin/env python
'''
CREATED:2013-03-01 08:19:26 by Brian McFee <brm2132@columbia.edu>

Convolutional Dictionary Learning

'''

import numpy
import scipy.linalg, scipy.sparse, scipy.sparse.linalg
import scipy.weave
import functools
import multiprocessing as mp

#--- magic numbers              ---#
RHO_MIN     =   1e-4        # Minimum allowed scale for augmenting term rho
RHO_MAX     =   1e4         # Maximum allowed scale for rho
ABSTOL      =   1e-4        # absolute tolerance for convergence criteria
RELTOL      =   1e-3        # relative tolerance
MU          =   3.0         # maximum ratio between primal and dual residuals
TAU         =   2           # scaling for rho when primal/dual exceeds MU
T_CHECKUP   =   10          # number of steps between convergence tests
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
        return Y[:d, :] + 1.j * Y[d:, :]
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
        D[:d, k/d] = Q[range(d), range(k, k + d)]
        D[d:, k/d] = Q[range(d, d2), range(k, k + d)]
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
    A = __sparseDiagonalBlock(D[:d, :])

    # Block the imaginary component
    B = __sparseDiagonalBlock(D[d:, :])

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

    A = numpy.reshape(X[:d, :], (d * m, 1), order='F')
    B = numpy.reshape(X[d:, :], (d * m, 1), order='F')
    return numpy.vstack( (A, B) ).flatten()

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
def reg_l1_real(X, rho, lam, nonneg=False, Xout=None):
    '''
    Input:  X:      matrix of reals
            rho:    augmented lagrangian scaling parameter
            lam:    weight on the regularization term
            nonneg: flag to indicate non-negative l1
            Xout:   destination for the shrunken value

    Output:
            Xout = shrinkage(X, lam / rho)
    Note:
            This routine exists for use within reg_l1_time and reg_l1_space.
            Not to be used directly.
    '''

    WEAVE = True

    if Xout is None:
        # order=A to preserve indexing order of X
        Xout = numpy.empty_like(X, order='A')
        pass

    numel       = X.size

    threshold   = float(lam / rho)

    shrinkage   = r"""
        for (int i = 0; i < numel; i++) {
            if (X[i] - threshold > 0.0) {
                Xout[i]     = X[i] - threshold;
            } else {
                if (X[i] + threshold < 0.0 && nonneg == 0) {
                    Xout[i] = X[i] + threshold;
                } else {
                    Xout[i] = 0.0;
                }
            }
        }
    """
    if WEAVE:
        scipy.weave.inline(shrinkage, ['numel', 'threshold', 'X', 'Xout', 'nonneg'])
    else:
        Xout[:] = (X  > threshold) * (X - threshold)
        if not nonneg:
            Xout[:] = Xout[:] + (X < -threshold) * (X + threshold)
            pass
        pass

    # Apply the soft-thresholding operator
    return Xout

def reg_l1_space(A, rho, lam, width=None, height=None, nonneg=False, Xout=None):
    '''
        Spatial L1 sparsity: assumes each column of X is a columnsToVectord 2d-DFT of a 2d-signal

        Input: 
                A   = 2*d*m-by-n
                rho > 0
                lam > 0
                w, h: d = w * h
                Xout:   destination (must be same shape as A)

    '''

    (d2m, n) = A.shape
    d       = width * height
    m       = d2m / (2 * d)

    if Xout is None:
        Xout = numpy.empty_like(A, order='A')
        pass

    # Reshape activations, transform each one back into image space
    Aspace  = numpy.fft.ifft2(numpy.reshape(real2ToComplex(A), (height, width, m, n), order='F'), axes=(0, 1)).real

    # Apply shrinkage
    # FIXME:  2013-03-11 12:19:56 by Brian McFee <brm2132@columbia.edu>
    # this is some brutal hackery, but weave doesn't like 4-d arrays for some reason...     
    Aspace = Aspace.flatten(order='F')
    reg_l1_real(Aspace, rho, lam, nonneg, Aspace)
    Aspace = Aspace.reshape((height, width, m, n), order='F')

    # Transform back, reshape, and separate real from imaginary
    Xout[:] = complexToReal2(numpy.reshape(numpy.fft.fft2(Aspace, axes=(0, 1)), (height * width * m, n), order='F'))[:]
    return Xout

def reg_l1_complex(X, rho, lam, Xout=None):
    '''
    Input:  X:      2*d*m-by-n      matrix of codeword activations
            rho:    augmented lagrangian scaling parameter
            lam:    weight on the regularization term
            Xout:   destination for the shrunken value

    Output:
            (lam/rho)*Group-l2 shrunken version of X

    Note:
            This function applies shrinkage toward the disk in the complex plane.
            For the standard l1 shrinkage operator, see reg_l1_real.
    '''

    (d2m, n)    = X.shape

    dm          = d2m / 2

    if Xout is None:
        Xout = numpy.empty_like(X, order='A')
        pass


    threshold   = float(lam / rho)

    complex_shrinkage   = r"""
        for (int i = 0; i < n; i++) {
            // iterate over data points

            for (int j = 0; j < dm ; j++) {
                // iterate over activations

                // compute magnitude
                float mag   = sqrt(pow(X[j * n + i], 2) + pow(X[(j + dm) * n + i], 2));
                float scale = (mag < threshold) ? 0.0 : ( 1 - threshold / mag);

                // rescale
                Xout[j * n    + i]  = scale * X[j * n    + i];
                Xout[(j+dm)*n + i]  = scale * X[(j+dm)*n + i];
            }
        }
    """
    scipy.weave.inline(complex_shrinkage, ['n', 'dm', 'threshold', 'X', 'Xout'])

    # Apply the soft-thresholding operator
    return Xout


def reg_l2_group(X, rho, lam, m, Xout=None):
    '''
    Input:  X:      2*d*m-by-n      matrix of codeword activations
            rho:    augmented lagrangian scaling parameter
            lam:    weight on the regularization term
            m:      number of codewords (defines group size)
            Xout:   destination for the shrunken value

    Output:
            (lam/rho)*Group-l2 shrunken version of X
    '''

    (d2m, n)    = X.shape
    dm          = d2m / 2
    d           = int(dm / m)
    
    #   1.  compute sub-vector l2 norms
    #   2.  apply soft-thresholding group-wise

    # Group 2-norm by codeword
    Z           = numpy.empty( (m, n) )

    l2_subvectors = r"""
        for (int i = 0; i < n; i++) {
            // loop over data points

            for (int k = 0; k < m; k++) {
                // loop over codewords

                Z[(k*n) + i] = 0.0;
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

    threshold   = float(lam / rho)

    if Xout is None:
        Xout     = numpy.empty_like(X, order='A')
        pass

    group_shrinkage =   r"""
        for (int i = 0; i < n; i++) {
            // loop over data points
            for (int k = 0; k < m; k++) {
                // loop over codewords
                float scale = 0.0;
                if (Z[(k*n) + i] > threshold) {
                    scale = 1.0 - threshold / Z[(k*n) + i];
                    for (int j = 0; j < d; j++) {
                        // loop over coordinates
                        Xout[(k * d + j) * n       + i]  = scale *   X[(k * d + j) * n       +   i];
                        Xout[((k + m) * d + j) * n + i]  = scale *   X[((k + m) * d + j) * n +   i];
                    }
                } else {
                    for (int j = 0; j < d; j++) {
                        // loop over coordinates
                        Xout[(k * d + j) * n       + i]  = 0.0;
                        Xout[((k + m) * d + j) * n + i]  = 0.0;
                    }
                }
            }
        }
    """
    scipy.weave.inline(group_shrinkage, ['n', 'm', 'd', 'threshold', 'X', 'Z', 'Xout'])

    # Apply the soft-thresholding operator
    return Xout


def reg_lowpass(A, rho, lam, width=None, height=None, Xout=None):
    '''
        Sobel regularization: assumes each column of X is a columnsToVectord 2d-DFT of a 2d-signal

        Input: 
                A   = 2*d*m-by-n
                rho > 0
                lam > 0
                w, h: d = w * h
                Xout:   destination (must be same shape as A)

    '''

    (d2m, n) = A.shape
    d       = width * height
    m       = d2m / (2 * d)

    if Xout is None:
        Xout = numpy.empty_like(A, order='A')
        pass

    # Build the lowpass filter
    lowpass   = numpy.array([ [-1, 0, 1] ]) / 2
#     lowpass   = 1.0 - lowpass
#     lowpass   = (lowpass + lowpass.T) / 2
    H       = numpy.fft.fft2(lowpass, s=(height, width)).reshape((d, 1), order='F')
    H       = numpy.tile(numpy.abs(H), (2 * m, 1))

    S       = (rho / lam) * (1.0 + H**2)**(-1)
    # Invert the filter
    Xout[:] = S * A


    return Xout

def proj_l2_ball(X, m):
    '''
        Input:      X 2*d*m-by-1 vector  (ndarray) of real and imaginary codewords
                    m >0    number of codewords

        Output:     X where each codeword is projected onto the unit l2 ball
    '''
    d2m     = X.shape[0]
    d       = d2m / (2 * m)

    #         Real part        Imaginary part
    Xnorm   = X[:(d*m)]**2 + X[(d*m):]**2   

    # Group by codewords
    Z = numpy.empty(m)
    for k in xrange(m):
        Z[k] = max(1.0, numpy.sum(Xnorm[k*d:(k+1)*d])**0.5)
        pass
    
    # Repeat and tile each norm
    Z       = numpy.tile(numpy.repeat(Z, d), (1, 2)).flatten()

    # Project
    Xp      = numpy.zeros(2 * d * m)
    Xp[:]   = X / Z
    return Xp
#---                            ---#



#--- Encoder                    ---#
def encoder(X, D, reg, max_iter=1000, dynamic_rho=True, output_diagnostics=True):
    '''
    Encoder

    Input:
        X:          2d-by-n     data
        D:          2d-by-2dm   codebook
        reg:        regularization function.

                    Example:
                    reg = functools.partial(CDL.reg_l2_group, lam=0.5, m=num_codewords)

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
    rho = TAU ** -5             # (MAGIC) Encoder rho wants to start small

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

    # diagnostics data
    _DIAG     = {
        'converged':    False,
        'err_primal':   [],
        'err_dual':     [],
        'eps_primal':   [],
        'eps_dual':     [],
        'rho':          []
    }

    # ADMM loop
    for t in xrange(max_iter):
        # Encode all the data
        A       = __ridge(D, DX + rho * (Z - O), Dinv)

        # Apply the regularizer
        Zold    = Z.copy()
        reg(A + O, rho, Xout=Z)

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

        # reporting
        _DIAG['err_primal'].append(ERR_primal)
        _DIAG['err_dual'].append(ERR_dual)
        _DIAG['eps_primal'].append(eps_primal)
        _DIAG['eps_dual'].append(eps_dual)
        _DIAG['rho'].append(rho)
        
        
        if ERR_primal < eps_primal and ERR_dual <= eps_dual:
            _DIAG['converged']  = True
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

    # Append to diagnostics
    _DIAG['err_primal' ]    = numpy.array(_DIAG['err_primal'])
    _DIAG['err_dual' ]      = numpy.array(_DIAG['err_dual'])
    _DIAG['eps_primal' ]    = numpy.array(_DIAG['eps_primal'])
    _DIAG['eps_dual' ]      = numpy.array(_DIAG['eps_dual'])
    _DIAG['rho' ]           = numpy.array(_DIAG['rho'])
    _DIAG['num_steps']      = t

    if output_diagnostics:
        return (Z, _DIAG)
    else:
        return Z

def parallel_encoder(X, D, reg, n_threads=4, max_iter=1000, dynamic_rho=True, output_diagnostics=True):

    n   = X.shape[1]
    dm  = D.shape[1]

    A   = numpy.empty( (dm, n), order='F')

    def __consumer(inQ, out_Q):
        while True:
            try:
                (i, j)          = in_Q.get(True, 1)
                if output_diagnostics:
                    (Aij, diags)    = encoder(X[:,i:j], D, reg, max_iter, dynamic_rho, output_diagnostics)
                else:
                    Aij             = encoder(X[:,i:j], D, reg, max_iter, dynamic_rho, output_diagnostics)
                    diags           = None

                out_Q.put( (i, j, Aij, diags) )
            except:
                break
        out_Q.close()
        pass

    in_Q    = mp.Queue()
    out_Q   = mp.Queue()

    # Build up the input queue
    num_Q   = 0
    B       = n / n_threads
    for i in xrange(0, n, B):
        j = min(n, i + B)
        in_Q.put( (i, j) )
        num_Q += 1
        pass

    # Launch encoders
    for i in range(n_threads):
        mp.Process(target=__consumer, args=(in_Q, out_Q)).start()
        pass

    diagnostics = []
    while num_Q > 0:
        (i, j, Aij, diags) = out_Q.get(True)
        A[:,i:j] = Aij
        diagnostics.append(diags)
        num_Q    -= 1
        pass

    if output_diagnostics:
        return (A, diagnostics)
    else:
        return A
#---                            ---#

#--- Dictionary                 ---#
def dictionary(X, A, max_iter=1000, dynamic_rho=True, Dinitial=None):

    (d2, n) = X.shape
    d2m     = A.shape[0]

    d       = d2  / 2
    m       = d2m / d2

    # Initialize ADMM variables
    rho     = TAU ** 8                  # (MAGIC) Dictionary rho likes to get big

    D       = numpy.zeros( 2 * d * m )  # Unconstrained codebook
    E       = numpy.zeros_like(D, order='A')       # l2-constrained codebook
    W       = numpy.zeros_like(E, order='A')       # Scaled dual variables

    if Dinitial is not None:
        E   = columnsToVector(diagsToColumns(Dinitial))[:, 0]
        pass

    # Aggregate the scatter and target matrices
    def __aggregator():
        Si      = columnsToDiags(vectorToColumns(A[:, 0], m))
        # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
        #  sparse multiply ~= hadamard multiply
        StX     = Si.T * X[:, 0]
        # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
        #  sparse multiply ~= hadamard multiply
        StS     = Si.T * Si
        for i in xrange(1, n):
            Si          = columnsToDiags(vectorToColumns(A[:, i], m))
            # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
            #  sparse multiply ~= hadamard multiply
            StX         = StX + Si.T * X[:, i]
            # XXX:    2013-03-06 14:02:10 by Brian McFee <brm2132@columbia.edu>
            #  sparse multiply ~= hadamard multiply
            StS         = StS + Si.T * Si
            pass
        return (StS / n, StX / n)

    (StS, StX) = __aggregator()

    # We need to solve:
    #   D <- (rho * I + StS) \ (StX + rho * (E - W) )
    #   Use the sparse factorization solver to pre-compute cholesky factors

    SOLVER  = scipy.sparse.linalg.factorized( rho * scipy.sparse.eye(2 * d * m, 2 * d * m) + StS)

    # diagnostics data
    _DIAG     = {
        'converged':    False,
        'err_primal':   [],
        'err_dual':     [],
        'eps_primal':   [],
        'eps_dual':     [],
        'rho':          []
    }

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
        
        # reporting
        _DIAG['err_primal'].append(ERR_primal)
        _DIAG['err_dual'].append(ERR_dual)
        _DIAG['eps_primal'].append(eps_primal)
        _DIAG['eps_dual'].append(eps_dual)
        _DIAG['rho'].append(rho)
        
        if ERR_primal < eps_primal and ERR_dual <= eps_dual:
            _DIAG['converged'] = True
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

    # Numpyfy the diagnostics
    _DIAG['err_primal' ]    = numpy.array(_DIAG['err_primal'])
    _DIAG['err_dual' ]      = numpy.array(_DIAG['err_dual'])
    _DIAG['eps_primal' ]    = numpy.array(_DIAG['eps_primal'])
    _DIAG['eps_dual' ]      = numpy.array(_DIAG['eps_dual'])
    _DIAG['rho' ]           = numpy.array(_DIAG['rho'])
    _DIAG['num_steps']      = t

    # XXX:    2013-03-06 14:49:59 by Brian McFee <brm2132@columbia.edu>
    #  diags construction
    return (columnsToDiags(vectorToColumns(E, m)), _DIAG)
#---                            ---#

#--- Alternating minimization   ---#
def learn_dictionary(X, m, reg='l2_group', lam=1e0, max_steps=20, max_admm_steps=1000, D=None, n_threads=1, **kwargs):
    '''
    Alternating minimization to learn convolutional dictionary

    Input:
        X:              2d-by-n     data matrix, real/imaginary-separated
        m:              number of filters to learn
        reg:            regularizer for activations. One of the following:

                l2_group        l2 norm per activation map (Default)
                l1              l1 norm per (complex) activation map
                l1_space        l1 norm of codeword activations in space domain (2d activations)


        max_steps:      number of outer-loop steps
        max_admm_steps: number of inner loop steps
        D:              initial codebook
        n_threads:      number of parallel encoders to run while training (default: 1)

        kwargs:         Additional keyword arguments to be supplied to regularizer functions

                l1_space:
                    width:      width of the activation patch
                    height:     width of the activation patch
    Output:
        (D, A, encoder, diagnostics) 
        
        where 

        D:          2d-by-m     is the learned dictionary
        A:          2dm-by-n    is the set of activations for X
        encoder:                is the learned encoder function
                        e.g.,   A2 = encoder(X2)
        diagnostics:            is a report of the learning algorithm

    '''

    (d2, n) = X.shape
    d = d2 / 2

    if D is None:
        # Pick m random columns from the input
        D = X[:, numpy.random.randint(0, n, m)]

        #   could be dangerous to normalize if we hit a small column.
        #   random noise should fix it
        D = D + 0.001 * numpy.random.randn(D.shape[0], D.shape[1])**2

        D = normalizeDictionary(columnsToDiags(D))
        pass

    # TODO:   2013-03-08 08:35:57 by Brian McFee <brm2132@columbia.edu>
    #   supervised regularization should be compatible with all other regs
    #   write a wrapper that squashes all offending coefficients to 0, then
    #   calls the specific regularizers
    #   will need to take Y as an auxiliary parameter...

    if reg == 'l2_group':
        g   = functools.partial(reg_l2_group, lam=lam, m=m)
    elif reg == 'l1':
        g   = functools.partial(reg_l1_complex, lam=lam)
    elif reg == 'l1_space':
        g   = functools.partial(reg_l1_space, lam=lam, **kwargs)
    elif reg == 'lowpass':
        g   = functools.partial(reg_lowpass, lam=lam, **kwargs)
    else:
        raise ValueError('Unknown regularization: %s' % reg)

    # Reset the diagnostics output
    diagnostics   = {
        'encoder':          [],
        'dictionary':       [],
        'parameters':       {
            'n':                X.shape[1],
            'd':                X.shape[0] / 2,
            'm':                m,
            'reg':              reg,
            'lam':              lam,
            'max_steps':        max_steps,
            'max_admm_steps':   max_admm_steps,
            'auxiliary':        kwargs
        },
        'globals':  {
            'rho_min':      RHO_MIN,
            'rho_max':      RHO_MAX,
            'abs_tol':      ABSTOL,
            'rel_tol':      RELTOL,
            'mu':           MU,
            'tau':          TAU,
            't_checkup':    T_CHECKUP
        }
    }

    Xnorm = numpy.mean(X ** 2)

    error = []
    for T in xrange(max_steps):
        # Encode the data
        if n_threads == 1:
            (A, A_diagnostics) = encoder(X, D, g, max_iter=max_admm_steps)
        else:
            (A, A_diagnostics) = parallel_encoder(X, D, g, n_threads=n_threads, max_iter=max_admm_steps)
            pass
        diagnostics['encoder'].append(A_diagnostics)
        
        error.append(numpy.mean((D * A - X)**2))
        SNR = 10 * (numpy.log10(Xnorm) - numpy.log10(error[-1]))
        print '%2d| A-step MSE=%.3e   | SNR=%3.2fdB' % (T, error[-1], SNR)

        # Optimize the codebook
        (D, D_diagnostics) = dictionary(X, A, max_iter=max_admm_steps)
        diagnostics['dictionary'].append(D_diagnostics)

        error.append(numpy.mean((D * A - X)**2))
        SNR = 10 * (numpy.log10(Xnorm) - numpy.log10(error[-1]))
        print '__| D-step MSE=%.3e   | SNR=%3.2fdB' %  (error[-1], SNR)
        pass

    diagnostics['error']    = numpy.array(error)

    # Re-encode the data with the final codebook
    if n_threads > 1:
        (A, A_diagnostics) = parallel_encoder(X, D, g, n_threads=n_threads, max_iter=max_admm_steps)
    else:
        (A, A_diagnostics) = encoder(X, D, g, max_iter=max_admm_steps)
        pass
    diagnostics['final_encoder'] = A_diagnostics
    
    # Package up the learned encoder function for future use
    if n_threads > 1:
        my_encoder  = functools.partial(parallel_encoder, n_threads=n_threads, D=D, reg=g, max_iter=max_admm_steps, output_diagnostics=False)
    else:
        my_encoder  = functools.partial(encoder, D=D, reg=g, max_iter=max_admm_steps, output_diagnostics=False)
        pass

    return (D, A, my_encoder, diagnostics)
#---                            ---#
