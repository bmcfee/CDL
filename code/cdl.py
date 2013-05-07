#!/usr/bin/env python
"""Convolutional Dictionary Learning

CREATED:2013-03-01 08:19:26 by Brian McFee <brm2132@columbia.edu>
"""

import functools
import multiprocessing as mp

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.weave

#--- magic numbers              ---#
#  it is of utmost importance that these numbers be floats

RHO_INIT_A  =   2e-3        # Initial value for rho (encoder)
RHO_INIT_D  =   2e-1        # Initial value for rho (dictionary)
RHO_MIN     =   1e-6        # Minimum allowed scale for augmenting term rho
RHO_MAX     =   1e6         # Maximum allowed scale for rho
ABSTOL      =   1e-4        # absolute tolerance for convergence criteria
RELTOL      =   1e-3        # relative tolerance
MU          =   1e1         # maximum ratio between primal and dual residuals
TAU         =   2e0         # scaling for rho when primal/dual exceeds MU
T_CHECKUP   =   5           # number of steps between convergence tests
BETA        =   1e0         # decay factor for mini-batch learning
#---                            ---#

#--- Utility functions          ---#

def patches_to_vectors(patches, pad_data=False):
    """Convert a stack of patches into their 2D fourier transforms.

    This function takes a stack of 2-D patches and converts them into the 
    format used by the encoding algorithm.

    The steps are as follows:

    1. 2D-DFT each patch
    2. Vectorize the transformed patches
    3. Stack vectors into columns
    4. Separate real and imaginary components

    See also: vectors_to_patches()

    Arguments:
      patches   --  (ndarray)   height-by-width-by-n data matrix
      pad_data  --  (boolean)   pad the DFT?                |default: False
    
    Returns X:
      vectors   --  (ndarray)   (2*height*width*?)-by-n matrix
                                If padding, the matrix is 4 times as tall.

    """
    
    (height, width, num_samples) = patches.shape

    if pad_data:
        x_shape = (2 * height, 2 * width)
    else:
        x_shape = (height, width)

    patch_dft   = np.fft.fft2(patches, s=x_shape, axes=(0, 1))
    return complex_to_real2(patch_dft.reshape( (np.prod(x_shape), 
                                                num_samples), order='F'))

def vectors_to_patches(vectors, width, pad_data=False, real=True):
    """Convert a matrix of real-imag separated columns into a stack of patches

    1. Combine real+imag components
    2. Reshape into a 3d-stack
    3. Inverse DFT each frame in the stack
    4. Undo padding effects

    See also: patches_to_vectors()

    Arguments:
      vectors   -- (ndarray) 
      width     -- (int>0)      width of the patches
      pad_data  -- (boolean)    is the data padded?         | default: False
      real      -- (boolean)    force output to be real     | default: True

    Returns:
      patches   -- (ndarray)    height-by-width-by-n real-valued

    """

    # First, convert to complex
    vectors = real2_to_complex(vectors)

    (size, num_samples) = vectors.shape

    # Reshape into patches
    if pad_data:
        height = size / (4 * width)
        vectors = vectors.reshape((2 * height, 2 * width, num_samples), 
                                  order='F')
    else:
        height = size / width
        vectors = vectors.reshape((height, width, num_samples), 
                                  order='F')

    # Inverse DFT and truncate
    patches = np.fft.ifft2(vectors, axes=(0, 1))[:height, :width, :]

    if real:
        patches = patches.real

    return patches


def complex_to_real2(x_complex):
    """Separate the real and imaginary components of a matrix

    See also: real2_to_complex()

    Arguments:
        x_complex   -- (ndarray)    complex d-by-n matrix

    Returns:
        x_real2     -- (ndarray)    [real(x_complex) ; imag(x_complex)]

    """
    return np.vstack((x_complex.real, x_complex.imag))

def real2_to_complex(x_real2):
    """Combine the real and imaginary components of a matrix

    See also: complex_to_real2()

    Arguments:
        x_real2     --  (ndarray)   real 2d-by-n from complex_to_real2()

    Returns X:
        x_complex   --  (ndarray)   complex d-by-n

    """
    d_imag = x_real2.shape[0] / 2

    if x_real2.ndim > 1:
        return x_real2[:d_imag, :] + 1.j * x_real2[d_imag:, :]
    else:
        return x_real2[:d_imag] + 1.j * x_real2[d_imag:]


def diags_to_columns(Q):
    """Convert a sparse, diagonal block matrix into a dense column matrix
    
    See also: columns_to_diags()

    Arguments:  
        Q       -- (scipy.sparse)   2d-by-2dm with nonzeros along diagonals

    Returns: 
        D       -- (ndarray)        2d-by-m dense matrix D of diagonals 
                                    from the upper and lower block of Q

    """
    # Q = [A, -B ; B A]
    # cut to the first half of columns
    # then break vertically

    (d2, d2m)  = Q.shape

    d   = d2    / 2
    dm  = d2m   / 2
    m   = dm    / d
    
    D = np.empty( (d2, m) )

    for k in xrange(0, d * m, d):
        D[:d, k/d] = Q[range(d), range(k, k + d)]
        D[d:, k/d] = Q[range(d, d2), range(k, k + d)]

    return D

def columns_to_diags(D):
    """Convert a dense column matrix into a sparse, diagonal-block matrix.
    
    See also: diags_to_columns()

    Arguments:
        D   -- (ndarray)        2d-by-m matrix of real+imaginary vectors

    Returns:
        Q   -- (scipy.sparse)   2d-by-2dm sparse diagonal block matrix 
                                Q = [A, -B ; B, A]
                                where A and B are derived from the real and 
                                imaginary components (top and bottom halves)
                                of D.

    """

    def _sparse_dblock(matrix):
        """Rearrange a d-by-m matrix D into a sparse d-by-dm matrix Q
        The i'th d-by-d block of Q = diag(D[:, i])

        """

        return scipy.sparse.spdiags(matrix.T, 
                                    range(0, - matrix.size, - matrix.shape[0]), 
                                    matrix.size, matrix.shape[0]).T

    # Get the size of each codeword
    d = D.shape[0] / 2

    # Block the real component
    A = _sparse_dblock(D[:d, :])

    # Block the imaginary component
    B = _sparse_dblock(D[d:, :])

    # Block up everything in csr format
    return scipy.sparse.bmat([ [ A, -B], [B, A] ], format='csr')

def columns_to_vector(X):
    """Vectorize a stacked, real+imag matrix.
    
    See also: vector_to_columns()

    Arguments:  
        X   --  (ndarray)   2d-by-m matrix of stacked real+imag columns
                            

    Returns: 
        Y   --  (ndarray)   2dm-by-1 vector
                            If  X = [A ; B], then 
                                Y = [vec(A) ; vec(B)]

    Note: this stacking places all imaginary parts below all real parts.
    """

    (d2, m) = X.shape

    d = d2 / 2

    A = np.reshape(X[:d, :], (d * m, 1), order='F')
    B = np.reshape(X[d:, :], (d * m, 1), order='F')
    return np.vstack( (A, B) ).flatten()

def vector_to_columns(AB, m):
    """Unstack a column vector into a real+imag matrix
    
    See also: columns_to_vector()

    Arguments:  
        AB      --  (ndarray)   2dm-by-1 stacked column vector
        m       --  (int>0)     number of columns to produce

    Returns: 
        X       --  (ndarray)   2d-by-m array where each column
                                is stacked real+imag components

    """

    d2m = AB.shape[0]

    d = d2m / (2 * m)

    A = np.reshape(AB[:(d*m)], (d, m), order='F')
    B = np.reshape(AB[(d*m):], (d, m), order='F')

    return np.vstack( (A, B) )

def normalize_dictionary(D):
    """Normalize a dictionary to have all unit-length bases.

    Arguments:
        D       -- (sparse) 2d-by-2dm diagonal-block dictionary
                            See: columns_to_diags()

    Returns:
        Dhat    -- (sparse) D with each codeword normalized to unit length

    """
    D = diags_to_columns(D)
    D = D / (np.sum(D**2, axis=0) ** 0.5)
    D = columns_to_diags(D)
    return D


def pool(A, depth_h=0, depth_w=0, agg=np.sum, absval=True):
    """Spatial pyramid pooling

    Arguments:
        A           -- (ndarray)    height-by-width activation patch
        depth_h     -- (int>=0)     # vertical splits
        depth_w     -- (int>=0)     # horizontal splits
        agg         -- (function)   aggregator function
        absval      -- (boolean)    apply abs() before pooling

    Returns:
        P           -- (ndarray)    vector of activation energy aggregated 
                                    at multiple tree levels

    Notes:
        The dimensions of A are assumed to be powers of 2
        Output vector has dimension 
            ((2^(1+depth_h) - 1) * (2^(1+depth_w) - 1))
    """

    (height, width) = A.shape

    if absval:
        A = np.abs(A)

    P = []
    for h_level in range(1 + depth_h):
        step_h = int(height * 2.0 **(-h_level))

        for w_level in range(1 + depth_w):
            step_w = int(width * 2.0 **(-w_level))

            for u in range(0, height, step_h):
                for v in range(0, width, step_w):
                    P.append(agg(A[u:u+step_h, v:v+step_w]))

    return np.array(P)
#---                            ---#



#--- Codebook initialization    ---#
def init_svd(X, m):
    """Initializes a dictionary with (approximate) left-singular vectors

    Arguments:
        X   --  (ndarray)   2d-by-n data array
        m   --  (int>0)     number of basis elements to initialize

    Returns:
        D   --  (sparse)    2d-by-2dm normalized dictionary

    Note: the SVD is computed from a subsample of max(n, m**2) columns from X

    """
    # Draw a subsample
    n           = X.shape[1]
    n_sample    = min(n, m**2)
    Xsamp       = X[:, np.random.randint(0, n, n_sample)]
    U           = scipy.linalg.svd(Xsamp)[0]
    return normalize_dictionary(columns_to_diags(U[:, :m]))

def init_columns(X, m):
    """Initializes a dictionary with random columns of the data

    Arguments:
        X   -- (ndarray)    2d-by-n data array
        m   -- (int>0)      number of basis elements to initialize

    Returns:
        D   -- (sparse) 2d-by-2dm normalized dictionary

    """

    n           = X.shape[1]

    Xsamp       = X[:, np.random.randint(0, n, m)]
    return normalize_dictionary(columns_to_diags(Xsamp))
#---                            ---#

#--- Regularization functions   ---#
def reg_l1_real(X, rho, alpha, nonneg=False, Xout=None):
    """l1 regularization
    
    Arguments:  
      X         --  (ndarray)   array of reals
      rho       --  (float>0)   augmented lagrangian scaling parameter
      alpha     --  (float>0)   weight on the regularization term
      nonneg    --  (boolean)   force non-negativity            |default: False
      Xout      --  (ndarray)   (optional)  destination for the shrunken value

    Returns:
      Xout      --  (ndarray)   shrinkage(X, alpha / rho)

    Note: This routine exists for use within reg_l1_time and reg_l1_space.
          Not to be used directly.
    """

    if Xout is None:
        # order=A to preserve indexing order of X
        Xout = np.empty_like(X, order='A')

    numel       = X.size

    threshold   = float(alpha / rho)

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
    scipy.weave.inline( shrinkage, 
                        ['numel', 'threshold', 'X', 'Xout', 'nonneg'])

    # Apply the soft-thresholding operator
    return Xout

def reg_l1_space(A, rho, alpha, height=None, width=None, nonneg=False, 
                 pad_data=False, Xout=None):
    """Spatial L1 sparsity: 
        - inverse 2d-DFT to A
        - l1 shrinkage
        - 2d-DFT back
    
    Each column of A is a columns_to_vector'd 2d-DFT of an activation matrix

    Arguments: 
      A         --  (ndarray)   2dm-by-n matrix of codeword activations
      rho       --  (float>0)   augmented lagrangian scaling parameter
      alpha     --  (float>0)   weight on the regularization term
      width     --  (int)       d = width * height specifies the patch shape
      height    --  (int)       
      nonneg    --  (boolean)   force non-negativity        |default: False
      pad_data  --  (boolean)   is the input padded         |default: False
      Xout      --  (ndarray)   (optional) output destination

    Returns:
      Xout      --  (ndarray)   fft2(shrinkage(ifft2(A), alpha/rho))

    """

    if pad_data:
        # If we have a padded FFT, then width and height should double
        width   = 2 * width
        height  = 2 * height

    (d2m, n)    = A.shape
    d           = width * height
    m           = d2m / (2 * d)

    if Xout is None:
        Xout    = np.empty_like(A, order='A')

    # Reshape activations, transform each one back into image space
    Aspace      = np.fft.ifft2(np.reshape(real2_to_complex(A), 
                                          (height, width, m, n), 
                                          order='F'), 
                               axes=(0, 1)).real

    # Apply shrinkage
    # FIXME:  2013-03-11 12:19:56 by Brian McFee <brm2132@columbia.edu>
    # this is some brutal hackery, but weave doesn't like 4-d arrays 
    Aspace = Aspace.flatten(order='F')

    reg_l1_real(Aspace, rho, alpha, nonneg, Aspace)
    
    Aspace = Aspace.reshape((height, width, m, n), order='F')

    if pad_data:
        # In a padded FFT, threshold out all out-of-scope activations
        Aspace[(height/2):, (width/2):, :, :] = 0.0

    # Transform back, reshape, and separate real from imaginary
    Xout[:] = complex_to_real2(np.reshape(np.fft.fft2(Aspace, axes=(0, 1)), 
                                          (height * width * m, n), 
                                          order='F'))[:]
    return Xout

def reg_l1_complex(X, rho, alpha, Xout=None):
    """Complex l1 regularization

    Arguments:  
      X     --  (ndarray) 2*d*m-by-n matrix of codeword activations
      rho   --  (float>0) augmented lagrangian scaling parameter
      alpha --  (float>0) weight on the regularization term
      Xout  --  (ndarray) optional destination for the shrunken value

    Returns:
      Xout  --  (ndarray)   (alpha/rho)*Group-l2 shrunken version of X

    Note:
        This function applies shrinkage toward the disk in the complex plane.
        For the standard l1 shrinkage operator, see reg_l1_real.
    """

    (d2m, n)    = X.shape

    dm          = d2m / 2

    if Xout is None:
        Xout = np.empty_like(X, order='A')


    threshold   = float(alpha / rho)

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


def reg_l2_group(X, rho, alpha, m, Xout=None):
    """Group-l2 regularization

    Arguments:  
      X     -- (ndarray) 2*d*m-by-n  matrix of codeword activations
      rho   -- (float>0) augmented lagrangian scaling parameter
      alpha -- (float>0) weight on the regularization term
      m     -- (int>0)   number of codewords (defines group size)
      Xout  -- (ndarray) optional destination for the shrunken value

    Returns:
      Xout  -- (ndarray) (alpha/rho)*Group-l2 shrunken version of X

    """

    (d2m, n)    = X.shape
    dm          = d2m / 2
    d           = int(dm / m)
    
    #   1.  compute sub-vector l2 norms
    #   2.  apply soft-thresholding group-wise

    # Group 2-norm by codeword
    Z           = np.empty( (m, n) )

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

    threshold   = float(alpha / rho)

    if Xout is None:
        Xout     = np.empty_like(X, order='A')

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

    scipy.weave.inline(group_shrinkage, 
                       ['n', 'm', 'd', 'threshold', 'X', 'Z', 'Xout'])

    # Apply the soft-thresholding operator
    return Xout


def reg_lowpass(A, rho, alpha, height=None, width=None, Xout=None):
    """Lowpass regularization: penalizes square of first-derivative
    
    Arguments:
      A         --  (ndarray) 2*d*m-by-n activation matrix
      rho       --  (float>0) augmented lagrangian scaling parameter
      alpha     --  (float>0) weight on the regularization term
      height    --  (int>0)
      width     --  (int>0)   d = w * h specifies the patch shape
      Xout      --  (ndarray) optional output destination

    Returns:
      Xout      --  smoothed version of A

    """

    #     FIXME:  2013-04-01 11:55:08 by Brian McFee <brm2132@columbia.edu>
    # does not support fftpad 

    d2m     = A.shape[0]
    d       = width * height
    m       = d2m / (2 * d)

    if Xout is None:
        Xout = np.empty_like(A, order='A')

    # Build the lowpass filter
    lowpass  = np.array([ [-1, 0, 1] ]) / 2
    H   = np.fft.fft2(lowpass, s=(height, width)).reshape((d, 1), order='F')
    H   = np.tile(np.abs(H), (2 * m, 1))

    S   = (rho / alpha) * (1.0 + H**2)**(-1)

    # Invert the filter
    Xout[:] = S * A

    return Xout

def proj_l2_ball(X, m):
    """Project a vectorized matrix onto the unit l2 ball
    
    Arguments:  
      X     -- (ndarray)  2*d*m-by-1 vector of real+imaginary codewords
      m     -- (int>0)    number of codewords

    Output:
      Xhat  -- (ndarray)  Each sub-vector of X is projected onto the unit l2 ball

    """
    d2m     = X.shape[0]
    d       = d2m / (2 * m)

    #         Real part        Imaginary part
    Xnorm   = X[:(d*m)]**2 + X[(d*m):]**2   

    # Group by codewords
    Z = np.empty(m)
    for k in xrange(m):
        Z[k] = max(1.0, np.sum(Xnorm[k*d:(k+1)*d])**0.5)
    
    # Repeat and tile each norm
    Z       = np.tile(np.repeat(Z, d), (1, 2)).flatten()

    # Project
    Xp      = np.zeros(2 * d * m)
    Xp[:]   = X / Z
    return Xp
#---                            ---#



#--- Encoder                    ---#
def _encoder(X, D, reg, max_iter=200, output_diagnostics=True):
    """Encoder (single-threaded).  For internal use only.

    Arguments:
      X         --  (ndarray)   2d-by-n     data matrix
      D         --  (sparse)    2d-by-2dm   dictionary
      reg       --  (function)  regularization function, e.g.,

                    reg = functools.partial(CDL.reg_l2_group, alpha=0.5, m=num_codewords)

      max_iter  --  (int>0)     maximum number of steps     |default: 200

    Returns:
        A       --  (ndarray)   2dm-by-n activation matrix
                                where X ~= DA
    """

    (d, dm) = D.shape
    m       = dm / d
    n       = X.shape[1]

    # Initialize split parameter
    Z   = np.zeros( (d*m, n) )
    O   = np.zeros( (d*m, n) )

    # Initialize augmented lagrangian weight
    rho = RHO_INIT_A

    # Precompute D'X
    DX  = D.T * X   

    # Precompute dictionary normalization
    Dnorm   = (D * D.T).diagonal()
    Dinv    = scipy.sparse.spdiags( (1.0 + Dnorm / rho)**-1, 0, d, d)

    #--- Regression function        ---#
    def _ridge(_D, _b, _Z):
        """Specialized ridge regression solver for Hadamard products.
    
        Not for external use.
    
        Arguments:
          _D    -- (sparse)     2d-by-2dm
          _b    -- (ndarray)    2dm
          _Z    -- (ndarray)    2d > 0,  == diag(inv(I + 1/rho * A * A.T))

        Returns:
            X   -- (ndarray)    1/rho  *  (I - 1/rho * A' * Z * A) * b

        """
    
        return (_b - (_D.T * (_Z * (_D * _b)) / rho)) / rho
    #---                            ---#

    # diagnostics data
    _DIAG     = {
        'converged' :   False,
        'err_primal':   [],
        'err_dual'  :   [],
        'eps_primal':   [],
        'eps_dual'  :   [],
        'rho'       :   []
    }

    # ADMM loop
    t = 0
    for t in xrange(max_iter):
        # Encode all the data
        A       = _ridge(D, DX + rho * (Z - O), Dinv)

        # Apply the regularizer
        Zold    = Z.copy()
        reg(A + O, rho, Xout=Z)

        # Update residual
        O       = O + A - Z

        #   only compute the rest of this loop every T_CHECKUP iterations 
        if t % T_CHECKUP != 0:
            continue
    
        #  compute stopping criteria
        ERR_primal = scipy.linalg.norm(A - Z)
        ERR_dual   = rho * scipy.linalg.norm(Z - Zold)

        eps_primal = A.size**0.5 * ABSTOL + RELTOL * max(scipy.linalg.norm(A), 
                                                          scipy.linalg.norm(Z))

        eps_dual   = O.size**0.5 * ABSTOL + RELTOL * scipy.linalg.norm(O)

        # reporting
        _DIAG['err_primal'  ].append(ERR_primal)
        _DIAG['err_dual'    ].append(ERR_dual)
        _DIAG['eps_primal'  ].append(eps_primal)
        _DIAG['eps_dual'    ].append(eps_dual)
        _DIAG['rho'         ].append(rho)
        
        if ERR_primal < eps_primal and ERR_dual <= eps_dual:
            _DIAG['converged']  = True
            break

        if ERR_primal > MU * ERR_dual and rho * TAU < RHO_MAX:
            # Too much weight on primal, upscale dual
            rho         = rho   * TAU
            O           = O     / TAU
        elif ERR_dual > MU * ERR_primal and rho / TAU > RHO_MIN:
            # Too much weight on dual, upscale primal
            rho         = rho   / TAU
            O           = O     * TAU
        else:
            # Primal and dual are balanced, no need to rescale
            continue

        # Update Dinv
        Dinv = scipy.sparse.spdiags( (1.0 + Dnorm / rho)**-1.0, 0, d, d)


    # Append to diagnostics
    _DIAG['err_primal' ]    = np.array(_DIAG['err_primal'])
    _DIAG['err_dual' ]      = np.array(_DIAG['err_dual'])
    _DIAG['eps_primal' ]    = np.array(_DIAG['eps_primal'])
    _DIAG['eps_dual' ]      = np.array(_DIAG['eps_dual'])
    _DIAG['rho' ]           = np.array(_DIAG['rho'])
    _DIAG['num_steps']      = t

    if output_diagnostics:
        return (Z, _DIAG)
    else:
        return Z


def parallel_encoder(X, D, reg, n_threads=4, 
                                max_iter=200, 
                                output_diagnostics=False):
    """Parallel encoder

    Arguments:
      X         -- (ndarray)    2d-by-n data to be encoded
      D         -- (sparse)     2d-by-2dm dictionary
      reg       -- (function)   regularization function, eg:
                    
                    reg = functools.partial(CDL.reg_l2_group, alpha=0.5, m=num_codewords)

      n_threads -- (int>0)      number of parallel threads      | default: 4
      max_iter  -- (int>0)      maximum number of steps         | default: 200

    Returns:
        A       -- (ndarray)    2dm-by-n activation matrix,  X ~= D*A

    """
    n   = X.shape[1]
    dm  = D.shape[1]

    step = n / n_threads

    # Punt to the local encoder if we're single-threaded or don't have enough 
    # data to distrubute
    if n_threads == 1 or step == 0:
        return _encoder(X, D, reg, 
                         max_iter=max_iter, 
                         output_diagnostics=output_diagnostics)

    A   = np.empty( (dm, n), order='F')

    def _consumer(input_queue, output_queue):
        while not input_queue.empty():
            (i, j) = input_queue.get(True, 1)

            if output_diagnostics:
                (a_ij, diag)   = _encoder(X[:, i:j], D, reg, 
                                            max_iter=max_iter, 
                                            output_diagnostics=True)
            else:
                a_ij = _encoder(X[:, i:j], D, reg, 
                                max_iter=max_iter, 
                                output_diagnostics=False)
                diag = None

            output_queue.put((i, j, a_ij, diag))

        output_queue.close()
        output_queue.join_thread()


    input_queue    = mp.Queue()
    output_queue   = mp.Queue()

    # Build up the input queue
    num_queue = 0
    for i in xrange(0, n, step):
        input_queue.put( (i, min(n, i + step)) )
        num_queue += 1

    # Launch encoders
    for i in range(n_threads):
        mp.Process(target=_consumer, args=(input_queue, output_queue)).start()

    # We're done writing
    input_queue.close()

    diagnostics = []
    while num_queue > 0:
        (i, j, a_ij, diags) = output_queue.get(True)
        A[:, i:j]   = a_ij
        diagnostics.append(diags)
        num_queue   -= 1

    if output_diagnostics:
        return (A, diagnostics)
    
    return A
#---                            ---#

#--- Dictionary                 ---#
def _encoding_statistics(A, X):
    """Compute the empirical average of encoding statistics.

    This function is only used by the dictionary learning algorithm.

    It has been factored out of dictionary() to facilitate online
    mini-batch learning.

    Arguments:
      A     -- (ndarray)    2dm-by-n    activation matrix
      X     -- (ndarray)    2d-by-n     data matrix

    Returns (StS, StX):
      StS   -- (sparse)     1/n sum_i A[:,i].T * A[:,i]  in diagonal-block form
      StX   -- (ndarray)    1/n sum_i A[:,i].T * X[:,i]  in dense form

    """

    n = A.shape[1]
    m = A.shape[0] / X.shape[0]

    Si      = columns_to_diags(vector_to_columns(A[:, 0], m))
    StX     = Si.T * X[:, 0]
    StS     = Si.T * Si
    
    for i in xrange(1, n):
        Si          = columns_to_diags(vector_to_columns(A[:, i], m))

        StS         = StS + Si.T * Si
        StX         = StX + Si.T * X[:, i]

    return (StS / n, StX / n)


def dictionary(StS, StX, m, max_iter=200, Dinitial=None):
    """Learn a dictionary from encoding statistics.

    Arguments:
      StS       --  (sparse)    matrix of activation cross-interactions
      StX       --  (ndarray)   matrix is activation-data interactions

      m         --  (int>0)     size of the dictionary
      max_iter  --  (int>0)     maximum number of steps         | default: 200
      Dinitial  --  (ndarray)   2dm-by-1 initial dictionary     | default: None
    
    Returns D:
      D         --  (sparse)    2d-by-2dm diagonal-block dictionary

    Note: StS and StX are produced by _encoding_statistics()
    """
    d2m     = StX.shape[0]

    # Initialize ADMM variables
    rho     = RHO_INIT_D

    D       = np.zeros( d2m, order='F' )         # Unconstrained dictionary
    E       = np.zeros_like(D, order='A')        # l2-constrained dictionary
    W       = np.zeros_like(E, order='A')        # Scaled dual variables

    if Dinitial is not None:
        E[:]    = columns_to_vector(diags_to_columns(Dinitial))

    # We need to solve:
    #   D <- (rho * I + StS) \ (StX + rho * (E - W) )
    #   Use the sparse factorization solver to pre-compute cholesky factors

    ident = scipy.sparse.eye(d2m, d2m)
    SOLVER  = scipy.sparse.linalg.factorized(StS + rho * ident)

    # diagnostics data
    _DIAG     = {
        'converged' :   False,
        'err_primal':   [],
        'err_dual'  :   [],
        'eps_primal':   [],
        'eps_dual'  :   [],
        'rho'       :   []
    }

    t = 0
    for t in xrange(max_iter):
        # Solve for the unconstrained dictionary
        D       = SOLVER( StX + rho * (E - W) )

        # Project each basis element onto the l2 ball
        Eold    = E
        E       = proj_l2_ball(D + W, m)

        # Update the residual
        W       = W + D - E

        #   only compute the rest of this loop every T_CHECKUP iterations
        if t % T_CHECKUP != 0:
            continue

        #  compute stopping criteria
        ERR_primal = scipy.linalg.norm(D - E)
        ERR_dual   = rho * scipy.linalg.norm(E - Eold)

        eps_primal = D.size**0.5 * ABSTOL + RELTOL * max(scipy.linalg.norm(D), 
                                                         scipy.linalg.norm(E))

        eps_dual   = W.size**0.5 * ABSTOL + RELTOL * scipy.linalg.norm(W)
        
        # reporting
        _DIAG['err_primal'  ].append(ERR_primal)
        _DIAG['err_dual'    ].append(ERR_dual)
        _DIAG['eps_primal'  ].append(eps_primal)
        _DIAG['eps_dual'    ].append(eps_dual)
        _DIAG['rho'         ].append(rho)
        
        if ERR_primal < eps_primal and ERR_dual <= eps_dual:
            _DIAG['converged'] = True
            break

        if ERR_primal > MU * ERR_dual and rho * TAU < RHO_MAX:
            # Too much weight on primal, upscale dual
            rho = rho   * TAU
            W   = W     / TAU
        elif ERR_dual > MU * ERR_primal and rho / TAU > RHO_MIN:
            # Too much weight on dual, upscale primal
            rho = rho   / TAU
            W   = W     * TAU
        else:
            # Primal and dual are balanced, no need to rescale
            continue

        # Update the solver with the new rho value
        SOLVER = scipy.sparse.linalg.factorized(StS + rho * ident)


    # Numpyfy the diagnostics
    _DIAG['err_primal' ]    = np.array(_DIAG['err_primal'])
    _DIAG['err_dual' ]      = np.array(_DIAG['err_dual'])
    _DIAG['eps_primal' ]    = np.array(_DIAG['eps_primal'])
    _DIAG['eps_dual' ]      = np.array(_DIAG['eps_dual'])
    _DIAG['rho' ]           = np.array(_DIAG['rho'])
    _DIAG['num_steps']      = t

    return (columns_to_diags(vector_to_columns(E, m)), _DIAG)
#---                            ---#

#--- Alternating minimization   ---#
def _batches(X, batch_size, max_steps, shuffle):
    """Mini-batch generator

    Arguments:
      X             --  (ndarray)   2d-by-n data matrix
      batch_size    --  (int>0)     number of points per btach
      max_steps     --  (int>0)     maximum number of batches to yield
      shuffle       --  (bool)      shuffle between batches
    Yields:
      Xbatch        --  (ndarray)   2d-by-batch_size random column sample of X

    Note: if batch_size == n, then Xbatch = X is always yielded

    Note: per-batch sampling is done with replacement
    """
    n = X.shape[1]

    for _ in xrange(max_steps):
        # Sample a random subset of columns (with replacement)
        if batch_size == n:
            yield X
        else:
            yield X[:, np.random.randint(0, n, batch_size)]


def learn_dictionary(X, m,  reg='l1_space', 
                            alpha=1e-1, 
                            max_steps=20, 
                            max_admm_steps=200, 
                            n_threads=1, 
                            batch_size=64, 
                            verbose=False,
                            shuffle=True,
                            **kwargs):
    """Alternating minimization to learn a convolutional dictionary

    Arguments:
      X             -- (ndarray) 2d-by-n.  Each column is the fourier transform
                                           of an example point, and the 
                                           real//imaginary components have been
                                           separated by complex_to_real2().

      m             -- (int>0)  number of filters to learn
      reg           -- (string) regularizer for activations. One of the following:

        'l1_space'  --  l1 of (2-dimensional) activations       | default
        'l1'        --  l1 norm per (complex) activation map
        'l2_group'  --  l2 norm per activation map 

      max_steps     -- (int>0)  number of alternating minimization steps
      max_admm_steps-- (int>0)  number of steps for the internal optimizer
      n_threads     -- (int>0)  number of encoder threads       | default: 1
      batch_size    -- (int>0)  number of points per step       | default: 64
                                specify None to use all of X
      shuffle       -- (bool)   shuffle data between batches
      verbose       -- (bool)   show training progress

      **kwargs      --  Additional keyword arguments to regularizers:

        For l1_space:
            height  -- (int>0)    patch height: d = width * height
            width   -- (int>0)    patch width
            pad_data -- (boolean)  are the patches zero-padded?  | default: False

    Returns (encode, D, diagnostics):
        
      encode        -- (function) the learned encoder function. 
                                  To encode new data, say:
                                    A_test = encode(X_test)

      D             -- (ndarray)  2d-by-m  the learned dictionary (dense form)
      diagnostics   -- (dict)     report of the learning algorithm

    """

    n = X.shape[1]


    # TODO:   2013-03-08 08:35:57 by Brian McFee <brm2132@columbia.edu>
    #   supervised regularization should be compatible with all other regs
    #   write a wrapper that squashes all offending coefficients to 0, then
    #   calls the specific regularizers
    #   will need to take Y as an auxiliary parameter...

    ###
    # Configure the encoding regularizer
    if reg == 'l2_group':
        regularize  = functools.partial(    reg_l2_group,   alpha=alpha, m=m)

    elif reg == 'l1':
        regularize  = functools.partial(    reg_l1_complex, alpha=alpha)

    elif reg == 'l1_space':
        regularize  = functools.partial(    reg_l1_space,   alpha=alpha, **kwargs)

    elif reg == 'lowpass':
        regularize  = functools.partial(    reg_lowpass,    alpha=alpha, **kwargs)

    else:
        raise ValueError('Unknown regularization: %s' % reg)


    # Configure online updates
    if batch_size is None or batch_size > n:
        batch_size = n

    ###
    # Reset the diagnostics output
    diagnostics   = {
        'encoder':          [],
        'dictionary':       [],
        'parameters':       {
            'n':                X.shape[1],
            'd':                X.shape[0] / 2,
            'm':                m,
            'reg':              reg,
            'alpha':            alpha,
            'max_steps':        max_steps,
            'max_admm_steps':   max_admm_steps,
            'batch_size':       batch_size,
            'auxiliary':        kwargs
        },
        'globals':  {
            'rho_init_a':   RHO_INIT_A,
            'rho_init_d':   RHO_INIT_D,
            'rho_min':      RHO_MIN,
            'rho_max':      RHO_MAX,
            'abs_tol':      ABSTOL,
            'rel_tol':      RELTOL,
            'mu':           MU,
            'tau':          TAU,
            't_checkup':    T_CHECKUP
        }
    }

    ###
    # Configure the encoder
    local_encoder = functools.partial(parallel_encoder, 
                                      n_threads=n_threads,
                                      output_diagnostics=True)

    error   = []

    ###
    # Initialize the dictionary
    #D = init_svd(X, m)
    D = init_columns(X, m)
    
    for (T, X_batch) in enumerate(_batches(X, batch_size, max_steps, shuffle), 1):

        ###
        # Encode the data bacth
        (A, A_diags) = local_encoder(X_batch, D, regularize, 
                                     max_iter=max_admm_steps)

        diagnostics['encoder'].append(A_diags)
        
        error.append(np.mean((D * A - X_batch)**2))
        if verbose:
            print '%4d| [A] MSE=%.3e' % (T, error[-1]),

        (StS_new, StX_new)  = _encoding_statistics(A, X_batch)

        gamma = (1.0 - 1.0/T)**BETA

        if T == 1:
            # For the first batch, take the encoding statistics as is
            StS     = StS_new
            StX     = StX_new
        else:
            # All subsequent batches get averaged into to the previous totals
            StS     = gamma * StS     + (1.0-gamma) * StS_new
            StX     = gamma * StX     + (1.0-gamma) * StX_new

        ###
        # Optimize the dictionary
        (D, D_diags)  = dictionary(StS, StX, m, 
                                   max_iter=max_admm_steps, 
                                   Dinitial=D)

        diagnostics['dictionary'].append(D_diags)

        error.append(np.mean((D * A - X_batch)**2))
        if verbose:
            print '\t| [D] MSE=%.3e' %  error[-1],
            print '\t| [A-D] %.3e' % (error[-2] - error[-1])

        # Rescale the dictionary: this can only help
        D = normalize_dictionary(D)


    diagnostics['error'] = np.array(error)

    # Package up the learned encoder function for future use
    my_encoder = functools.partial(parallel_encoder, 
                                   n_threads=n_threads, 
                                   D=D, 
                                   reg=regularize, 
                                   max_iter=max_admm_steps, 
                                   output_diagnostics=False)

    return (my_encoder, D, diagnostics)
#---                            ---#
