# CREATED:2013-03-10 10:59:10 by Brian McFee <brm2132@columbia.edu>
# unit tests for convolutional dictionary learning 

import cdl
import numpy
import numpy.fft
import functools

# short-cut for numerical tolerance
EQ = functools.partial(numpy.allclose, atol=1e-6, rtol=1e-4)

#-- Utility functions --#
def test_complex_to_real2():
    # Generate random d-by-n complex matrices of various sizes
    # verify that complex_to_real2 correctly separates real / imaginary

    def __test(d, n):
        X_cplx  = numpy.random.randn(d, n) + 1.j * numpy.random.randn(d, n)
        X_real2 = cdl.complex_to_real2(X_cplx)

        # Verify shape match
        assert (X_cplx.shape[0] * 2  == X_real2.shape[0] and
                X_cplx.shape[1]      == X_real2.shape[1])

        # Verify numerical match
        assert (EQ(X_cplx.real, X_real2[:d, :]) and
                EQ(X_cplx.imag, X_real2[d:, :]))
        pass

    for d in 2**numpy.arange(0, 12, 2):
        for n in 2**numpy.arange(0, 12, 4):
            yield (__test, int(d), int(n))
        pass
    pass

def test_real2_to_complex():
    # Generate random 2d-by-n real matrices
    # verify that real2_to_complex correctly combines the top and bottom halves

    def __test(d, n):
        X_real2 = numpy.random.randn(2 * d, n)
        X_cplx  = cdl.real2_to_complex(X_real2)

        # Verify shape match
        assert (X_cplx.shape[0] * 2  == X_real2.shape[0] and
                X_cplx.shape[1]      == X_real2.shape[1])

        # Verify numerical match
        assert (EQ(X_cplx.real, X_real2[:d, :]) and
                EQ(X_cplx.imag, X_real2[d:, :]))
        pass

    for d in 2**numpy.arange(0, 12, 2):
        for n in 2**numpy.arange(0, 12, 4):
            yield (__test, int(d), int(n))
        pass
    pass

def test_columns_to_diags():

    def __test(d, m):
        # Generate a random 2d-by-n matrix
        X = numpy.random.randn(2 * d, m)

        # Cut it in half to get the real and imag components
        A = X[:d, :]
        B = X[d:, :]

        # Convert to its diagonal-block form
        Q = cdl.columns_to_diags(X)
        
        for k in range(m):
            for j in range(d):
                # Verify A
                assert (EQ(A[j, k], Q[j, k * d + j]) and
                        EQ(A[j, k], Q[d + j, m * d + k *d + j]))

                # Verify B
                assert (EQ(B[j, k], - Q[j, m * d + k * d + j]) and
                        EQ(B[j, k], Q[d + j, k *d + j]))
                pass
            pass
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass

def test_diags_to_columns():
    # This test assumes that columns_to_diags is correct.

    def __test(d, m):
        X = numpy.random.randn(2 * d, m)
        Q = cdl.columns_to_diags(X)
        X_back  = cdl.diags_to_columns(Q)
        assert EQ(X, X_back)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass


def test_columns_to_vector():
    def __test(d, m):
        # Generate a random matrix
        X = numpy.random.randn(2 * d, m)

        # Split real and imaginary components
        A = X[:d, :]
        B = X[d:, :]

        # Vectorize
        V = cdl.columns_to_vector(X)

        # Equality-test
        # A[:,k] => V[d * k : d * (k+1)]
        # B[:,k] => V[d * (m + k) : d * (m + k + 1)]
        for k in range(m):
            # We need to flatten here due to matrix/vector subscripting
            assert (EQ(A[:, k], V[k*d:(k + 1)*d].flatten()) and
                    EQ(B[:, k], V[(m + k)*d:(m + k + 1)*d].flatten()))
            pass
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass

def test_vector_to_columns():
    # This test assumes that columns_to_vector is correct.

    def __test(d, m):
        X = numpy.random.randn(2 * d, m)
        V = cdl.columns_to_vector(X)
        X_back = cdl.vector_to_columns(V, m)

        assert EQ(X, X_back)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass

def test_normalize_dictionary():
    def __test(d, m):
        # Generate a random dictionary
        X       = numpy.random.randn(2 * d, m)

        # Convert to diagonals
        Xdiags  = cdl.columns_to_diags(X)

        # Normalize
        N_Xdiag = cdl.normalize_dictionary(Xdiags)

        # Convert back
        N_X     = cdl.diags_to_columns(N_Xdiag)

        # 1. verify unit norms
        norms   = numpy.sum(N_X**2, axis=0)**0.5
        assert EQ(norms, numpy.ones_like(norms))

        # 2. verify that directions are correct:
        #       projection onto the normalized basis should equal norm 
        #       of the original basis
        norms_orig = numpy.sum(X**2, axis=0)**0.5
        projection = numpy.sum(X * N_X, axis=0)

        assert EQ(norms_orig, projection)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass
#--                   --#

#-- Regularization and projection --#
def test_proj_l2_ball():
    
    def __test(d, m):
        # Build a random dictionary in diagonal form
        X       = numpy.random.randn(2 * d, m)

        # Normalize the dictionary and convert back to columns
        X_norm  = cdl.diags_to_columns(cdl.normalize_dictionary(cdl.columns_to_diags(X)))

        
        # Rescale the normalized dictionary to have some inside, some outside
        R       = 2.0**numpy.linspace(-4, 4, m)
        X_scale = X_norm * R

        # Vectorize
        V_scale = cdl.columns_to_vector(X_scale)

        # Project
        V_proj  = cdl.proj_l2_ball(V_scale, m)

        # Rearrange the projected matrix into columns
        X_proj  = cdl.vector_to_columns(V_proj, m)


        # Compute norms
        X_scale_norms   = numpy.sum(X_scale**2, axis=0)**0.5
        X_proj_norms    = numpy.sum(X_proj**2, axis=0)**0.5

        Xdot            = numpy.sum(X_scale * X_proj, axis=0)

        for k in range(m):
            # 1. verify norm is at most 1
            #       allow some fudge factor here for numerical instability
            assert X_proj_norms[k] <= 1.0 + 1e-10

            # 2. verify that points with R < 1 were untouched
            assert R[k] > 1.0 or EQ(X_proj[:, k], X_scale[:, k])

            # 3. verify that points with R >= 1.0 preserve direction
            assert R[k] < 1.0 or EQ(Xdot[k], X_scale_norms[k])

        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(2, 10):
            yield (__test, int(d), int(m))
        pass
    pass

def test_reg_l1_real():

    def __test(d, n, rho, lam, nonneg):
        # Generate a random matrix, scale by lam/rho to encourage
        # non-trivial solutions
        X = numpy.random.randn(d, n) * lam / rho

        # Compute shrinkage on X
        Xshrunk = (X > lam / rho) * (X - lam / rho)

        if not nonneg:
            Xshrunk = Xshrunk + (X < -lam / rho) * (X + lam / rho)
            pass

        # First, test without pre-allocation
        Xout = cdl.reg_l1_real(X, rho, lam, nonneg)

        assert EQ(Xout, Xshrunk)

        # Now test with pre-allocation
        Xout_pre = numpy.zeros_like(X, order='A')
        cdl.reg_l1_real(X, rho, lam, nonneg, Xout_pre)

        assert EQ(Xout_pre, Xshrunk)
        pass

    for d in 2**numpy.arange(0, 4):
        for n in 2**numpy.arange(0, 4):
            for rho in 2.0**numpy.arange(-4, 4):
                for lam in 10.0**numpy.arange(-3, 1):
                    for nonneg in [False, True]:
                        yield (__test, int(d), int(n), rho, lam, nonneg)
    pass

def test_reg_l1_complex():

    def __test(d, n, rho, lam):
        # Generate a random matrix, scale by lam/rho to encourage
        # non-trivial solutions
        X = numpy.random.randn(2 * d, n) * lam / rho

        # Compute magnitudes of complex values
        X_cplx  = cdl.real2_to_complex(X)
        X_abs   = numpy.abs(X_cplx)

        # Compute shrinkage on X
        # x -> (1 - (lam/rho) / |x|)_+ * x
        S       = X_abs.copy()

        # Avoid numerical instability here
        S[S < lam / rho] = (lam / rho)
        S   = (1 - (lam/rho) / S)
        
        # Tile it to get real and complex
        S   = numpy.tile(S, (2, 1))

        # Compute shrinkage
        Xshrunk = X * S

        # First, test without pre-allocation
        Xout = cdl.reg_l1_complex(X, rho, lam)

        assert EQ(Xout, Xshrunk, rtol=1e-3, atol=1e-6)

        # Now test with pre-allocation
        Xout_pre = numpy.zeros_like(X, order='A')
        cdl.reg_l1_complex(X, rho, lam, Xout_pre)

        assert EQ(Xout_pre, Xshrunk, rtol=1e-3, atol=1e-6)
        pass

    for d in 2**numpy.arange(0, 4):
        for n in 2**numpy.arange(0, 4):
            for rho in 2.0**numpy.arange(-4, 4):
                for lam in 10.0**numpy.arange(-3, 1):
                    yield (__test, int(d), int(n), rho, lam)
    pass


def test_reg_l2_group():

    def __test(d, m, n, rho, lam):
        # Generate a random matrix 2*d*m-by-n matrix
        # scale by lam/rho to encourage non-trivial solutions
        X = numpy.random.randn(2 * d * m, n) * lam / rho

        # Compute the properly shrunk matrix
        X_cplx  = cdl.real2_to_complex(X)
        S       = (X_cplx.conj() * X_cplx).real

        X_norms = numpy.zeros((m, n))

        for k in range(m):
            X_norms[k, :] = numpy.sum(S[k*d:(k+1)*d, :], axis=0)**0.5

        Xshrunk = numpy.zeros_like(X, order='A')

        for i in range(n):
            for k in range(m):
                if X_norms[k, i] > lam / rho:
                    scale = 1.0 - lam / (rho * X_norms[k, i])
                else:
                    scale = 0.0
                Xshrunk[(k*d):(k+1)*d, i] = scale * X[d*k:d*(k+1), i]
                Xshrunk[d*(m+k):(m+k+1)*d, i] = scale * X[d*(m+k):d*(m+k+1), i]
                pass
            pass

        # First, test without pre-allocation
        Xout = cdl.reg_l2_group(X, rho, lam, m)
        assert EQ(Xout, Xshrunk)

        # Now test with pre-allocation

        Xout_pre = numpy.zeros_like(X, order='A')
        cdl.reg_l2_group(X, rho, lam, m, Xout_pre)

        assert EQ(Xout_pre, Xshrunk)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            for n in 2**numpy.arange(0, 8, 2):
                for rho in 2.0**numpy.arange(-2, 2):
                    for lam in 10.0**numpy.arange(-2, 2):
                        yield (__test, int(d), int(m), int(n), rho, lam)
    pass

def test_reg_l1_space():

    def __test(w, h, m, n, rho, lam, nonneg):
        # Generate several frames of data
        X_space = numpy.random.randn(h, w, m, n) * lam / rho

        # Compute shrinkage point-wise
        Xshrunk = (X_space > lam / rho) * (X_space - lam / rho)

        if not nonneg:
            Xshrunk = Xshrunk + (X_space < -lam / rho) * (X_space + lam / rho)

        # Compute 2d-dfts
        X_freq  = numpy.fft.fft2(X_space, axes=(0, 1))

        # Reshape the data into columns
        X_cols  = X_freq.reshape((h * w * m, n), order='F')

        # Split real and complex
        X       = cdl.complex_to_real2(X_cols)

        # First try without pre-allocation
        Xout    = cdl.reg_l1_space(X, rho, lam, width=w, height=h, nonneg=nonneg)

        # Convert back to complex
        Xout_cplx = cdl.real2_to_complex(Xout)

        # reshape back into four dimensions
        Xout_freq = Xout_cplx.reshape((h, w, m, n), order='F')


        # Invert the fourier transform
        Xout_space = numpy.fft.ifft2(Xout_freq, axes=(0, 1)).real

        assert EQ(Xout_space, Xshrunk)

        # Now do it again with pre-allocation
        Xout_pre = numpy.empty_like(X, order='A')
        cdl.reg_l1_space(X, rho, lam, width=w, height=h, nonneg=nonneg, Xout=Xout_pre)

        # Convert back to complex
        Xout_cplx_pre = cdl.real2_to_complex(Xout_pre)

        # reshape back into four dimensions
        Xout_freq_pre = Xout_cplx_pre.reshape((h, w, m, n), order='F')
        # Invert the fourier transform
        Xout_space_pre = numpy.fft.ifft2(Xout_freq_pre, axes=(0, 1)).real

        assert EQ(Xout_space_pre, Xshrunk)
        pass

    for w in 2**numpy.arange(0, 8, 4):
        for h in 2**numpy.arange(0, 8, 4):
            for m in 2**numpy.arange(0, 8, 4):
                for n in 2**numpy.arange(0, 8, 4):
                    for rho in 2.0**numpy.arange(-2, 2):
                        for lam in 10.0**numpy.arange(-2, 1):
                            for nonneg in [False, True]:
                                yield (__test, int(w), int(h), int(m), int(n), rho, lam, nonneg)
    pass
#--                               --#
