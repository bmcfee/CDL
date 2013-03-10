# CREATED:2013-03-10 10:59:10 by Brian McFee <brm2132@columbia.edu>
# unit tests for convolutional dictionary learning 

import CDL
import numpy

#-- Utility functions --#
def test_complexToReal2():
    # Generate random d-by-n complex matrices of various sizes
    # verify that complexToReal2 correctly separates real / imaginary

    def __test(d, n):
        X_cplx  = numpy.random.randn(d, n) + 1.j * numpy.random.randn(d, n)
        X_real2 = CDL.complexToReal2(X_cplx)

        # Verify shape match
        assert (X_cplx.shape[0] * 2  == X_real2.shape[0] and
                X_cplx.shape[1]      == X_real2.shape[1])

        # Verify numerical match
        assert (numpy.allclose(X_cplx.real, X_real2[:d, :]) and
                numpy.allclose(X_cplx.imag, X_real2[d:, :]))
        pass

    for d in 2**numpy.arange(0, 12, 2):
        for n in 2**numpy.arange(0, 12, 4):
            yield (__test, int(d), int(n))
        pass
    pass

def test_real2ToComplex():
    # Generate random 2d-by-n real matrices
    # verify that real2ToComplex correctly combines the top and bottom halves

    def __test(d, n):
        X_real2 = numpy.random.randn(2 * d, n)
        X_cplx  = CDL.real2ToComplex(X_real2)

        # Verify shape match
        assert (X_cplx.shape[0] * 2  == X_real2.shape[0] and
                X_cplx.shape[1]      == X_real2.shape[1])

        # Verify numerical match
        assert (numpy.allclose(X_cplx.real, X_real2[:d, :]) and
                numpy.allclose(X_cplx.imag, X_real2[d:, :]))
        pass

    for d in 2**numpy.arange(0, 12, 2):
        for n in 2**numpy.arange(0, 12, 4):
            yield (__test, int(d), int(n))
        pass
    pass

def test_columnsToDiags():

    def __test(d, m):
        # Generate a random 2d-by-n matrix
        X = numpy.random.randn(2 * d, m)

        # Cut it in half to get the real and imag components
        A = X[:d, :]
        B = X[d:, :]

        # Convert to its diagonal-block form
        Q = CDL.columnsToDiags(X)
        
        for k in range(m):
            for j in range(d):
                # Verify A
                assert (numpy.allclose(A[j, k], Q[j, k * d + j]) and
                        numpy.allclose(A[j, k], Q[d + j, m * d + k *d + j]))

                # Verify B
                assert (numpy.allclose(B[j, k], - Q[j, m * d + k * d + j]) and
                        numpy.allclose(B[j, k], Q[d + j, k *d + j]))
                pass
            pass
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass

def test_diagsToColumns():
    # This test assumes that columnsToDiags is correct.

    def __test(d, m):
        X = numpy.random.randn(2 * d, m)
        Q = CDL.columnsToDiags(X)
        X_back  = CDL.diagsToColumns(Q)
        assert numpy.allclose(X, X_back)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass


def test_columnsToVector():
    def __test(d, m):
        # Generate a random matrix
        X = numpy.random.randn(2 * d, m)

        # Split real and imaginary components
        A = X[:d, :]
        B = X[d:, :]

        # Vectorize
        V = CDL.columnsToVector(X)

        # Equality-test
        # A[:,k] => V[d * k : d * (k+1)]
        # B[:,k] => V[d * (m + k) : d * (m + k + 1)]
        for k in range(m):
            # We need to flatten here due to matrix/vector subscripting
            assert (numpy.allclose(A[:, k], V[k*d:(k + 1)*d].flatten()) and
                    numpy.allclose(B[:, k], V[(m + k)*d:(m + k + 1)*d].flatten()))
            pass
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass

def test_vectorToColumns():
    # This test assumes that columnsToVector is correct.

    def __test(d, m):
        X = numpy.random.randn(2 * d, m)
        V = CDL.columnsToVector(X)
        X_back = CDL.vectorToColumns(V, m)

        assert numpy.allclose(X, X_back)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, int(d), int(m))
        pass
    pass

def test_normalizeDictionary():
    def __test(d, m):
        # Generate a random dictionary
        X       = numpy.random.randn(2 * d, m)

        # Convert to diagonals
        Xdiags  = CDL.columnsToDiags(X)

        # Normalize
        N_Xdiag = CDL.normalizeDictionary(Xdiags)

        # Convert back
        N_X     = CDL.diagsToColumns(N_Xdiag)

        # 1. verify unit norms
        norms   = numpy.sum(N_X**2, axis=0)**0.5
        assert numpy.allclose(norms, numpy.ones_like(norms))

        # 2. verify that directions are correct:
        #       projection onto the normalized basis should equal norm 
        #       of the original basis
        norms_orig = numpy.sum(X**2, axis=0)**0.5
        projection = numpy.sum(X * N_X, axis=0)

        assert numpy.allclose(norms_orig, projection)
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
        X_norm  = CDL.diagsToColumns(CDL.normalizeDictionary(CDL.columnsToDiags(X)))

        
        # Rescale the normalized dictionary to have some inside, some outside
        R       = 2.0**numpy.linspace(-4, 4, m)
        X_scale = X_norm * R

        # Vectorize
        V_scale = CDL.columnsToVector(X_scale)

        # Project
        V_proj  = CDL.proj_l2_ball(V_scale, m)

        # Rearrange the projected matrix into columns
        X_proj  = CDL.vectorToColumns(V_proj, m)


        # Compute norms
        X_scale_norms   = numpy.sum(X_scale**2, axis=0)**0.5
        X_proj_norms    = numpy.sum(X_proj**2, axis=0)**0.5

        Xdot            = numpy.sum(X_scale * X_proj, axis=0)

        for k in range(m):
            # 1. verify norm is at most 1
            #       allow some fudge factor here for numerical instability
            assert X_proj_norms[k] <= 1.0 + 1e-10

            # 2. verify that points with R < 1 were untouched
            assert R[k] > 1.0 or numpy.allclose(X_proj[:, k], X_scale[:, k])

            # 3. verify that points with R >= 1.0 preserve direction
            assert R[k] < 1.0 or numpy.allclose(Xdot[k], X_scale_norms[k])

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
        Xout = CDL.reg_l1_real(X, rho, lam, nonneg)

        assert numpy.allclose(Xout, Xshrunk)

        # Now test with pre-allocation
        Xout_pre = numpy.zeros_like(X, order='A')
        CDL.reg_l1_real(X, rho, lam, nonneg, Xout_pre)

        assert numpy.allclose(Xout_pre, Xshrunk)
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
        X_cplx  = CDL.real2ToComplex(X)
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
        Xout = CDL.reg_l1_complex(X, rho, lam)

        assert numpy.allclose(Xout, Xshrunk)

        # Now test with pre-allocation
        Xout_pre = numpy.zeros_like(X, order='A')
        CDL.reg_l1_complex(X, rho, lam, Xout_pre)

        assert numpy.allclose(Xout_pre, Xshrunk)
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
        X_cplx  = CDL.real2ToComplex(X)
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
        Xout = CDL.reg_l2_group(X, rho, lam, m)
        assert numpy.allclose(Xout, Xshrunk)

        # Now test with pre-allocation

        Xout_pre = numpy.zeros_like(X, order='A')
        CDL.reg_l2_group(X, rho, lam, m, Xout_pre)

        assert numpy.allclose(Xout_pre, Xshrunk)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            for n in 2**numpy.arange(0, 8, 2):
                for rho in 2.0**numpy.arange(-4, 4, 2):
                    for lam in 10.0**numpy.arange(-3, 1):
                        yield (__test, int(d), int(m), int(n), rho, lam)
    pass
#--                               --#
