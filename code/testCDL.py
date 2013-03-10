# CREATED:2013-03-10 10:59:10 by Brian McFee <brm2132@columbia.edu>
# unit tests for convolutional dictionary learning 

import CDL
import numpy

#-- Utility functions --#
def testComplexToReal2():
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
            yield (__test, d, n)
        pass
    pass

def testReal2ToComplex():
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
            yield (__test, d, n)
        pass
    pass

def testColumnsToDiags():

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
            yield (__test, d, m)
        pass
    pass

def testDiagsToColumns():
    # This test assumes that columnsToDiags is correct.

    def __test(d, m):
        X = numpy.random.randn(2 * d, m)
        Q = CDL.columnsToDiags(X)
        X_back  = CDL.diagsToColumns(Q)
        assert numpy.allclose(X, X_back)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, d, m)
        pass
    pass


def testColumnsToVector():
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
            yield (__test, d, m)
        pass
    pass

def testVectorToColumns():
    # This test assumes that columnsToVector is correct.

    def __test(d, m):
        X = numpy.random.randn(2 * d, m)
        V = CDL.columnsToVector(X)
        X_back = CDL.vectorToColumns(V, m)

        assert numpy.allclose(X, X_back)
        pass

    for d in 2**numpy.arange(0, 8, 2):
        for m in 2**numpy.arange(0, 8, 2):
            yield (__test, d, m)
        pass
    pass

def testNormalizeDictionary():
    pass
#--                   --#

#-- Regularization and projection --#
#--                               --#
