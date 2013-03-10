# CREATED:2013-03-10 10:59:10 by Brian McFee <brm2132@columbia.edu>
# unit tests for convolutional dictionary learning 

import CDL
import numpy

#-- Utility functions --#
def testComplexToReal2():
    # Generate random d-by-n complex matrices of various sizes
    # verify that complexToReal2 correctly separates real / imaginary

    for d in 2**numpy.arange(0, 12, 2):
        for n in 2**numpy.arange(0, 12, 4):
            X_cplx  = numpy.random.randn(d, n) + 1.j * numpy.random.randn(d, n)
            X_real2 = CDL.complexToReal2(X_cplx)

            # Verify shape match
            assert (X_cplx.shape[0] * 2  == X_real2.shape[0] and
                    X_cplx.shape[1]      == X_real2.shape[1])

            # Verify numerical match
            assert (numpy.allclose(X_cplx.real, X_real2[:d, :]) and
                    numpy.allclose(X_cplx.imag, X_real2[d:, :]))
            pass
        pass
    pass

def testReal2ToComplex():
    # Generate random 2d-by-n real matrices
    #
    for d in 2**numpy.arange(0, 12, 2):
        for n in 2**numpy.arange(0, 12, 4):
            X_real2 = numpy.random.randn(2 * d, n)
            X_cplx  = CDL.real2ToComplex(X_real2)

            # Verify shape match
            assert (X_cplx.shape[0] * 2  == X_real2.shape[0] and
                    X_cplx.shape[1]      == X_real2.shape[1])

            # Verify numerical match
            assert (numpy.allclose(X_cplx.real, X_real2[:d, :]) and
                    numpy.allclose(X_cplx.imag, X_real2[d:, :]))
            pass
        pass
    pass

#--                   --#

#-- Regularization and projection --#
#--                               --#
