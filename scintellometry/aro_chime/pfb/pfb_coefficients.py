import numpy as np

def pfb():
    """Calculate the CHIME PFB.

    This is within rounding error of the one stored in
    pfb_coefficients_used.mat
    """
    return (np.sinc((np.arange(8192)-4096)/2048.) *
            (0.08+0.92*np.hanning(8192)))

if __name__ == '__main__':
    import scipy.io
    m = scipy.io.loadmat('pfb_coefficients_used.mat')
    c = m['coeff'][0]
    calc = pfb()
    print("RMS difference between matlab and calculated pfb = {0}"
          .format((c-calc).std()))
