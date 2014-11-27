from pylab import *

def edgeSmoothing(amplitude, smoothPoints=None, smoothFraction=0.05):
    N = len(amplitude)
    if smoothPoints is None:
        n = int(N * smoothFraction)
    else:
        n = smoothPoints
    amp = amplitude.copy()
    amp[0] = 0
    amp[1:n-1] *= (cos(arange(n-2)*pi/2/(n-2))**2)[::-1]
    amp[-n+1:-1] *= (cos(arange(n-2)*pi/2/(n-2))**2)
    amp[-1] = 0
    return amp
    

def noise(signal, SNR):
    return real(ifft( SNR / ( SNR + 1 ) * fft(signal[:]) ))

def deconvolution(signal, SNR, response=None):
    "Return a wiener filter deconvolution"
    if response is None:
        R = 1
    else:
        R = fft(response)
    return real(ifft( conj(R) * SNR  / ( abs(R)**2 * SNR  + 1 ) * fft(signal[:])))
    
