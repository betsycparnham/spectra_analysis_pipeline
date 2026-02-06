
import numpy as np
class fft():
  def __init__(self):
    self.fft_zeros = np.zeros
    self.irfft = np.fft.irfft
    self.rfft = np.fft.rfft
    self.fft = np.fft.fft
    self.fft_all = np.fft.fft
  #### Conversion from Time to Frequency ####
  def timetofreq(self,N,dt):
      #Sampling Frequency (Max Freq)
      fs = 1/dt
      #Duration
      T = N*dt
      #Resolution
      Res = 1/T
      #Frequency Range
      #F = np.arange(0,fs, Res)
      f = np.linspace(0,(N-1)*(fs/N),N)

      #Only taking up to and inckuding the Nyquist Frequency: frequency domain can only be accurately represented without aliasing up to Fny
      fny = f[0:int(N/2+1)]

      return f,fny,Res

  ##### Time FFT Function #####
  def tfft(self,data,N):

      Y = self.fft(data)

      #Normalization and Scaling
      #Finding magnitude of Y
      Ymag = abs(Y)/N

      #Scaling, *2 for Oscillatory terms, then divide first term by 2 as DC terms don't need to be multiplied, the x2 is already done by np.fft i think
      Ymagplot = Ymag[0:int(N/2+1)]
      Ymagplot[0] = (1/2) * (Ymagplot[0])

      Ymagfull = Ymag
      Ymagfull[0] = (1/2) * Ymagfull[0]

      return Ymagplot,Ymagfull