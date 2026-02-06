import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from ..utils.fft_functions import fft
import scipy.io as sci
matplotlib.rcParams['font.size']=20
from scipy.signal import find_peaks
import matplotlib.colors as colors


############# Returns All Necessary data to plot frequency spectrum for a given quantity at given radial points ##############

class BackgroundProfile:
    def __init__(self, nnew=1024, ntotal=722, numprocs=30, time_step=1000, refstate = None,vzc50_file = None, vzs50_file = None):
        self.vzs50_file = vzs50_file
        self.vzc50_file = vzc50_file
        backgroundstate = np.genfromtxt(refstate,delimiter='\t')
        self.vzc50 = sci.FortranFile(str(vzc50_file)).read_record(dtype="f8").reshape((39000,11,150),order='F')
        self.vzs50 = sci.FortranFile(str(vzs50_file)).read_record(dtype="f8").reshape((39000,11,150),order='F')
        # ---- parameters ----
        self.dt = time_step

        # ---- load background data ----
        backdat = np.genfromtxt(refstate, delimiter='\t')

        # ---- geometry ----
        self.nn = int(ntotal / 2)
        self.nzones = len(backdat)
        self.nz = int(self.nzones / numprocs)

        # ---- background quantities ----
        self.radius = backdat[:, 0]
        self.den = backdat[:, 1]
        self.rho = self.den
        self.temp = backdat[:, 2]
        self.dtbardz = backdat[:, 3]
        self.diff = backdat[:, 4]
        self.g = backdat[:, 5]
        self.hrho = backdat[:, 6]   # negative inverse scale height
        self.hrho2 = backdat[:, 7]
        self.a = backdat[:, 8:10]
        self.b = backdat[:, 11:13]

        # ---- derived quantities ----
        self.gamma = 5 / 3
        self.N2 = (self.g * self.diff) / (self.gamma * self.temp)


class Read_CompositeSpectrum:
    def __init__(self, filepath):
        dat = np.load(filepath)

        self.freq = dat['freq']
        self.period = dat['period']
        self.composite_spec = dat['compositespectrum']
        self.peakidxs = dat['peak_idxs']
        self.allspec = dat['allspec']

        self.freq_peaks = self.freq[self.peakidxs]
        self.period_peaks = self.period[self.peakidxs]
        self.all_amplitudes = self.allspec[self.peakidxs, :]

    def __repr__(self):
        return (f"CompositeSpectrum(n_freq={len(self.freq)}, "
                f"n_peaks={len(self.peakidxs)})")

class frequency_spectrum:
    def __init__(self,bg):
        self.rp = 100
        self.nrange = np.array([0,-1])
        self.rpoints = np.arange(0, bg.nzones, 10) //10
        self.f, self.vzspec1_50 = self.getspec(bg.vzc50, bg.vzs50, bg.dt)
        self.period = (1 / (self.f*1e-6)) / (3600*24) # in days
        self.spec = self.vzspec1_50[:, self.rp]
        self.breaks = [0.05, 0.0661, 0.125, 0.148, 0.27]
        self.rps = [100, 100, 100, 100, 140, 140]
        self.smoothingparams = [10, 10, 10, 50, 50, 20]
        self.fullspectrum = self.generate_composite_normalized_smoothed_spectrum(self.period, self.vzspec1_50, self.breaks, self.rps, self.smoothingparams)
        self.p_thresh, self.d_thresh, self.min_w, self.max_w, self.h_thresh = self.peaksparameters(self.fullspectrum, 0.075, 45, 200)
    def getspec(self,vzc, vzs, dt, Nt = 30000, Nr = 150, rpoints = 100, m = 5):
        vzc = vzc[:Nt, :, :]
        vzs = vzs[:Nt, :, :]
        vzhat = vzc + 1j*vzs
        freq, vzspec_m1 = self.fspec_m(vzhat, Nt, dt, Nr, self.rpoints, m)
        f = freq*1e6
        return f, vzspec_m1
    def FreqSpectrum(self,data,tnum,dt,Nr,Rpoints):
        import tqdm as tq
        fft_f = fft()
        f,fny,Res = fft_f.timetofreq(tnum,dt)  #Returns full frequency range, Nyquist Frequency and Frequency resolution

        dataTfull = np.zeros([tnum,Nr])
        dataTmag = np.zeros([int(tnum/2)+1,Nr]) #Initialising both the full set of data and the set of data to be used for plotting
        for i in tq.trange(0,Nr):
            dataTmag[:,i],dataTfull[:,i] = fft_f.tfft(data[:,Rpoints[i]],tnum) # Performing Time fft for selected radii and adding them to arrays to use for plotting

        return f,fny,Res,dataTmag,dataTfull



    def fspec_m(self,data_full, tnum, dt, Nr, Rpoints, mode):
          data = data_full[:, mode, :]
          foo,f,Res,spectrum,foo = self.FreqSpectrum(data,tnum,dt,Nr,Rpoints)
          return f, spectrum


    def generate_composite_normalized_smoothed_spectrum(self,period, spectrum, breaks, rps, smoothingparams):
        from scipy.signal import savgol_filter
        idxs = [np.argmin(np.abs(period - t)) for t in breaks]
        fullspectrum = np.zeros(len(spectrum[:, 0]))
        bounds = [None] + list(idxs) + [None]
        # Loop over each segment
        for i in range(len(rps)):
            start = bounds[i+1]      # lower index
            end   = bounds[i]        # upper index

            if start is None:
                start = 0
            if end is None:
                end = len(spectrum)

            segment = spectrum[start:end, rps[i]]

            smooth = savgol_filter(segment, smoothingparams[i], 2)
            smooth /= np.max(smooth)

            fullspectrum[start:end] = smooth

        return fullspectrum
        # FUNCTION TO CALCULATE FINDPEAKS PARAMETERS
    def peaksparameters(self,spectrum, prominence, distance, maxwidth):
        promthresh = prominence
        distthresh = distance
        med = np.median(spectrum)
        sigma = np.std(spectrum)
        heightthresh = 8*med
        return [promthresh, distthresh, 0, maxwidth, heightthresh]


def calculate_expectedpeaks(period, peaks):
    periodpeaksrev_nof = period[(np.sort(peaks)[::-1])[1:]]
    start = periodpeaksrev_nof[0]
    step = np.average( np.diff(periodpeaksrev_nof[0:8]) )
    num_lines= len(periodpeaksrev_nof)
    expected_p = start + step * np.arange(num_lines)
    expected = np.array(expected_p)
    midpoints = (expected[:-1] + expected[1:]) / 2
    # Build left/right edges for each region
    left_edges  = np.concatenate(([expected[0] - (midpoints[0] - expected[0])], midpoints))
    right_edges = np.concatenate((midpoints, [expected[-1] + (expected[-1] - midpoints[-1])]))
    chunks = []
    for L, R in zip(left_edges, right_edges):
        # find nearest indices
        iL = np.argmin(np.abs(period - L))
        iR = np.argmin(np.abs(period - R))
        # ensure slice goes forward in index space
        i0, i1 = sorted([iL, iR])
        chunks.append((i0, i1))

    residuals = (periodpeaksrev_nof - np.sort(expected_p))
    percentresiduals = (residuals / np.sort(expected_p)) * 100
    detpeaknum = np.arange(1, len(expected_p)+1)

    return expected_p, residuals, percentresiduals, detpeaknum, chunks, left_edges, right_edges

class compute_composite_spectra():
    def __init__(self, data_dir,bg_file, vzc50_file, vzs50_file, nnew,ntotal,numprocs,time_step):
        # Getting Background State Profiles
        bg = BackgroundProfile(nnew=1024, ntotal=722, numprocs=30, time_step=1000, refstate=bg_file,vzc50_file= vzc50_file,vzs50_file=vzs50_file)
        #Take every 10th radial point as that's how the velocities are written
        radius, rho, temp, dtbardz, diff, g, N2 = bg.radius[::10], bg.rho[::10], bg.temp[::10], bg.dtbardz[::10], bg.diff[::10], bg.g[::10], bg.N2[::10]

        ############################################### no field ##########################################################

        # Calculating Frequency Spectrum
        freq_spectrum = frequency_spectrum(bg)

        peaks, props = find_peaks(freq_spectrum.fullspectrum, prominence=freq_spectrum.p_thresh, distance=freq_spectrum.d_thresh, width=(freq_spectrum.min_w, freq_spectrum.max_w), height=freq_spectrum.h_thresh)
        expected_p, residuals, percentresiduals, detpeaknum, chunks, left_edges, right_edges = calculate_expectedpeaks(freq_spectrum.period, peaks)

        # Saving Data
        frequency, period, spec_composite, peakidxs, specallradii = freq_spectrum.f, freq_spectrum.period, freq_spectrum.fullspectrum, peaks, freq_spectrum.vzspec1_50
        np.savez_compressed(data_dir + 'xc0.5-nofield-composite.npz', freq = frequency, period=period, compositespectrum=spec_composite, peak_idxs=peakidxs, allspec=specallradii)
        print('Data Saved')


        specdata = Read_CompositeSpectrum(data_dir + 'xc0.5-nofield-composite.npz')
        f, p, compspec, peakidxs, allspec, freqpeaks, periodpeaks, allamp = specdata.freq, specdata.period, specdata.composite_spec, specdata.peakidxs, specdata.allspec, specdata.freq_peaks, specdata.period_peaks, specdata.all_amplitudes
        print('Data Read Successfully')
        