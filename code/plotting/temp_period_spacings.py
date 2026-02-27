import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy import interpolate
import scipy.io as sci
from scipy import stats
from scipy.signal import find_peaks
import scipy.sparse.linalg as spla







class BackgroundProfile:
    def __init__(self, nnew, ntotal, numprocs, time_step, refstate):
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




def radial_order_calculation(foundpeakidxs, foundpeaks, spectrum, radius):
    freqs = foundpeaks[::-1]
    amps = spectrum[foundpeakidxs[::-1], :]


    from scipy.signal import find_peaks
    rad_orders = []
    lower = 44
    for i in range(len(freqs)):

        from scipy.ndimage import gaussian_filter1d
        if i < 10:
            A = gaussian_filter1d(amps[i, lower:135], sigma=1)
            dAdr = np.gradient(A, radius[lower:135])
            d2Adr2 = np.gradient(dAdr, radius[lower:135])
        else:
            A = amps[i, lower:135]
            dAdr = np.gradient(A, radius[lower:135])
            d2Adr2 = np.gradient(dAdr, radius[lower:135])

        absA = np.abs(A)
        nodes, foo = find_peaks(-absA, distance=2)
        rad_orders.append(len(nodes))

    return rad_orders



def pspmake2_period(period_days, nstart):
    import numpy as np

    period_days = np.asarray(period_days)

    # Period spacings (in days)
    deltap = -np.diff(period_days)

    # Reverse ordering if required
    period_r = period_days[::-1]

    radial_orders_r = np.arange(len(period_r)) + nstart
    radial_orders = -radial_orders_r[::-1]

    return deltap, period_days[1:], radial_orders[1:]



def spectrum_to_periodogram(spectrum, frequencies):
    # Input spectra at one radius and corresponding frequencies
    periods = 1 / (frequencies * 1e-6)  # in seconds
    P = periods / (3600 * 24)  # convert to days

    return spectrum



def expectedpeaks(foundfreqs, num_lines):
    # Takes in foundpeaks - peak frequencies in microhz going from high to low n
    peak_periods = 1 / (foundfreqs * 1e-6)  # in seconds
    peak_periods = peak_periods / (3600 * 24)  # in days
    start = peak_periods[-2] #n=-1
    print('start:', start)
    step = peak_periods[-4] - peak_periods[-3] # n=-2 - n=-1
    print('step:', step)
    expected_p = start + step * np.arange(num_lines)
    expected_f = 1 / ((expected_p * 3600 * 24)) * 1e6

    return expected_p, expected_f




bg = BackgroundProfile(nnew=1024, ntotal=722, numprocs=30, time_step=1000, refstate='/home/b9027676/PhD/fittinglorentzianswithbetsy/2MS_50_90.dat')
radius, rho, temp, dtbardz, diff, g, N2 = bg.radius[::10], bg.rho[::10], bg.temp[::10], bg.dtbardz[::10], bg.diff[::10], bg.g[::10], bg.N2[::10]


initial_datafile_50 = '/home/b9027676/PhD/fittinglorentzianswithbetsy/xc0.5-nofield-composite_m1.npz'

initial_datafile_214 = '/home/b9027676/PhD/fittinglorentzianswithbetsy/xc0.5-a214-composite_m1.npz'

initial_datafile_514 = '/home/b9027676/PhD/fittinglorentzianswithbetsy/xc0.5-a514-composite_m1.npz'

initial_datafile_brn = '/home/b9027676/PhD/fittinglorentzianswithbetsy/xc0.5-brn-composite_m1.npz'

initial_datafile_nrbrn1 = '/home/b9027676/PhD/fittinglorentzianswithbetsy/xc0.5-nrbrn1-composite_m1.npz'

initial_datafile_fr2brn1 = '/home/b9027676/PhD/fittinglorentzianswithbetsy/xc0.5-fr2brn1-composite_m1.npz'

"""def read_data(initial_datafile):
    data = np.load(initial_datafile)

    allf = data['allfrequencies']
    allspectrum = data['allspectrum']
    foundpeaks = data['freq_peaks']
    foundpeakidxs = data['peak_indices']
    allfoundamps = data['allpeakamplitudes']

    f1 = data['frequencies1']
    spectrum1 = data['spectrum1']
    foundamps1 = data['peak_amplitudes1']


    #data1 = np.load(final_datafile)


    return allf, allspectrum, allfoundamps, foundpeaks, foundpeakidxs, f1, spectrum1, foundamps1"""


def read_data1(initial_datafile):
    data = np.load(initial_datafile)

    freq = data['freq']
    period = data['period']
    spec_composite = data['compositespectrum']
    peakidxs = data['peak_idxs']
    specallradii = data['allspec']


    return freq, period, spec_composite , peakidxs, specallradii





print('---- No Field m1 ----')
freq50, period50, spec_composite50 , peakidxs50, specallradii50 = read_data1(initial_datafile_50)
periodpeaks50 = period50[peakidxs50]
deltap_peaks_i50, period_peaks_i50, radial_orders_i50 = pspmake2_period(periodpeaks50, nstart=0)
spectrum1_50 = spectrum_to_periodogram(spec_composite50, freq50)


print('---- a214 m1 ----')
freqa214, period214, spec_composite214 , peakidxs214, specallradii214 = read_data1(initial_datafile_214)
periodpeaks214 = period214[peakidxs214]
deltap_peaks_i214, period_peaks_i214, radial_orders_i214 = pspmake2_period(periodpeaks214, nstart=0)
spectrum1_214 = spectrum_to_periodogram(spec_composite214, freqa214)

print('---- a514 m1 ----')
freq514, period514, spec_composite514 , peakidxs514, specallradii514 = read_data1(initial_datafile_514)
periodpeaks514 = period514[peakidxs514]
deltap_peaks_i514, period_peaks_i514, radial_orders_i514 = pspmake2_period(periodpeaks514, nstart=0)
spectrum1_514 = spectrum_to_periodogram(spec_composite514, freq514)


print('---- brn m1 ----')
freqbrn, periodbrn, spec_compositebrn , peakidxsbrn, specallradiibrn = read_data1(initial_datafile_brn)
periodpeaksbrn = periodbrn[peakidxsbrn]
deltap_peaks_ibrn, period_peaks_ibrn, radial_orders_ibrn = pspmake2_period(periodpeaksbrn, nstart=0)
spectrum1_brn = spectrum_to_periodogram(spec_compositebrn, freqbrn)


print('---- nrbrn1 m1 ----')
freqnrbrn1, periodnrbrn1, spec_compositenrbrn1 , peakidxsnrbrn1, specallradiinrbrn1 = read_data1(initial_datafile_nrbrn1)
periodpeaksnrbrn1 = periodnrbrn1[peakidxsnrbrn1]
deltap_peaks_inrbrn1, period_peaks_inrbrn1, radial_orders_inrbrn1 = pspmake2_period(periodpeaksnrbrn1, nstart=0)
spectrum1_nrbrn1 = spectrum_to_periodogram(spec_compositenrbrn1, freqnrbrn1)

print('---- fr2brn1 m1 ----')
freqfr2brn1, periodfr2brn1, spec_compositefr2brn1 , peakidxsfr2brn1, specallradiifr2brn1 = read_data1(initial_datafile_fr2brn1)
periodpeaksfr2brn1 = periodfr2brn1[peakidxsfr2brn1]
deltap_peaks_ifr2brn1, period_peaks_ifr2brn1, radial_orders_ifr2brn1 = pspmake2_period(periodpeaksfr2brn1, nstart=0)
spectrum1_fr2brn1 = spectrum_to_periodogram(spec_compositefr2brn1, freqfr2brn1)

"""

print('---- A214 ----')
allf214, allspectrum214, allfoundamps214, foundpeaks214, foundpeakidxs214, f1_214, spectrum1_214, foundamps1_214 = read_data(initial_datafile_a214)
deltap_peaks_a214, period_peaks_a214, radial_orders_a214 = pspmake2(foundpeaks214, nstart=0)
#deltap_peaks_a214, period_peaks_a214, radial_orders_a214 = pspmake(foundpeaks214, foundpeakidxs214, allspectrum214, nstart=0)
print(radial_orders_a214[0:4], len(radial_orders_a214))


print('---- A514 ----')
allf514, allspectrum514, allfoundamps514, foundpeaks514, foundpeakidxs514, f1_514, spectrum1_514, foundamps1_514 = read_data(initial_datafile_a514)
deltap_peaks_a514, period_peaks_a514, radial_orders_a514 = pspmake2(foundpeaks514, nstart=0)
#deltap_peaks_a514, period_peaks_a514, radial_orders_a514 = pspmake(foundpeaks514, foundpeakidxs514, allspectrum514, nstart=0)
print(radial_orders_a514[0:4], len(radial_orders_a514))"""


plt.figure()
#plt.title('Initial Model')
plt.plot(period_peaks_i50, deltap_peaks_i50, label='No Field', color='blue', marker='o', linestyle='solid')
#plt.plot(period_peaks_i214, deltap_peaks_i214, label='a214', color='orange', marker='o', linestyle='solid')
#plt.plot(period_peaks_i514, deltap_peaks_i514, label='a514', color='green', marker='o', linestyle='solid')
plt.plot(period_peaks_ibrn, deltap_peaks_ibrn, label='brn', color='lightsteelblue', marker='o', linestyle='solid')
#plt.plot(period_peaks_inrbrn1, deltap_peaks_inrbrn1, label='nrbrn1', color='purple', marker='o', linestyle='solid')
#plt.plot(period_peaks_ifr2brn1, deltap_peaks_ifr2brn1, label='fr2brn1', color='grey', marker='o', linestyle='solid')
plt.xlabel('Period (d)')
plt.ylabel('Period Spacing (d)')
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.title('Initial Model')
plt.plot(radial_orders_i50, deltap_peaks_i50, label='No Field', color='blue', marker='o', linestyle='solid')
#plt.plot(radial_orders_a214, deltap_peaks_a214, label='A214 Peaks', color='orange', marker='o', linestyle='solid')
#plt.plot(radial_orders_a514, deltap_peaks_a514, label='A514 Peaks', color='green', marker='o', linestyle='solid')
plt.xlabel('Radial Order')
plt.ylabel('Period Spacing (s)')
plt.legend()
plt.grid()
plt.show()













############################ RADIAL ORDER IDENTIFICATION FUNCTION DOWN HERE!!!!!!!!!!
"""for i in range(1, len(freqs50)):

    from scipy.ndimage import gaussian_filter1d
    if i < 11:
        A = gaussian_filter1d(amps50[i, lower:135], sigma=1)
        dAdr = np.gradient(A, radius[lower:135])
        d2Adr2 = np.gradient(dAdr, radius[lower:135])
    elif i >11 and i<14:
        A = gaussian_filter1d(amps50[i, lower:135], sigma=0.5)
        dAdr = np.gradient(amps50[i, lower:135], radius[lower:135])
        d2Adr2 = np.gradient(dAdr, radius[lower:135])
    else:
        A = amps50[i, lower:135]
        dAdr = np.gradient(amps50[i, lower:135], radius[lower:135])
        d2Adr2 = np.gradient(dAdr, radius[lower:135])


    num_new_points = 10000 # or 10000 
    x_new = np.linspace(radius[lower:135].min(), radius[lower:135].max(), num_new_points) # Interpolate linearly 
    y_new = np.interp(x_new, radius[lower:135], A)
    dydr_new = np.interp(x_new, radius[lower:135], dAdr)
    d2ydr2_new = np.interp(x_new, radius[lower:135], d2Adr2)

    sign_changes = np.where((np.diff(np.sign(dydr_new))) != 0)[0]
    print("sign",np.sign(dydr_new))
    print("diff",np.diff(np.sign(dydr_new)))
    zero_crossings = sign_changes
    print('expected:', i)
    #print('found:', len(zero_crossings)//2)

    troughs = []
    peaks = []
    for i in range(len(sign_changes)):
        dy_dr_i = dydr_new[sign_changes[i]]
        dy_dr_i_1 = dydr_new[sign_changes[i]+1]
        if dy_dr_i<0 and dy_dr_i_1>0:
            troughs.append(x_new[zero_crossings[i]])
        elif dy_dr_i>0 and dy_dr_i_1<0:
            peaks.append(x_new[zero_crossings[i]])
    
    print('found:', len(troughs))
    plt.plot(x_new[1:], np.diff(np.sign(dydr_new)))
    plt.show()


    plt.figure()
    #plt.plot(radius[lower:135], A, label=f'Freq: {freqs50[i]} microHz, Idx: {idxs50[i]}, Suspected Radial Order: {i}')
    plt.plot(x_new, y_new, label='Interpolated A')
    plt.hlines(0, np.min(radius[lower:135]), np.max(radius[lower:135]), linestyle='dashed', color='black')
    #plt.vlines(radius[lower:135][zero_crossings], np.min(dAdr), np.max(dAdr), linestyle='dashed', color='orange', label='Detected Zero Crossings')
    #plt.vlines(x_new[zero_crossings], np.min(dAdr), np.max(dAdr), linestyle='dashed', color='orange', label='Detected Zero Crossings')
    plt.vlines(troughs, np.min(A), np.max(A), linestyle='dashed', color='orange', label='Detected troughs')
    plt.vlines(peaks, np.min(A), np.max(A), linestyle='dashed', color='red', label='Detected peaks')
    
    plt.xlabel('Radius')
    plt.ylabel('Mode Amplitude')
    plt.legend()
    plt.grid()
    plt.savefig(f"peaks and troughs_{i}.png")"""