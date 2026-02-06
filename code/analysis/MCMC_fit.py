import numpy as np
import astropy
from astropy.modeling.functional_models import Lorentz1D
from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
import pymc as pm
import arviz as az
import argparse
from scipy.optimize import curve_fit
from pathlib import Path


class Read_CompositeSpectrum:
    def __init__(self, filepath):
        dat = np.load(filepath)

        self.fullfreq = dat['freq']
        self.period = dat['period']
        self.spectra = dat['compositespectrum']
        self.peakidxs = dat['peak_idxs']
        self.spectra_all = dat['allspec']

        self.frequencies = self.fullfreq[self.peakidxs]
        self.period_peaks = self.period[self.peakidxs]
        self.amps = self.spectra[self.peakidxs]

        plt.figure()
        plt.plot(self.period,self.spectra)
        plt.vlines(self.period_peaks,min(self.spectra),max(self.spectra),color = "red")
        plt.xlim(0,0.5)
        plt.savefig("test.png")


    def __repr__(self):
        return (f"CompositeSpectrum(n_freq={len(self.freq)}, "
                f"n_peaks={len(self.peakidxs)})")


def generate_test_spectra(x,loc_array,fwhm_array,amplitude_array,noise_bool):
    if len(x)==0:
        x = np.linspace(-np.max(abs(loc_array)+1),np.max(abs(loc_array)+1))
    f_noiseless = np.zeros(len(x))
    for i,loc in enumerate(loc_array):
        f_noiseless+= Lorentz1D(amplitude_array[i],loc,fwhm_array[i])(x)
    noise = np.random.randn(len(x))
    if noise_bool:

        f= f_noiseless+0.5*noise
    else:
        f = f_noiseless
    if noise_bool:
        return x,f,f_noiseless
    else:

        return x,f

def lorentz_function(x,loc,amp,fwhm):
    fwhm = fwhm/2
    return amp*(fwhm**2)/(fwhm**2 + (x-loc)**2)

class frequencies_from_file():
    def __init__(self,file):
        data = np.load(file)
        self.fullfreq = data['allfrequencies']       # Full Frequency Range
        self.spectra = data['spectrum']       # 1D Spectrum in the radiation zone
        try:
            self.frequencies = data['freq_peaks']       # Frequencies of peaks
            self.amps = data['peak_amplitudes']       # Amplitudes of peaks
        except:
            self.frequencies = None
            self.amps = None



class fit_lorentzian_least_squares():
    def __init__(self,x,f,n_peaks,peak_locations = None):
        self.n_peaks = n_peaks
        self.x = x
        self.f = f
        if peak_locations is None:
            self.peak_locations = np.linspace(min(x),max(x),n_peaks)
        else:
            self.peak_locations = peak_locations
        self.loc_array, self.fwhm_array,self.amp_array = self.least_squares(x,f)
    def lorentz_function(self,x,loc,amp,fwhm):
        fwhm = fwhm/2
        return amp*(fwhm**2)/(fwhm**2 + (x-loc)**2)
    def lorentzian_model(self,x,*param_array):
        output = np.zeros_like(x)
        for peak in range(self.n_peaks):
            loc,amp,fwhm = param_array[peak*3:peak*3+3]
            f_i = amp*(fwhm**2)/(fwhm**2 + (x-loc)**2)
            output +=f_i
        return output

    def least_squares(self,x,f):
        initial_params = []
        for peak in range(self.n_peaks):
            initial_params.extend([self.peak_locations[peak],1,1])
        popt,pcov = curve_fit(self.lorentzian_model,x,f,p0 = initial_params)
        
        loc_array = []
        amp_array = []
        fwhm_array = []
        for peak in range(self.n_peaks):
            params = popt[peak*3:peak*3+3]
            loc,amp,fwhm = params
            loc_array.append([loc])
            amp_array.append([amp])
            fwhm_array.append([fwhm])
        return loc_array, fwhm_array,amp_array

class fit_lorentzian_mcmc():
    def __init__(self,results_dir,title,n_draws,n_tune,x,f,n_peaks,prior_frequencies = None,prior_amplitudes = None):
        self.x = x
        self.f = f
        self.prior_frequencies = prior_frequencies
        #if self.prior_frequencies is not None:
        #    self.prior_amplitudes = prior_amplitudes[:,120]
        #else:
        self.prior_amplitudes = prior_amplitudes
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.n_peaks = n_peaks
        #self.super_lorentzian_fit(x,f)

        #self.polynom_fit(x,f)

        print(f'analyze n_peaks: {n_peaks}')
        if n_peaks>len(prior_frequencies):
            self.add_extra_peaks(n_peaks-len(prior_frequencies))
        trace = self.mcmc_fit(x,self.f,n_peaks)
        self.trace = trace
        self.loc_array, self.fwhm_array,self.amp_array = self.visualize_fit(x,trace,n_peaks)
        self.save_results(results_dir,title)

    def super_lorentzian_fit(self,x,f):
        from scipy import optimize as opt
        def super_lorentzian(x,w_0,alpha_0,nu_char,gamma):
            return w_0 + alpha_0*((x/nu_char)/(1+(x/nu_char)**(gamma)))
        popt,pcov = opt.curve_fit(super_lorentzian,x[np.where((x>0)&(x<200))], np.log10(f)[np.where((x>0)&(x<200))])
        f_fit = super_lorentzian(x,*popt)
        plt.figure()
        plt.plot(x,np.log10(f))
        plt.plot(x,f_fit)
        plt.xlim(0,75)
        plt.savefig("super_lorentzian.png")
    def polynom_fit(self,x, f):
        # Fitting uisng a 4th order polynomial
        from scipy import optimize as opt
        def polynomial(x,a,b,c, d, e):
            return a + b*x + c*x**2 + d*x**3 + e*x**4


        logf = np.log10(f)

        fitting_function = polynomial
        #Fitting in log space
        popt, pcov = opt.curve_fit(fitting_function, x[np.where((x>5)&(x<200))], np.log10(f)[np.where((x>5)&(x<200))])
        #popt, pcov = opt.curve_fit(fitting_function, x, np.log10(f))

        f_fit = fitting_function(x,*popt)

        residuals = f_fit - np.log10(f)
        ss_res = np.sum(residuals**2)
        print('Sum of Squares of Residuals:', ss_res)
        plt.figure()
        #plt.plot(x,np.log10(f))
        plt.plot(x,np.log10(f) - f_fit)
        plt.xlim(0,200)
        plt.ylim(-4,2)
        #plt.yscale("log")
        plt.savefig("polynomial_fit.png")


    def add_extra_peaks(self,num_extra):
        extra_peaks_freq = np.linspace(9,25,num_extra)
        extra_peaks_amp = np.ones_like(extra_peaks_freq)
        self.prior_frequencies = np.concatenate((self.prior_frequencies,extra_peaks_freq))
        self.prior_amplitudes = np.concatenate((self.prior_amplitudes,extra_peaks_amp))

    def lorentz_function(self,x,loc,amp,fwhm):
        fwhm = fwhm/2
        return amp*(fwhm**2)/(fwhm**2 + (x-loc)**2)
    def mcmc_fit(self,x,f,n_peaks):
        print("starting mcmc")
        with pm.Model() as model:
            loc_dist = []
            amplitude_dist = []
            fwhm_dist = []
            mu = np.zeros(len(x))
            for peak in range(n_peaks):
                if self.prior_frequencies is None:
                    mu_s = x[int((1/(n_peaks+1))*(peak+1)*len(x))]
                else:
                    mu_s = self.prior_frequencies[peak]
                loc_i = pm.Normal(f"loc_{peak}",mu_s,sigma = 1)
                loc_dist.append(loc_i)
                A_max = np.max(self.f)
                if self.prior_amplitudes is None:
                    amp_i = pm.HalfNormal(f"amp_{peak}", sigma=1)

                else:
                    amp0 = self.prior_amplitudes[peak]
                    amp_i = pm.LogNormal(
                            f"amp_{peak}",
                            mu=np.log(max(amp0, 1e-6)),
                            sigma=0.5
                            )
                amplitude_dist.append(amp_i)
                fwhm_i = pm.HalfNormal(f'fwhm_{peak}',sigma = 1)

                mu_i = self.lorentz_function(x,loc_i,amp_i,fwhm_i)
                mu+=mu_i
            sigma = pm.HalfNormal("sigma",sigma = 1)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=f)

        with model:
            trace = pm.sample(self.n_draws, tune=self.n_tune, chains = 4, cores = 4, target_accept=0.9)

        return trace
    def save_results(self,results_dir,title):
        x_fit, f_fit = generate_test_spectra(self.x, self.loc_array,self.fwhm_array,self.amp_array,False)
        directory = results_dir + f"{title}_full_freq_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}/"
        Path(directory).mkdir(parents=True, exist_ok=True)
        np.savetxt(directory + f"{title}_loc_array_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.txt",self.loc_array)
        np.savetxt(directory + f"{title}_fwhm_array_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.txt",self.fwhm_array)
        np.savetxt(directory + f"{title}_amp_array_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.txt",self.amp_array)
        np.savetxt(directory + f"{title}_full_freq_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.txt",self.x)
        np.savetxt(directory + f"{title}_spectra_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.txt",self.f)
        np.savetxt(directory + f"{title}_x_fit_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.txt",x_fit)
        np.savetxt(directory + f"{title}_f_fit_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.txt",f_fit)
        self.trace.to_netcdf(directory + f"{title}_trace_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.nc")
    def visualize_fit(self,x,trace,n_peaks):
        loc_array = []
        amp_array = []
        fwhm_array = []
        for peak in range(n_peaks):
            loc_i = trace.posterior[f'loc_{peak}'].mean(("chain", "draw")).values
            amp_i = trace.posterior[f'amp_{peak}'].mean(("chain", "draw")).values
            fwhm_i = trace.posterior[f'fwhm_{peak}'].mean(("chain", "draw")).values

            loc_array.append(loc_i)
            amp_array.append(amp_i)
            fwhm_array.append(fwhm_i)
        return loc_array, fwhm_array,amp_array


"""
parser = argparse.ArgumentParser()
parser.add_argument("--n_peaks", type = int,default = 15)
parser.add_argument("--n_draws", type = int, default = 1000)
parser.add_argument("--n_tune", type = int, default = 1000)
parser.add_argument("--data_file", type = str, default = "/home/c4052420/spectra_analysis/Initialdataforbetsy.npz")
args = parser.parse_args()

#data = frequencies_from_file(args.data_file)
data = Read_CompositeSpectrum("/home/c4052420/spectra_analysis/xc0.5-nofield-composite.npz")

fit = fit_lorentzian_mcmc(args.n_draws,args.n_tune,data.fullfreq,data.spectra,len(data.frequencies),prior_frequencies = data.frequencies,prior_amplitudes = data.amps)
#fit = fit_lorentzian_mcmc(args.n_draws,args.n_tune,data.fullfreq,data.spectra,args.n_peaks,prior_frequencies = data.frequencies,prior_amplitudes = data.amps)
fit_analysis_plots(fit.n_draws,fit.n_tune,fit.n_peaks)

"""