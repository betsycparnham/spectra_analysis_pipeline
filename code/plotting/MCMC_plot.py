import numpy as np
import astropy
from astropy.modeling.functional_models import Lorentz1D
import argparse
from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
import pymc as pm
import arviz as az
from pathlib import Path



    



class fit_analysis_plots():
    def __init__(self,results_dir, title, n_draws,n_tune,n_peaks,plot=True):
        self.n_peaks = n_peaks
        self.n_draws = n_draws
        self.n_tune = n_tune
        print("results dir",results_dir)
        print("title",title)
        self.title = title
        self.results_dir = results_dir
        self.load_directories(results_dir,title,n_draws,n_tune,n_peaks)
        if plot==True:
            self.plot_fits(results_dir,title,n_draws,n_tune,n_peaks,self.full_freq,self.spectra,self.x_fit,self.f_fit,zoom = True,log = True)
            self.plot_fits(results_dir,title,n_draws,n_tune,n_peaks,self.full_freq,self.spectra,self.x_fit,self.f_fit,zoom = True,log = False)
            self.plot_fits(results_dir,title,n_draws,n_tune,n_peaks,self.full_freq,self.spectra,self.x_fit,self.f_fit,zoom = False,log = False)
            self.plot_fits(results_dir,title,n_draws,n_tune,n_peaks,self.full_freq,self.spectra,self.x_fit,self.f_fit,zoom = False,log = True)


    def load_directories(self,results_dir,title,n_draws,n_tune,n_peaks):
        directory = results_dir + f"{title}_full_freq_d{n_draws}_t{n_tune}_peaks{n_peaks}/"
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.full_freq = np.loadtxt(directory + f"{title}_full_freq_d{n_draws}_t{n_tune}_peaks{n_peaks}.txt")
        self.spectra = np.loadtxt(directory + f"{title}_spectra_d{n_draws}_t{n_tune}_peaks{n_peaks}.txt")
        self.x_fit = np.loadtxt(directory + f"{title}_x_fit_d{n_draws}_t{n_tune}_peaks{n_peaks}.txt")
        self.f_fit = np.loadtxt(directory + f"{title}_f_fit_d{n_draws}_t{n_tune}_peaks{n_peaks}.txt")
        self.locs = np.loadtxt(directory + f"{title}_loc_array_d{n_draws}_t{n_tune}_peaks{n_peaks}.txt")
        self.amps = np.loadtxt(directory + f"{title}_amp_array_d{n_draws}_t{n_tune}_peaks{n_peaks}.txt")
        self.fwhms = np.loadtxt(directory + f"{title}_fwhm_array_d{n_draws}_t{n_tune}_peaks{n_peaks}.txt")

    def generate_test_spectra(self,x,loc_array,fwhm_array,amplitude_array):
        if len(x)==0:
            x = np.linspace(-np.max(abs(loc_array)+1),np.max(abs(loc_array)+1))
        f = np.zeros(len(x))
        for i,loc in enumerate(loc_array):
            f += Lorentz1D(amplitude_array[i],loc,fwhm_array[i])(x)
        return x,f

    def plot_fits(self,results_dir,title,n_draws,n_tune,n_peaks,fullfreq,spectra,x_fit,f_fit,zoom = True,log = True):
        if zoom:
            fig, axs = plt.subplots(2,1,figsize = (9,12),constrained_layout = True)
            axs.flatten()[1].set_xlabel("Frequency (mHz)")
            fig.suptitle(r"$n_{peaks}$ =" + f"{self.n_peaks}",fontsize = 15)
        else:
            fig, axs = plt.subplots(1,1,figsize = (9,3),constrained_layout = True)
            axs.set_title(r"$n_{peaks}$ =" + f"{self.n_peaks}",fontsize = 15)
        for ax in np.array([axs]).flatten():
            ax.plot(fullfreq,spectra,label = 'original spectrum')
            ax.plot(x_fit,f_fit,label = "fitted function")

            if log:
                ax.set_yscale("log")
            ax.set_xlim(0,200)
            ax.legend()
            ax.set_ylabel(r"Amplitude (cm$s^{-1}$)")

            for loc in self.locs:
                ax.plot(np.ones(100)*loc,np.linspace(0,np.max(f_fit),100),linestyle = "--",color = "red",linewidth = 0.8)

        directory = results_dir + f"{title}_full_freq_d{n_draws}_t{n_tune}_peaks{n_peaks}/"

        if log:
            if zoom:
                fig.savefig(directory + f"{title}_result_load_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}_zoom.png")
            else:
                fig.savefig(directory + f"{title}_result_load_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}.png")

        else:
            if zoom:
                fig.savefig(directory + f"{title}_result_load_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}_linear_zoom.png")
            else:
                fig.savefig(directory + f"{title}_result_load_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}_linear.png")

class MCMC_diagnostics():
    def __init__(self,results_dir, title, n_draws,n_tune,n_peaks):
        directory = results_dir + f"{title}_full_freq_d{n_draws}_t{n_tune}_peaks{n_peaks}/"
        trace = az.from_netcdf(directory + f"{title}_trace_d{n_draws}_t{n_tune}_peaks{n_peaks}.nc")
        print(az.summary(trace))
        self.locs = np.loadtxt(directory + f"{title}_loc_array_d{n_draws}_t{n_tune}_peaks{n_peaks}.txt")
        self.plot_trace(trace,results_dir, title, n_draws,n_tune,n_peaks)
    
    def plot_trace(self,trace,results_dir, title, n_draws,n_tune,n_peaks):
        plt.figure()
        peak_numbers = np.arange(1,len(self.locs))
        num_sections = np.floor(len(self.locs)/3)
        for n in range(int(num_sections)):
            l = n*3
            u = (n+1)*3
            freqs = self.locs[l:u]
            nums = peak_numbers[l:u]
            var_names = [f"loc_{i}" for i in nums]

            az.plot_trace(trace, var_names = var_names)
            plt.savefig(results_dir + f"{title}_trace_plot_d{n_draws}_t{n_tune}_peaks{n_peaks}_locs{l+1}-{u+1}.png" )


class loss_analysis():
    def __init__(self,fit_analysis):
        self.locs = fit_analysis.locs
        self.n_peaks = fit_analysis.n_peaks
        print(np.sort(self.locs))
        self.expected_p, self.expected_f = self.expectedpeaks(np.sort(self.locs)[::-1],len(self.locs))
        self.plot_comparison(self.expected_f,self.locs,fit_analysis.full_freq,fit_analysis.spectra)
    def expectedpeaks(self,foundfreqs, num_lines):
        # Takes in foundpeaks - peak frequencies in microhz going from high to low n
        peak_periods = 1 / (foundfreqs * 1e-6)  # in seconds
        peak_periods = peak_periods / (3600 * 24)  # in days
        start = peak_periods[-2] #n=-1
        print('start:', start)
        step = peak_periods[-3] - peak_periods[-2] # n=-2 - n=-1
        print('step:', step)
        expected_p = start + step * np.arange(num_lines)
        expected_f = 1 / ((expected_p * 3600 * 24)) * 1e6

        return expected_p, expected_f
    def plot_comparison(self,expected_f,locs,full_freq,spectra,zoom = True,log = True):

        if zoom:
            fig, axs = plt.subplots(2,1,figsize = (9,6),constrained_layout = True)
            axs.flatten()[1].set_xlabel("Frequency (mHz)")
            fig.suptitle(r"$n_{peaks}$ =" + f"{self.n_peaks}",fontsize = 15)
        else:
            fig, axs = plt.subplots(1,1,figsize = (9,6),constrained_layout = True)
            axs.set_title(r"$n_{peaks}$ =" + f"{self.n_peaks}",fontsize = 15)
        for ax in np.array([axs]).flatten():
            ax.plot(full_freq,spectra,label = 'original spectrum')


            if log:
                ax.set_yscale("log")
            ax.set_xlim(0,200)

            ax.set_ylabel(r"Amplitude (cm$s^{-1}$)")
            ax.legend()
            for loc in self.locs:
                ax.vlines(locs,0,np.max(spectra),color = "red", label = "Peaks Found")
                ax.vlines(expected_f,0,np.max(spectra),color = "green", linestyle = "--",label = "Expected Peaks")

        axs.flatten()[-1].set_xlim(9,25)
        fig.savefig("compartison.png")
class period_spacings():
    def __init__(self,fit_analysis: fit_analysis_plots):
        self.full_freq = fit_analysis.full_freq
        self.spectra = fit_analysis.spectra
        self.x_fit = fit_analysis.x_fit
        self.locs = fit_analysis.locs
        self.amps = fit_analysis.amps
        self.fwhms = fit_analysis.fwhms
        self.title = fit_analysis.title
        self.results_dir = fit_analysis.results_dir
        self.n_draws = fit_analysis.n_draws
        self.n_tune = fit_analysis.n_tune
        self.n_peaks = fit_analysis.n_peaks

        self.period_locs = self.spectrum_to_periodogram(self.locs)
        self.period_locs = np.sort(self.period_locs)
        del_p,period,radial_orders = self.pspmake2_period(self.period_locs)
        self.plot_period_spacings(self.title,self.results_dir,self.period_locs,del_p)

    def spectrum_to_periodogram(self,frequencies):
        # Input spectra at one radius and corresponding frequencies
        periods = 1 / (frequencies * 1e-6)  # in seconds
        P = periods / (3600 * 24)  # convert to days

        return P

    def pspmake2_period(self,period_days, nstart = 0):
        import numpy as np

        period_days = (np.asarray(period_days))

        # Period spacings (in days)
        deltap = -np.diff(period_days)

        # Reverse ordering if required
        period_r = period_days[::-1]

        radial_orders_r = np.arange(len(period_r)) + nstart
        radial_orders = -radial_orders_r[::-1]

        return deltap, period_days[1:], radial_orders[1:]
    def plot_period_spacings(self,title,results_dir,period_locs,del_p):
        directory = results_dir + f"{self.title}_full_freq_d{self.n_draws}_t{self.n_tune}_peaks{self.n_peaks}/"
        
        plt.figure()
        plt.plot(period_locs[1:],del_p,label='No Field', color='blue', marker='o', linestyle='solid')
        
        print("del_p",del_p)
        plt.ylabel(fr"Period Spacings ($days$)")
        plt.xlabel(fr"Period ($days$)")
        plt.title(fr"{self.title}")
        plt.savefig(directory + f"period_spacing_{self.title}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_peaks",default = 15)
    parser.add_argument("--n_draws", type = int, default = 1000)
    parser.add_argument("--n_tune", type = int, default = 1000)
    parser.add_argument("--results_dir",type = str)
    parser.add_argument("--title",type = str)
    args = parser.parse_args()

    fit = fit_analysis_plots(args.results_dir,args.title,args.n_draws,args.n_tune,args.n_peaks)
    period_spacings(fit)
    loss_analysis(fit)


