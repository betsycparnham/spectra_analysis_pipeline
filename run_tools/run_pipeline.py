import argparse
import configparser
import sys, os
from code.analysis.composite_spectra import compute_composite_spectra
from code.analysis.MCMC_fit import *
from code.plotting.MCMC_plot import *


parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str,default = "configs/config_composite.ini")
args = parser.parse_args()


cfg = configparser.ConfigParser()
print(args.config)
cfg.read(args.config)
c = cfg["settings"]

if __name__ == "__main__":
    
    if c.getboolean("composite"):
        compute_composite_spectra(c["data_dir"],c["bg_file"],c["vzc_file"],c["nnew"],c["ntotal"],c["numprocs"],c["time_step"], c["modelname"])
        data = Read_CompositeSpectrum(c["data_dir"] + f"xc0.5-{c["modelname"]}-composite.npz")
    else:
        #### insert other function here
        pass
    if c["run_title"] == "None":
        run_title = c["modelname"]
    else:
        run_title = c["run_title"]
    print("N PEAKS CONFIG",c["n_peaks"])
    print(c["peaks_extra"])
    if c["n_peaks"] == "None" and c["peaks_extra"] == "None":
        n_peaks = len(data.frequencies)
    elif c["peaks_extra"] == "None":
        n_peaks = c["n_peaks"]
    elif c["n_peaks"] != "None" and c["peaks_extra"] == "None":
        n_peaks = int(c["n_peaks"])
    elif c["n_peaks"] == "None" and c["peaks_extra"] != "None":
        n_peaks = len(data.frequencies) + int(c["peaks_extra"])
    print("N PEAKS",n_peaks)
    fit = fit_lorentzian_mcmc(c["modelname"],c["results_dir"],run_title,int(c["n_draws"]),int(c["n_tune"]),data.fullfreq,data.spectra,n_peaks,prior_frequencies = data.frequencies,prior_amplitudes = data.amps)
    MCMC_diagnostics(c["results_dir"],run_title,int(c["n_draws"]),int(c["n_tune"]),n_peaks)
    fit_analysis_plots(c["results_dir"],run_title,int(c["n_draws"]),int(c["n_tune"]),n_peaks)
