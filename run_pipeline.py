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
        compute_composite_spectra(c["data_dir"],c["bg_file"],c["vzc50_file"],c["vzs50_file"],c["nnew"],c["ntotal"],c["numprocs"],c["time_step"])
        data = Read_CompositeSpectrum(c["data_dir"] + "xc0.5-nofield-composite.npz")
    else:
        #### insert other function here
        pass
    fit = fit_lorentzian_mcmc(c["results_dir"],c["run_title"],int(c["n_draws"]),int(c["n_tune"]),data.fullfreq,data.spectra,len(data.frequencies),prior_frequencies = data.frequencies,prior_amplitudes = data.amps)

