modelnames=("lr" "nr" "fr2" "brn" "nrbrn1" "fr2brn1" "a214" "a514")
model_peaks=("31" "94" "none" "41" "41" "1" "45" "36")

i=0
for mn in "${modelnames[@]}"; do
    i=$((i+1))
    echo "$i"
    mp=${model_peaks[$((i-1))]}
    if [ -d "/home/c4052420/spectra_analysis_pipeline/results/${mn}_full_freq_d2000_t2000_peaks${mp}" ]; then
        python3 code/plotting/MCMC_plot.py --n_peaks $mp --n_draws 2000 --n_tune 2000 --title "$mn" --results_dir /home/c4052420/spectra_analysis_pipeline/results/
    else
        echo "/home/c4052420/spectra_analysis_pipeline/results/${mn}_full_freq_d2000_t2000_peaks${mp}" 
    fi
done