modelnames=("lr" "nr" "fr2" "brn" "nrbrn1" "fr2brn1" "a214" "a514")
peaks_extra=("None" "3" "5" "10")

for mn in "${modelnames[@]}"; do
    for pks in "${peaks_extra[@]}"; do 
        filename="./configs/config_${mn}_n_peaks:${pks}.ini"
        cat <<EOF > "$filename"

[settings]
# COMPOSITE SETTINGS
composite = true 
nnew = 1024 
ntotal = 722 
numprocs = 30
time_step = 1000
bg_file = /home/c4052420/spectra_analysis_pipeline/data/2MS_50_90.dat
vzc_file = /home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000-5000-44000et.dat
data_dir = /home/c4052420/spectra_analysis_pipeline/data/
modelname = ${mn}

#MCMC SETTINGS
n_peaks = None
peaks_extra = ${pks}
n_draws = 2
n_tune= 2
results_dir = /home/c4052420/spectra_analysis_pipeline/results/
run_title = None

EOF

        echo "Generated $filename"
        done
    done
          