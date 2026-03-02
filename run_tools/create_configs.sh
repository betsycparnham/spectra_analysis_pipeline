modelnames=("lr" "nr" "fr2" "brn" "nrbrn1" "fr2brn1" "a214" "a514" "fr2mag")
peaks_extra=("None" "3" "5" "10")


declare -A vzc_files
vzc_files["lr"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000-5000-44000et.dat"
vzc_files["nr"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000nr-5000-35000et.dat"
vzc_files["fr2"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000fr2-5000-25000et.dat"
vzc_files["brn"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000brn-5000-27000et.dat"
vzc_files["nrbrn1"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000nrbrn1-5000-21000et.dat"
vzc_files["fr2brn1"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000fr2brn1-2000-23000et.dat"
vzc_files["a214"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000a214-5000-27000et.dat"
vzc_files["a514"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000a514-5000-28000et.dat"
vzc_files["fr2mag"]="/home/c4052420/spectra_analysis_pipeline/data/vzcr-rad-2ms5000fr2mag-2000-32000et.dat"

for mn in "${modelnames[@]}"; do
    for pks in "${peaks_extra[@]}"; do
        filename="./configs/config_${mn}_n_peaks:${pks}.ini"
        vzc="${vzc_files[$mn]}"

        cat <<EOF > "$filename"

[settings]
# COMPOSITE SETTINGS
composite = true 
nnew = 1024 
ntotal = 722 
numprocs = 30
time_step = 1000
bg_file = /home/c4052420/spectra_analysis_pipeline/data/2MS_50_90.dat
vzc_file = ${vzc}
data_dir = /home/c4052420/spectra_analysis_pipeline/data/
modelname = ${mn}

#MCMC SETTINGS
n_peaks = None
peaks_extra = ${pks}
n_draws = 2000
n_tune= 2000
results_dir = /home/c4052420/spectra_analysis_pipeline/results/
run_title = None

EOF

        echo "Generated $filename"
        done
    done
          