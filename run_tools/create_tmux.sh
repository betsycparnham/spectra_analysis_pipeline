modelnames=("lr" "nr" "fr2" "brn" "nrbrn1" "fr2brn1" "a214" "a514")
peaks_extra=("None" "3" "5" "10")

for mn in "${modelnames[@]}"; do
    for pks in "${peaks_extra[@]}"; do 
        filename="./configs/config_${mn}_n_peaks:${pks}.ini"
        SESSION="${mn}_pks_${pks}"
        tmux new-session -d -s "$SESSION"
        tmux send-keys -t "${SESSION}:0" \
        "python3 run_pipeline.py --config $filename " C-m
    done
done 