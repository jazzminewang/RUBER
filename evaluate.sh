declare -a gru_num_units=(64 128 256 512)
echo "ok"
# Get all checkpoint dirs with twitter in the name, put in array
cd experiments
twitter_checkpoints=($(find -name \*twitter*))

# Get all checkpoint dirs with personachat in the name, put in array
personachat_checkpoints=($(find -name \*personachat*))
echo "got checkpoints"
cd ..
prefix="./"
# Loop through twitter validation on personachat and ADEM, parsing out gru_num_units and batch_norm
for item in ${twitter_checkpoints[*]}
do

    item=${item#"$prefix"}
    for gru_num_unit in ${gru_num_units[@]}
    do
        if [[ $item == *$gru_num_unit*  ]]; then
            if [[ $item == *"batchnorm"* ]]; then
                echo $item
                echo "edited item"
                tmux new-session -d -s $item \; \
                    send-keys "conda activate RUBER" \; \
                    send-keys "python hybrid_evaluation.py twitter ADEM validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50 -batch_norm=True"
            else
                tmux new-session -d -s $item \; \
                    send-keys "conda activate RUBER" \; \
                    send-keys "python hybrid_evaluation.py twitter ADEM validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50" 
            fi
        fi
    done
done

# Loop through personachat validation on personachat and ADEM, parsing out gru_num_units and batch_norm

# array for reply files for ADEM

# array for reply files for personachat
