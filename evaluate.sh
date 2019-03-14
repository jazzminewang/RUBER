declare -a gru_num_units=(64 128 256 512)
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
    session="${item//./}"
    for gru_num_unit in ${gru_num_units[@]}
    do
        if [[ $item == *$gru_num_unit*  ]]; then
            if [[ $item == *"batchnorm"* ]]; then

                tmux new-session -d -s $session \; \
                    send-keys "conda activate RUBER" Enter \; \
                    send-keys "python hybrid_evaluation.py twitter ADEM validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50 -batch_norm=True -checkpoint_dir="$item"" Enter
                tmux new-session -d -s p_$session \; \
                    send-keys "conda activate RUBER" Enter \; \
                    send-keys "python hybrid_evaluation.py twitter personachat validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50 -batch_norm=True -checkpoint_dir="$item"" Enter
            else
                tmux new-session -d -s $session \; \
                    send-keys "conda activate RUBER" Enter \; \
                    send-keys "python hybrid_evaluation.py twitter ADEM validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50 -checkpoint_dir="$item"" Enter
		tmux new-session -d -s p_$session \; \
                    send-keys "conda activate RUBER" Enter \; \
                    send-keys "python hybrid_evaluation.py twitter personachat validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50 -checkpoint_dir="$item"" Enter
            fi
        fi
    done
done

# Loop through personachat validation on personachat and ADEM, parsing out gru_num_units and batch_norm
for item in ${personachat_checkpoints[*]}
do
    item=${item#"$prefix"}
    for gru_num_unit in ${gru_num_units[@]}
    do
        if [[ $item == *$gru_num_unit*  ]]; then
            if [[ $item == *"batchnorm"* ]]; then
                tmux new-session -d -s $item \; \
                    send-keys "conda activate RUBER" Enter \; \
                    send-keys "python hybrid_evaluation.py personachat ADEM validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50 -batch_norm=True" Enter
                tmux new-session -d -s $item \; \
                    send-keys "conda activate RUBER" Enter \; \
                    send-keys "python hybrid_evaluation.py personachat personachat validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50 -batch_norm=True" Enter
            else
                tmux new-session -d -s $item \; \
                    send-keys "conda activate RUBER" Enter \; \
                    send-keys "python hybrid_evaluation.py personachat ADEM validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50" Enter
                tmux new-session -d -s $item \; \
                    send-keys "conda activate RUBER" Enter \; \
                    send-keys "python hybrid_evaluation.py personachat personachat validate -gru_num_units=${gru_num_unit} -init_learning_rate=1 -margin=50" Enter
            fi
        fi
    done
done

# array for reply files for ADEM

# array for reply files for personachat
