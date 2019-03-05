#!/bin/bash

declare -a gru_num_units=(64 128 256 512)
# declare -a init_learning_rate=(0.001 0.0015 0.01 0.015 0.1 0.15)
# declare -a margins=(0.1 0.25 0.5 0.75)
declare -a init_learning_rate=(1 10 15 100 150) #(/1000)
declare -a margins=(10 25 50 75) #(/100)

# stuff
echo before for lop
for gru_num_unit in ${gru_num_units[@]}
do
    for init_learning_rate in ${init_learning_rate[@]}
    do
        for margin in ${margins[@]}
        do
            tmux new-session -d -s personachat_gru_${gru_num_unit}_learning_${init_learning_rate}_margin_${margin}_batchnorm \
                "python hybrid_evaluation.py personachat personachat train -gru_num_units=${gru_num_unit} -init_learning_rate=${init_learning_rate} -margin=${margin} -batch_norm=True"
	    echo "python hybrid_evaluation.py personachat personachat train -gru_num_units=${gru_num_unit} -init_learning_rate=${init_learning_rate} -margin=${margin} -batch_norm=True"

            tmux new-session -d -s twitter_gru_${gru_num_unit}_learning_${init_learning_rate}_margin_${margin}_batchnorm \
                "python hybrid_evaluation.py twitter twitter train -gru_num_units=${gru_num_unit} -init_learning_rate=${init_learning_rate} -margin=${margin} -batch_norm=True" 

            tmux new-session -d -s personachat_gru_${gru_num_unit}_learning_${init_learning_rate}_margin_${margin} \
                "python hybrid_evaluation.py personachat personachat train -gru_num_units=${gru_num_unit} -init_learning_rate=${init_learning_rate} -margin=${margin}" 

            tmux new-session -d -s twitter_gru_${gru_num_unit}_learning_${init_learning_rate}_margin_${margin} \
                "python hybrid_evaluation.py twitter twitter train -gru_num_units=${gru_num_unit} -init_learning_rate=${init_learning_rate} -margin=${margin}" \
        
        done
    done
done
