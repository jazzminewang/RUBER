#python data_helpers.py -dataset="personachat" -scramble && \
#python create_persona_validation_set.py && \
python hybrid_evaluation.py \
    "personachat" "personachat" "train" -gru_num_units=128 -init_learning_rate=1 -margin=1 -batch_norm=True -log_dir="koustuv/" -scramble=True
