# python data_helpers.py -dataset="twitter" && \
python hybrid_evaluation.py \
    "twitter" "twitter" "train" -gru_num_units=128 -init_learning_rate=1 -margin=1 -batch_norm=True
