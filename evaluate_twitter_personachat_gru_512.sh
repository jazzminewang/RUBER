python hybrid_evaluation.py \
    "twitter" "personachat" "validate" \
    -reply_files="high_quality_responses.txt kevmemnn.txt language_model.txt random_response.txt seq_to_seq.txt tf_idf.txt" \
    -checkpoint_dirs="train_data_twitter_learning_rate_0.001_margin_0.5_gru_512 train_data_twitter_learning_rate_0.1_margin_0.5_gru_512"
