python hybrid_evaluation.py \
    "twitter" "ADEM" "validate" \
    -reply_files="human_replies.txt de_replies.txt tfidf_replies.txt hred_replies.txt" \
    -checkpoint_dirs="train_data_twitter_learning_rate_0.001_margin_0.5_gru_256"
