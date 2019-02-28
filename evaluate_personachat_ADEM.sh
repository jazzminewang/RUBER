

python data_helpers.py -dataset="personachat" && \
#python create_persona_validation_set.py && \
#python adem_data_helpers.py && \
python hybrid_evaluation.py \
    "personachat" "ADEM" "validate" \
    --reply_file "hred_replies.txt"
    # "de_replies.txt"
    # 'tfidf_replies.txt'
    # "human_replies.txt"
    # "hred_replies.txt"
