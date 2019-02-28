#python data_helpers.py -dataset="personachat" && \
#python create_persona_validation_set.py && \
#python adem_data_helpers.py && \
python hybrid_evaluation.py \
    "personachat" "ADEM" "train" 
