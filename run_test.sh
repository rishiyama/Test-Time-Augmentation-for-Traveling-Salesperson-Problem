# BEAM_WIDTH=2500
BATCH_SIZE=10
SEARCH=beamsearch
# SEARCH=greedy

X_TF=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer-default.pkl
X_TF_PE=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer_pe-default.pkl
DT_TF=checkpoint_epoch100/checkpoint_n50-gpu0-x_double_tree-transformer-default.pkl
DT_TF_PE=checkpoint_epoch100/checkpoint_n50-gpu0-x_double_tree-transformer_pe-default.pkl

X_TF_MAT=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer_matrix-default.pkl
X_TF_PE_MAT=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer_matrix_pe-default.pkl
DT_TF_MAT=checkpoint_epoch100/checkpoint_n50-gpu0-x_double_tree-transformer_matrix-default.pkl
DT_TF_PE_MAT=checkpoint_epoch100/checkpoint_n50-gpu0-x_double_tree-transformer_matrix_pe-default.pkl

# make array of path
CHECKPOINTS=($X_TF)
BEAM_WIDTH_ARR=(2500)


# all checkpoints do beam search
PYTHON_FILE=test_dataset.py
# PYTHON_FILE=test_dataset_tsp50_high_dim.py
for BEAM_WIDTH in ${BEAM_WIDTH_ARR[@]}
do
    for CHECKPOINT in ${CHECKPOINTS[@]}
    do
        echo $CHECKPOINT

        python3 $PYTHON_FILE --check_point=$CHECKPOINT \
            --nb_nodes=50 --beam_width=$BEAM_WIDTH --batch_size=$BATCH_SIZE --$SEARCH
        
        # python3 test_dataset.py --check_point=$CHECKPOINT \
        #     --data_type=x_double_tree --nb_nodes=50 --beam_width=$BEAM_WIDTH --batch_size=$BATCH_SIZE --$SEARCH
    done
done
