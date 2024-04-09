# Description: run test with TTA (test time augmentation)


BEAM_WIDTH=1
BATCH_SIZE=1

X_TF=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer-default.pkl
X_TF_PE=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer_pe-default.pkl
DT_TF=checkpoint_epoch100/checkpoint_n50-gpu0-x_double_tree-transformer-default.pkl
DT_TF_PE=checkpoint_epoch100/checkpoint_n50-gpu0-x_double_tree-transformer_pe-default.pkl

X_TF_MAT=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer_matrix-default.pkl
X_TF_PE_MAT=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer_matrix_pe-default.pkl
DT_TF_MAT=checkpoint_epoch100/checkpoint_n50-gpu0-x_double_tree-transformer_matrix-default.pkl
DT_TF_PE_MAT=checkpoint_epoch100/checkpoint_n50-gpu0-x_double_tree-transformer_matrix_pe-default.pkl

# make array of path
# CHECKPOINTS=($X_TF $X_TF_PE $DT_TF $DT_TF_PE $X_TF_MAT $X_TF_PE_MAT $DT_TF_MAT $DT_TF_PE_MAT)

# CHECKPOINTS=($X_TF_PE $DT_TF $DT_TF_PE $X_TF_MAT)
CHECKPOINTS=($X_TF_PE)

PERM_METHOD=random
NUM_K=None

# BATCH_SIZE_PERM=100
# BATCH_SIZE_PERM_ARR=(8 16  32 64 128 256 512 1024 2048 4096 8192)

BATCH_SIZE_PERM_ARR=(2500)

# BATCH_SIZE_PERM_ARR=(2)

## eg. 
# python3 test_dataset_tta.py --check_point=checkpoint_epoch100/checkpoint_n50-gpu0-x-transformer-default.pkl --data_type=x --nb_nodes=50 --beam_width=1 --batch_size=1 --greedy --batch_size_perm=100 --perm_method=random


# PYTHON_FILE=test_dataset_tta_tsp50_rotate_analyze.py
PYTHON_FILE=test_dataset_tta.py
# PYTHON_FILE=test_dataset_tta_save_feature.py


# all checkpoints do beam search
for CHECKPOINT in ${CHECKPOINTS[@]}
do
    echo $CHECKPOINT

    for BATCH_SIZE_PERM in ${BATCH_SIZE_PERM_ARR[@]}
    do 
        echo "BATCH_SIZE_PERM: $BATCH_SIZE_PERM"
        
        python3 $PYTHON_FILE --check_point=$CHECKPOINT \
            --data_type=x --nb_nodes=50 --beam_width=$BEAM_WIDTH --batch_size=$BATCH_SIZE --greedy \
            --batch_size_perm=$BATCH_SIZE_PERM \
            --perm_method=$PERM_METHOD #--num_k=$NUM_K


        # python3 $PYTHON_FILE --check_point=$CHECKPOINT \
        #     --data_type=x_double_tree --nb_nodes=50 --beam_width=$BEAM_WIDTH --batch_size=$BATCH_SIZE --greedy \
        #     --batch_size_perm=$BATCH_SIZE_PERM \
        #     --perm_method=$PERM_METHOD #--num_k=$NUM_K
    done

    echo "==================== done ===================="
done
