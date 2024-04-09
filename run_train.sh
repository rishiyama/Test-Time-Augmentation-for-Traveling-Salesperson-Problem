# python3 train_dataset.py --model_type=transformer --data_type=x_double_tree --param_type=default
# python3 train_dataset.py --model_type=transformer_pe --data_type=x_double_tree --param_type=default
# python3 train_dataset.py --model_type=transformer_matrix --data_type=x_double_tree --param_type=default
# python3 train_dataset.py --model_type=transformer_matrix_pe --data_type=x_double_tree --param_type=default

python3 train_dataset.py --model_type=transformer --data_type=x --param_type=default --nb_nodes=50
python3 train_dataset.py --model_type=transformer_pe --data_type=x --param_type=default --nb_nodes=50
python3 train_dataset.py --model_type=transformer_matrix --data_type=x --param_type=default --nb_nodes=50
python3 train_dataset.py --model_type=transformer_matrix_pe --data_type=x --param_type=default --nb_nodes=50