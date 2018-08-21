#!/usr/bin/env bash

# with windowing

#stacking_smac_cocob
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type stacking &

#stacking_smac_adagrad
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type stacking &

#stacking_smac_adam
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type stacking &

#stacking_bayesian_cocob
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning bayesian --model_type stacking &

#stacking_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning bayesian --model_type stacking &

#stacking_bayesian_adam
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adam --hyperparameter_tuning bayesian --model_type stacking &

#attention_smac_cocob
#python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type attention &

#attention_smac_adagrad
#python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type attention &

#attention_bayesian_cocob
#python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning bayesian --model_type attention &

#attention_bayesian_adagrad
#python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13.tfrecords --binary_valid_file datasets/binary_data/NN3/moving_window/nn3_stl_18i13v.tfrecords --binary_test_file datasets/binary_data/NN3/moving_window/nn3_test_18i13.tfrecords --txt_test_file datasets/text_data/NN3/moving_window/nn3_test_18i13.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --input_size 13 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning bayesian --model_type attention &

# without windowing

#seq2seq_smac_cocob
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18.tfrecords --binary_valid_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18v.tfrecords --binary_test_file datasets/binary_data/NN3/non_moving_window/nn3_test_18.tfrecords --txt_test_file datasets/text_data/NN3/non_moving_window/nn3_test_18.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_smac_adagrad
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18.tfrecords --binary_valid_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18v.tfrecords --binary_test_file datasets/binary_data/NN3/non_moving_window/nn3_test_18.tfrecords --txt_test_file datasets/text_data/NN3/non_moving_window/nn3_test_18.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_smac_adam
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18.tfrecords --binary_valid_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18v.tfrecords --binary_test_file datasets/binary_data/NN3/non_moving_window/nn3_test_18.tfrecords --txt_test_file datasets/text_data/NN3/non_moving_window/nn3_test_18.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_bayesian_cocob
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18.tfrecords --binary_valid_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18v.tfrecords --binary_test_file datasets/binary_data/NN3/non_moving_window/nn3_test_18.tfrecords --txt_test_file datasets/text_data/NN3/non_moving_window/nn3_test_18.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning bayesian --model_type seq2seq &

#seq2seq_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18.tfrecords --binary_valid_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18v.tfrecords --binary_test_file datasets/binary_data/NN3/non_moving_window/nn3_test_18.tfrecords --txt_test_file datasets/text_data/NN3/non_moving_window/nn3_test_18.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning bayesian --model_type seq2seq &

#seq2seq_bayesian_adam
python ./generic_model_trainer.py --dataset_name nn3 --contain_zero_values 0 --initial_hyperparameter_configs_file initial_hyperparameter_configs/nn3 --binary_train_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18.tfrecords --binary_valid_file datasets/binary_data/NN3/non_moving_window/nn3_stl_18v.tfrecords --binary_test_file datasets/binary_data/NN3/non_moving_window/nn3_test_18.tfrecords --txt_test_file datasets/text_data/NN3/non_moving_window/nn3_test_18.txt --actual_results_file datasets/text_data/NN3/nn3_results.txt --forecast_horizon 18 --optimizer adam --hyperparameter_tuning bayesian --model_type seq2seq &

wait