#!/usr/bin/env bash

#with windowing

#stacking_smac_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/moving_window/stl_12i15.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/moving_window/stl_12i15v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/moving_window/cif12test.tfrecords --txt_test_file datasets/text_data/CIF_2016/moving_window/cif12test.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --input_size 15 --forecast_horizon 12 --optimizer cocob --hyperparameter_tuning smac --model_type stacking &

#stacking_smac_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/moving_window/stl_12i15.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/moving_window/stl_12i15v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/moving_window/cif12test.tfrecords --txt_test_file datasets/text_data/CIF_2016/moving_window/cif12test.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --input_size 15 --forecast_horizon 12 --optimizer adagrad --hyperparameter_tuning smac --model_type stacking &

#stacking_smac_adam
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/moving_window/stl_12i15.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/moving_window/stl_12i15v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/moving_window/cif12test.tfrecords --txt_test_file datasets/text_data/CIF_2016/moving_window/cif12test.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --input_size 15 --forecast_horizon 12 --optimizer adam --hyperparameter_tuning smac --model_type stacking &

#stacking_bayesian_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/moving_window/stl_12i15.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/moving_window/stl_12i15v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/moving_window/cif12test.tfrecords --txt_test_file datasets/text_data/CIF_2016/moving_window/cif12test.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --input_size 15 --forecast_horizon 12 --optimizer cocob --hyperparameter_tuning bayesian --model_type stacking &

#stacking_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/moving_window/stl_12i15.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/moving_window/stl_12i15v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/moving_window/cif12test.tfrecords --txt_test_file datasets/text_data/CIF_2016/moving_window/cif12test.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --input_size 15 --forecast_horizon 12 --optimizer adagrad --hyperparameter_tuning bayesian --model_type stacking &

#stacking_bayesian_Fadam
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/moving_window/stl_12i15.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/moving_window/stl_12i15v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/moving_window/cif12test.tfrecords --txt_test_file datasets/text_data/CIF_2016/moving_window/cif12test.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --input_size 15 --forecast_horizon 12 --optimizer adam --hyperparameter_tuning bayesian --model_type stacking &

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
#without windowing

#attention_smac_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer cocob --hyperparameter_tuning smac --model_type attention &

#attention_smac_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adagrad --hyperparameter_tuning smac --model_type attention &

#attention_smac_adam
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adam --hyperparameter_tuning smac --model_type attention &

#attention_bayesian_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer cocob --hyperparameter_tuning bayesian --model_type attention &

#attention_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adagrad --hyperparameter_tuning bayesian --model_type attention &

#attention_bayesian_adam
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adam --hyperparameter_tuning bayesian --model_type attention &

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

#seq2seq_smac_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_smac_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_smac_adam
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adam --hyperparameter_tuning smac --model_type seq2seq &

#seq2seq_bayesian_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer cocob --hyperparameter_tuning bayesian --model_type seq2seq &

#seq2seq_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adagrad --hyperparameter_tuning bayesian --model_type seq2seq &

#seq2seq_bayesian_adam
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adam --hyperparameter_tuning bayesian --model_type seq2seq &

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

#seq2seqwithdenselayer_smac_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seqwithdenselayer &

#seq2seqwithdenselayer_smac_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seqwithdenselayer &

#seq2seqwithdenselayer_smac_adam
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adam --hyperparameter_tuning smac --model_type seq2seqwithdenselayer &

#seq2seqwithdenselayer_bayesian_cocob
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer cocob --hyperparameter_tuning bayesian --model_type seq2seqwithdenselayer &

#seq2seqwithdenselayer_bayesian_adagrad
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adagrad --hyperparameter_tuning bayesian --model_type seq2seqwithdenselayer &

#seq2seqwithdenselayer_bayesian_adam
python ./generic_model_trainer.py --dataset_name cif2016_O12 --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/cif2016_o12 --binary_train_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12.tfrecords --binary_valid_file datasets/binary_data/CIF_2016/non_moving_window/cif_stl_12v.tfrecords --binary_test_file datasets/binary_data/CIF_2016/non_moving_window/cif_test_12.tfrecords --txt_test_file datasets/text_data/CIF_2016/non_moving_window/cif_test_12.txt --actual_results_file datasets/text_data/CIF_2016/cif_results_o12.txt --forecast_horizon 12 --optimizer adam --hyperparameter_tuning bayesian --model_type seq2seqwithdenselayer &

wait