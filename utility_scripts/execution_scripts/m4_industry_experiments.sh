#!/usr/bin/env bash

# run the models multiple times to check for the reproducibility
for i in {1..1}

do
    echo Iteration $i of m4_industry
    #### with windowing

    #stacking_moving_window_smac_cocob
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type stacking --input_format moving_window --seed $i 

    #stacking_moving_window_smac_adagrad
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type stacking --input_format moving_window --seed $i 

    #stacking_moving_window_smac_adam
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adam --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type stacking --input_format moving_window --seed $i 

    ##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

    ##seq2seq_moving_window_smac_cocob
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq --input_format moving_window --seed $i
    #
    ##seq2seq_moving_window_smac_adagrad
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq --input_format moving_window --seed $i
    #
    ##seq2seq_moving_window_smac_adam
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type seq2seq --input_format moving_window --seed $i
    #
    ###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

    #seq2seqwithdenselayer_moving_window_smac_cocob
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seqwithdenselayer --input_format moving_window --seed $i 

    #seq2seqwithdenselayer_moving_window_smac_adagrad
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seqwithdenselayer --input_format moving_window --seed $i 

    #seq2seqwithdenselayer_moving_window_smac_adam
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adam --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type seq2seqwithdenselayer --input_format moving_window --seed $i 

    ##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

    ##seq2seq_moving_window_one_input_per_step_smac_cocob
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq --input_format moving_window_one_input_per_step --seed $i
    #
    ##seq2seq_moving_window_one_input_per_step_smac_adagrad
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq --input_format moving_window_one_input_per_step --seed $i
    #
    ##seq2seq_moving_window_one_input_per_step_smac_adam
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type seq2seq --input_format moving_window_one_input_per_step --seed $i
    #
    ###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

    ##attention_moving_window_smac_cocob
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type attention --input_format moving_window --seed $i
    #
    ##attention_moving_window_smac_adagrad
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type attention --input_format moving_window --seed $i
    #
    ##attention_moving_window_smac_adam
    #python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/moving_window/m4_stl_monthly_industry_18i15v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/moving_window/m4_test_monthly_industry_18i15.tfrecords --txt_test_file datasets/text_data/M4/moving_window/m4_test_monthly_industry_18i15.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --input_size 15 --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type attention --input_format moving_window --seed $i
    #
    ###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    #
    ##### without windowing

    #attention_non_moving_window_smac_cocob
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type attention --input_format non_moving_window --seed $i 

    #attention_non_moving_window_smac_adagrad
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type attention --input_format non_moving_window --seed $i 

    #attention_non_moving_window_smac_adam
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adam --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type attention --input_format non_moving_window --seed $i 

    ###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    ###without stl decomposition

#    #attention_non_moving_window_smac_cocob
#    python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/without_stl_decomposition/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type attention --input_format non_moving_window --without_stl_decomposition 1 --seed $i
#
#    #attention_non_moving_window_smac_adagrad
#    python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/without_stl_decomposition/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type attention --input_format non_moving_window --without_stl_decomposition 1 --seed $i
#
#    #attention_non_moving_window_smac_adam
#    python ./generic_model_trainer.py --dataset_name m4_industry_adagrad --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/without_stl_decomposition/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/without_stl_decomposition/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type attention --input_format non_moving_window --without_stl_decomposition 1 --seed $i

    ###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

    #seq2seq_non_moving_window_smac_cocob
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seq --input_format non_moving_window --seed $i 

    #seq2seq_non_moving_window_smac_adagrad
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seq --input_format non_moving_window --seed $i 

    #seq2seq_non_moving_window_smac_adam
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adam --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type seq2seq --input_format non_moving_window --seed $i 

    #########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

    #seq2seqwithdenselayer_non_moving_window_smac_cocob
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer cocob --hyperparameter_tuning smac --model_type seq2seqwithdenselayer --input_format non_moving_window --seed $i 

    #seq2seqwithdenselayer_non_moving_window_smac_adagrad
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adagrad --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer adagrad --hyperparameter_tuning smac --model_type seq2seqwithdenselayer --input_format non_moving_window --seed $i 

    #seq2seqwithdenselayer_non_moving_window_smac_adam
    python ./generic_model_trainer.py --dataset_name m4_industry --contain_zero_values 0 --initial_hyperparameter_values_file configs/initial_hyperparameter_values/m4_industry_adam --binary_train_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18.tfrecords --binary_valid_file_train_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_train_file_test_mode datasets/binary_data/M4/non_moving_window/m4_stl_monthly_industry_18v.tfrecords --binary_test_file_test_mode datasets/binary_data/M4/non_moving_window/m4_test_monthly_industry_18.tfrecords --txt_test_file datasets/text_data/M4/non_moving_window/m4_test_monthly_industry_18.txt --actual_results_file datasets/text_data/M4/m4_result_monthly_industry.txt --forecast_horizon 18 --optimizer adam --hyperparameter_tuning smac --model_type seq2seqwithdenselayer --input_format non_moving_window --seed $i 


#wait
done
echo All Iterations Completed!