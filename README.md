## Software Requirements ##

### Python Packages ###
| Software  | Version |
| ------------- | ------------- |
| `Python`  |  `>=3.6`  |
| `Tensorflow`  | `2.0.0`  |
| `smac`  | `0.11.1` |

### R Packages ###
* smooth
* MASS
* forecast

## Path Variables ##

Set the `PYTHONPATH` env variable of the system. Append absolute paths of both the project root directory and the directory of the `external_packages/cocob_optimizer` into the `PYTHONPATH`  

For R scripts, make sure to set the working directory to the project root folder.

## Preprocessing the Data ##
The data files used for the NN5 experiments are available as a Google Drive folder at this [link](https://drive.google.com/drive/folders/1kp--LAlG6kXbcjjkUYgJGfFsGowkXzF_?usp=sharing)

#### Create the text files of the data ####

Three files need to be created for every model, one per training, validation and testing. Example preprocessing scripts to create text data files are in `preprocess_scripts` directory.

Sample Record of validation file in moving window format:

`1|i -0.120404761911659 -0.138029849544217 -0.158262315258994 -0.117573911196581 -0.047514354625692 -0.054921000563831 -0.087502195404757 -0.0468948356427585 -0.0265522120279886 -0.0259454546421436 -0.0149743425531481 -0.0750882944138711 0.0300152959591582 0.037022158965339 0.0168685236725015 |o -0.0487566395401897 -0.00313169841363781 -0.0356365611845675 0.11323901494058 0.0498791083802557 -0.0222170965043569 0.0324163879281905 0.0353096916266837 0.085162256512592 0.0644041024983562 0.0970988030382305 0.100330957527596 |# 6.88534358640275 -0.00313977170055892 -0.0044384039516765 0.00223114486678285 0.00574405742601041 0.00832797755707766 0.00264786188838812 0.00821557645548867 0.0196038788714076 -0.0082329067304395 -0.0136679361428553 -0.00526828286265864 -0.0120231978314266`


`input_window_size = 15`\
`output_window_size = 12`\
`meta_data_size (normalization factor + forecast horizon seasonality values) =  1 + output_window_size = 13`

Sample Record of validation file in non moving window format:

`1|i -0.567828815938088 -0.585453903570646 -0.605686369285423 -0.56499796522301 -0.494938408652121 -0.50234505459026 -0.534926249431186 -0.494318889669188 -0.473976266054418 -0.473369508668573 -0.462398396579577 -0.5225123484403 -0.417408758067271 -0.41040189506109 -0.430555530353928 -0.496180693566619 -0.450555752440067 -0.483060615210997 -0.33418503908585 -0.397544945646174 -0.469641150530786 -0.415007666098239 -0.412114362399746 -0.362261797513837 -0.383019951528073 -0.350325250988199 -0.347093096498833 -0.385629851372523 -0.33765528593077 -0.345559091168428 -0.318001882157549 -0.316676017389535 -0.352726522297433 -0.313242921446948 -0.282102294051863 -0.39715699799074 -0.295846540532483 -0.31886420750242 -0.284579499161778 -0.327664544654104 -0.291079231868399 -0.265463866077922 -0.301299949523727 -0.248215274184239 -0.257073832944534 -0.232876174866125 -0.274657317980361 -0.235741883456098 -0.293700188394566 -0.257319750871742 -0.227493242898714 -0.182332265822538 -0.226778708376941 -0.183411643917652 -0.175304442585818 -0.185429704907653 -0.174450279897044 -0.165652924361955 -0.164673044962693 -0.153843029747357 -0.153398758376659 -0.173864098477157 -0.175699109712323 -0.149000031675115 -0.118212150653881 -0.125336123838077 -0.11559266124817 -0.162446928605872 -0.142920770031087 -0.182453963932645 -0.0843017018306496 -0.0400239133080627 -0.123235344673406 -0.108910774407422 -0.0955200648753625 -0.074071004750234 -0.0886897595416567 -0.107996723281453 -0.0489798864064088 -0.0897408886572117 -0.0556907895876506 -0.113434652776987 -0.0662362512991423 -0.0736203504006339 -0.0543147221095195 -0.0214098332257642 -0.00792657237932826 0.00371722858574586 -0.0390213828131403 -0.0522027300921888 -0.129052228394151 -0.072525671787675 0.00452387954346101 -0.00802511272308681 -0.0258166978136511 0.0131909044091891 |o 0.0296250975171493 0.0263897446789843 0.0333614516505714 0.0821482410736927 0.000100754260286884 0.0588968744959812 -0.00878589639917671 0.0440999722838731 0.0424870606078542 0.0765977431805602 -0.00316200785091869 0.0483167388332202 |# 7.33276764042918 -0.0136679361428553 -0.00526828286265864 -0.0120231978314266 -0.00313977170055892 -0.0044384039516765 0.00223114486678285 0.00574405742601041 0.00832797755707766 0.00264786188838812 0.00821557645548867 0.0196038788714076 -0.0082329067304395`

Sample Record of validation file in moving window format without STL Decomposition:

`1|i -0.376703850726204 -0.385929285078564 -0.41291666576211 -0.363344835568828 -0.294583911249058 -0.295321008368737 -0.324389290650435 -0.28119801075737 -0.26653550281129 -0.260361030858344 -0.238001616353429 -0.325952353816 -0.226283792855386 -0.210877276569009 -0.237785826830614 |o -0.294527563912438 -0.250201255037004 -0.276036568989474 -0.123648080305099 -0.184424066734356 -0.262200387287658 -0.20199918828801 -0.187717582173598 -0.165701802889537 -0.191894986316188 -0.150800632496117 -0.15432339297552 |# 1246.35022471111`

`meta_data_size (normalization factor) =  1 `
#### Create the tfrecord files of the data ####

For faster execution, the text data files are converted to tfrecord binary format. The `tfrecords_handler` module converts the text data into tfrecords format (using `tfrecord_writer.py`) as well as reads in tfrecord data (using `tfrecord_reader.py`) during execution. Example scripts to convert text data into tfrecords format can be found in the `preprocess_scripts` directory.
## Execution Instructions ##

Example bash scripts are in the directory `utility_scripts/execution_scripts`. 

#### External Arguments ####
The model expects a number of arguments.
1. dataset_name - Any unique string for the name of the dataset
2. contain_zero_values - Whether the dataset contains zero values(0/1)
3. address_near_zero_instability - Whether the dataset contains zero values(0/1) - Whether to use a custom SMAPE function to address near zero instability(0/1). Default value is 0
4. integer_conversion - Whether to convert the final forecasts to integers(0/1). Default is 0
5. initial_hyperparameter_values_file - The file for the initial hyperparameter range configurations
6. binary_train_file_train_mode - The tfrecords file for train dataset in the training mode
7. binary_valid_file_train_mode - The tfrecords file for validation dataset in the training mode
8. binary_train_file_test_mode - The tfrecords file for train dataset in the testing mode
9. binary_test_file_test_mode - The tfrecords file for test dataset in the testing mode
10. txt_test_file - The text file for test dataset
11. actual_results_file - The text file of the actual results
12. original_data_file - The text file of the original dataset with all the given data points
13. cell_type - The cell type of the RNN(LSTM/GRU/RNN). Default is LSTM
14. input_size - The input size of the moving window. Default is 0 in the case of non moving window format
15. seasonality_period - The seasonality period of the time series
16. forecast_horizon - The forecast horizon of the dataset
17. optimizer - The type of the optimizer(cocob/adam/adagrad)
18. without_stl_decomposition - Whether not to use stl decomposition(0/1). Default is 0
19. no_of_series - The number of series of the dataset.

#### Execution Flow ####

##### Invoking the Script #####
The first point of invoking the models is the `generic_model_handler.py`. The `generic_model_handler.py` parses the external arguments and identifies the required type of optimizer, cell etc... The actual stacking model is inside the directory `rnn_architectures`. 
First, the hyperparameter tuning is carried out using the validation errors of the stacking model. Example initial hyperparameter ranges can be found inside the directory `configs/initial_hyperparameter_values`. The found optimal hyperparameter combination is  written to a file in the directory `results/nn_model_results/rnn/optimized_configurations`. 
Then the found optimal hyperparameter combination is used on the respective model to generate the final forecasts. Every model is run on 10 Tensorflow graph seeds (from 1 to 10). The forecasts are written to 10 files inside the directory `results/nn_model_results/rnn/forecasts`.

##### Ensembling Forecasts #####
The forecasts from the 10 seeds are ensembled by taking the median. The `utility_scripts/ensembling_forecasts.py` script does this. This script is invoked implicitly inside the `generic_model_handler.py`. The ensembled forecasts are written to the directory `results/nn_model_results/rnn/ensemble_forecasts`.

##### Error Calculation #####
The SMAPE and MASE errors are calculated per each series for each model using the error calcualtion scripts in the directory `error_calculator`. The name of the script is `final_evaluation.R`. This script is also implicitly invoked inside the `generic_model_handler.py`. The script perform the post processing of the forecasts to reverse initial preprocessing. The errors of the ensembles are written to the directory `results/nn_model_results/rnn/ensemble_errors`. 
