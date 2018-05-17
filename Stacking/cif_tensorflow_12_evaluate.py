import numpy as np
import pandas as pd
import random
import math as m

import sys
import os
import csv

from cntk.device import *
from cntk import Trainer
from cntk.layers import *
from cntk.layers.typing import *
from cntk.learners import *
from cntk.ops import *
from cntk.logging import *
from cntk.metrics import *
from cntk.losses import *
from cntk.io import *
from cntk import sequence

import cntk
import cntk.ops as o
import cntk.layers as l

# from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms

# Input/Output Window size.
INPUT_SIZE = 70
OUTPUT_SIZE = 56


# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Training and Validation file paths.
train_file_path = '/home/hban0001/Documents/cntk/cntk2.2/Baysian/baysianoptimization/Code/NN5/All/nn5_validation.txt'
validate_file_path = '/home/hban0001/Documents/cntk/cntk2.2/Baysian/baysianoptimization/Code/NN5/All/nn5_validation.txt'
test_file_path = '/home/hban0001/Documents/cntk/cntk2.2/Baysian/baysianoptimization/Code/NN5/All/nn5_all_test.txt'


# Custom error measure.
def sMAPELoss(z, t):
    loss = o.reduce_mean(o.abs(t - z) / (t + z)) * 2
    return loss

def L1Loss(z, t):
    loss = o.reduce_mean(o.abs(t - z))
    return loss

# Preparing training dataset.
def create_train_data():
    listOfTuplesOfInputsLabels = []

    # Reading the training dataset.
    train_df = pd.read_csv(train_file_path, nrows=10)

    float_cols = [c for c in train_df if train_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    train_df = pd.read_csv(train_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    train_df = train_df.rename(columns={0: 'series'})

    # Returns unique number of time series in the dataset.
    series = np.unique(train_df['series'])

    # Construct input and output training tuples for each time series.
    for ser in series:
        oneSeries_df = train_df[train_df['series'] == ser]
        inputs_df = oneSeries_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        labels_df = oneSeries_df.iloc[:, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
        tup = (np.ascontiguousarray(inputs_df, dtype=np.float32), np.ascontiguousarray(labels_df, dtype=np.float32))
        listOfTuplesOfInputsLabels.append(tup)

    listOfTestTuples = []
    listOfTestInputs = []
    listOfTestLabels = []

    # Reading the validation dataset.
    val_df = pd.read_csv(validate_file_path, nrows=10)

    float_cols = [c for c in val_df if val_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    val_df = pd.read_csv(validate_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    val_df = val_df.rename(columns={0: 'series'})
    val_df = val_df.rename(columns={(INPUT_SIZE + OUTPUT_SIZE + 3): 'level'})
    series = np.unique(val_df['series'])

    for ser in series:
        oneSeries_df = val_df[val_df['series'] == ser]
        inputs_df_test = oneSeries_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        labels_df_test = oneSeries_df.iloc[:, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
        level = np.ascontiguousarray(oneSeries_df['level'], dtype=np.float32)
        level = level[level.shape[0] - 1]
        trueValues_df = oneSeries_df.iloc[
            oneSeries_df.shape[0] - 1, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
        trueSeasonailty_df = oneSeries_df.iloc[
            oneSeries_df.shape[0] - 1, range((INPUT_SIZE + OUTPUT_SIZE + 4), oneSeries_df.shape[1])]
        tup = (
            np.ascontiguousarray(inputs_df_test, dtype=np.float32), 
            np.ascontiguousarray(labels_df_test, dtype=np.float32), 
            level,
            np.ascontiguousarray(trueValues_df, dtype=np.float32),
            np.ascontiguousarray(trueSeasonailty_df, dtype=np.float32))
        listOfTestTuples.append(tup)
        listOfTestInputs.append(tup[0])
        listOfTestLabels.append(tup[1])

    #############################################
    # Reading the test file.
    testInputs = []

    test_df = pd.read_csv(test_file_path, nrows=10)

    float_cols = [c for c in test_df if test_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    test_df = pd.read_csv(test_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    test_df = test_df.rename(columns={0: 'series'})

    series1 = np.unique(test_df['series'])

    for ser in series1:
        test_series_df = test_df[test_df['series'] == ser]
        test_inputs_df = test_series_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        test_tup = (np.ascontiguousarray(test_inputs_df, dtype=np.float32))
        testInputs.append(test_tup[0])

    return listOfTuplesOfInputsLabels, listOfTestTuples, listOfTestInputs, testInputs


# Training the time series
def train_model(listofTuplesOfInputsLabels, listOfTestTuples, listOfTestInputs,testInputs):
    # gaussian_noise = 0.0004# l2_regularization_weight = 0.0005# minibatch_size = 128# test_minitbacth_size = 1
    maxNumOfEpochs =20
    maxEpochSize = 1
    learningRate = 0.0013262220421187676
    lstmCellDimension = 23
    l2_regularization = 0.00015753660121731034
    gaussianNoise = 0.00023780395225712772
    mbSize =10

    input = o.sequence.input_variable((INPUT_SIZE), np.float32)
    label = o.sequence.input_variable((OUTPUT_SIZE), np.float32)

    netout = Sequential([For(range(1), lambda i: Recurrence(
        LSTM(int(lstmCellDimension), use_peepholes=LSTM_USE_PEEPHOLES,
             enable_self_stabilization=LSTM_USE_STABILIZATION))),
                         Dense(OUTPUT_SIZE, bias=BIAS)])(input)

    ce = L1Loss(netout, label)
    em = sMAPELoss(o.exp(netout), o.exp(label))

    lr_schedule = cntk.learning_parameter_schedule(learningRate)
    learner = cntk.adagrad(netout.parameters, lr=lr_schedule, l2_regularization_weight=l2_regularization,
                           gaussian_noise_injection_std_dev=gaussianNoise)

    progress_printer = ProgressPrinter(1)
    trainer = Trainer(netout, (ce, em), learner, progress_printer)

    INFO_FREQ = 1;
    iscan = 0;
    iseries = 0;
    epochsize = 0;
    sMAPE_final_list = []

    for iscan in range(int(maxNumOfEpochs)):
        print("Epoch->", iscan)
        random.shuffle(listofTuplesOfInputsLabels)
        numberOfTimeseries = 0
        listOfInputs = [];
        listOfLabels = []

        for epochsize in range(int(maxEpochSize)):
            for isseries in range(len(listofTuplesOfInputsLabels)):
                series = listofTuplesOfInputsLabels[iseries]
                listOfInputs.append(series[0])
                listOfLabels.append(series[1])
                numberOfTimeseries += 1
                if numberOfTimeseries >= int(mbSize) or iseries == len(listofTuplesOfInputsLabels) - 1:
                    trainer.train_minibatch({input: listOfInputs, label: listOfLabels})
                    # trainer.train_minibatch(({input:listOfInputs, label:listOfLabels},newSeqList))
                    # training_loss = get_train_loss(trainer)
                    # loss.append(training_loss)
                    numberOfTimeseries = 0
                    listOfInputs = [];
                    listOfLabels = []

            if iscan % INFO_FREQ == 0:
                sMAPE_list = []
                test_output = trainer.model.eval({input: listOfTestInputs})

                for il in range(len(test_output)):
                    series = listOfTestTuples[il]
                    oneTestOut = test_output[il]
                    lastOneTestOutput = oneTestOut[oneTestOut.shape[0] - 1,]
                    trueSeasonailtyValues = series[4][range(series[4].shape[0] - OUTPUT_SIZE, series[4].shape[0])]
                    testOutputUnwind = np.exp(lastOneTestOutput + trueSeasonailtyValues + series[2])
                    trueValues = series[3][range(series[3].shape[0] - OUTPUT_SIZE, series[3].shape[0])]
                    actualValue = np.exp(trueValues + trueSeasonailtyValues + series[2])
                    sMAPE = np.mean(np.abs(testOutputUnwind - actualValue) / (testOutputUnwind + actualValue)) * 2
                    sMAPE_list.append(sMAPE)

        sMAPEprint = np.mean(sMAPE_list)
        sMAPE_final_list.append(sMAPEprint)

    # Finally applying the model to test dataset.
    test_final = trainer.model.eval({input: testInputs})

    return test_final


if __name__ == '__main__':
    cntk.device.try_set_default_device(cpu())
    np.random.seed(1)
    random.seed(1)
    cntk.cntk_py.set_fixed_random_seed(1)
    cntk.cntk_py.force_deterministic_algorithms()  # force_deterministic_algorithms(true)

    listofTuplesOfInputsLabels, listOfTestTuples, listOfTestInputs,testInputs = create_train_data()
    test_model = train_model(listofTuplesOfInputsLabels, listOfTestTuples, listOfTestInputs,testInputs)

    listOfTestOutput = []

    # Writes the last test output(i.e Forecast) of each time series to a file
    for kl in range(len(test_model)):
        seriesPrint = test_model[kl]
        listvalue = seriesPrint[seriesPrint.shape[0]-1, ]
        finallistvalue = np.array(listvalue).tolist()
        listOfTestOutput.append(finallistvalue)

    with open("forecasting.txt", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(listOfTestOutput)
