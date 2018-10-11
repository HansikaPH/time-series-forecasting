import numpy as np

def performrollingwindowtrain(dataItem, input_size, output_size, log_transform, price_factor):
    if (log_transform):
        dataItem['scaled_sales'] = logtransform(dataItem.scaled_sales)
    dataItem = dataItem.reset_index(drop=True)
    ts_length = dataItem.shape[0]
    ts_dimension = dataItem.shape[1]
    item_id = dataItem['item_id'].iloc[0]
    training_ts_length = ts_length - (output_size * 2)

    print("Processing" + " " + str(item_id))

    window_index = 0
    time_series_windows = list()
    for index in range(input_size, training_ts_length):
        sales_window_mean = np.mean(dataItem.loc[window_index:(index - 1), 'scaled_sales'])
        window_index += 1

        time_series_id = [str(item_id) + '|i']

        input_start_index = (index - input_size)
        input_end_index = index

        output_start_index = index
        output_end_index = index + output_size

        one_hot_encode_vector = (dataItem.iloc[input_start_index:input_end_index, 7:ts_dimension]).values
        one_hot_encode_vector = list(itertools.chain(*one_hot_encode_vector))

        time_series_input_vector = (dataItem.iloc[input_start_index:input_end_index, 1]).values - sales_window_mean
        time_series_output_vector = (dataItem.iloc[output_start_index:output_end_index, 1]).values - sales_window_mean

        output_id = ['|o']

        if (price_factor):
            time_series_price_vector = (dataItem.iloc[input_start_index:input_end_index, 2]).values
            window = list(itertools.chain(time_series_id, one_hot_encode_vector, time_series_price_vector,
                                          time_series_input_vector, output_id,
                                          time_series_output_vector))

        window = list(itertools.chain(time_series_id, one_hot_encode_vector, time_series_input_vector, output_id,
                                      time_series_output_vector))

        window_df = pd.DataFrame(np.array(window).reshape(1, len(window)))
        time_series_windows.append(window_df)

    time_series = pd.concat(time_series_windows, ignore_index=True)
    return time_series
