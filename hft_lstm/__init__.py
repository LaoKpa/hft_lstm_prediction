from __future__ import print_function

from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.bricks.cost import SquaredError

from itertools import ifilter

import theano
import theano.tensor as T
import numpy as np
import pdb

import matplotlib.pyplot as plt

import pandas as pd

from converters import SlidingWindowPandasIterator, build_stream
from cost import AbsolutePercentageError
from custom_bricks import LinearLSTM

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


def find_theano_var_in_list(name, list_to_search):
    return list(ifilter(lambda l: l.name == name, list_to_search))[0]


def plot_test(x, y_hat, converter, execution_name):

    predict = theano.function([x], y_hat)

    stream_train, stream_test = converter.get_streams()
    day_predict = list(stream_test.get_epoch_iterator())[0]
    fig = plt.figure(figsize=(10, 10), dpi=100)
    idxs = list(converter.get_test_scheme().get_request_iterator())[0]
    day = converter.pd_data.iloc[idxs]

    y = predict(day_predict[0])

    y = y.reshape(y.shape[0])

    if converter.scheme == 'window':
        plt.plot(day['Time'], day['Close'] * converter.maxmin['Close'] + converter.mean['Close'], 'r',
                 day['Time'], y, 'b')
    else:
        plt.plot(day['Date'], day['Close'] * converter.maxmin['Close'] + converter.mean['Close'], 'r',
                 day['Date'], y, 'b')

    fig.savefig(execution_name + '.png')
    plt.close(fig)


def main(save_path, data_path, lstm_dim, batch_size, num_epochs):
    # The file that contains the model saved is a concatenation of information passed
    if save_path.endswith('//') is False:
        save_path += '//'

    # company = 'petr'
    company = 'vale'

    report_file = open("report", "a+")

    datafile = 'dados_{}.csv'.format(company)

    sw_iterator = SlidingWindowPandasIterator(data_path=data_path + datafile,
                                              normalize=True,
                                              normalize_target=False,
                                              train_size=pd.Timedelta(days=21),
                                              test_size=pd.Timedelta(days=7),
                                              step_size=pd.Timedelta(days=7))

    columns_list = [
                    ['Close'],  # 1
                    # ['Close', 'High', 'Low'],  # 2
                    # ['Close', 'Open'],  # 3
                    # ['Close', 'High', 'Low', 'Open'],  # 4
                    # ['Close', 'Qty'],  # 5
                    # ['Close', 'Open', 'Qty'],  # 6
                    # ['Close', 'High', 'Low', 'Open', 'Qty'],  # 7
                    # ['Close', 'Vol'],  # 8
                    # ['Close', 'High', 'Low', 'Open', 'Vol'],  # 9
                    # ['Close', 'Qty', 'Vol'],  # 10
                    # ['Open', 'High', 'Low', 'Close', 'Qty', 'Vol']
                    ]  # 11

    report_file.write('starting lstm training for data file {} \n'.format(datafile))

    for columns in columns_list:
        report_file.write('\n*****************************************************************************\n')
        report_file.write('Training for the following columns {}'.format(columns))

        print('Training columns: {}\n'.format(columns))

        for dimension in lstm_dim:

            avg_test_cost = 0
            number_of_sets = 0

            execution_name = 'lstm_{}_{}_{}'.format(company, len(columns), dimension)
            save_file_pre = save_path + execution_name

            report_file.write('\n***********************************\n')
            report_file.write('hidden dimension of {}\n'.format(dimension))

            print('hidden dimension of {}\n'.format(dimension))

            sw_iterator.reset()

            for train, test in sw_iterator:

                number_of_sets += 1
                best_test_cost, epochs_done = train_lstm(train, test, len(columns), dimension, columns, num_epochs,
                                                         save_file_pre + "_" + str(number_of_sets),
                                                         execution_name + "_" + str(number_of_sets))

                report_file.write('set {}: best test cost {}, epochs done {} \n'.format(number_of_sets,
                                                                                        best_test_cost,
                                                                                        epochs_done))

                print('set {}: best test cost {}, epochs done {} \n'.format(number_of_sets,
                                                                            best_test_cost,
                                                                            epochs_done))

                avg_test_cost += best_test_cost

                # due to dubious definition of the masters thesis
                if number_of_sets == 35:
                    break

                report_file.flush()

            report_file.write('number of sets {} average test cost: {}\n'.format(number_of_sets,
                                                                                 avg_test_cost / float(number_of_sets)))

            print('number of sets {} average test cost: {}\n'.format(number_of_sets,
                                                                     avg_test_cost / float(number_of_sets)))
            report_file.flush()
            # plot_test(x, y_hat, converter, save_file)

    report_file.close()

    return 0


def train_lstm(train, test, input_dim, hidden_dimension, columns, epochs, save_file, execution_name):
    stream_train = build_stream(train, columns)
    stream_test = build_stream(test, columns)

    # The train stream will return (TimeSequence, BatchSize, Dimensions) for
    # and the train test will return (TimeSequence, BatchSize, 1)

    x = T.tensor3('x')
    y = T.tensor3('y')

    y = y.reshape((y.shape[1], y.shape[0], y.shape[2]))

    # input_dim = 6
    # output_dim = 1
    linear_lstm = LinearLSTM(input_dim, 1, hidden_dimension,
                             # print_intermediate=True,
                             print_attrs=['__str__', 'shape'])

    y_hat = linear_lstm.apply(x)
    linear_lstm.initialize()

    c = AbsolutePercentageError().apply(y, y_hat)
    # c = SquaredError().apply(y, y_hat)
    c.name = 'cost'

    cg = ComputationGraph(c)

    def one_perc_min(current_value, best_value):
        if (1 - best_value / current_value) > 0.01:
            return best_value
        else:
            return current_value

    extensions = [DataStreamMonitoring(variables=[c], data_stream=stream_test, prefix='test'),
                  TrainingDataMonitoring(variables=[c], prefix='train', after_epoch=True),
                  FinishAfter(after_n_epochs=epochs),
                  # Printing(),
                  # ProgressBar(),
                  TrackTheBest('test_cost', choose_best=one_perc_min),
                  FinishIfNoImprovementAfter('test_cost_best_so_far', epochs=500)]

    # Save only parameters, not the whole main loop and only when best_test_cost is updated
    checkpoint = Checkpoint(save_file, save_main_loop=False, after_training=False)
    checkpoint.add_condition(['after_epoch'], predicate=OnLogRecord('test_cost_best_so_far'))
    extensions.append(checkpoint)

    if BOKEH_AVAILABLE:
        extensions.append(Plot(execution_name, channels=[['train_cost', 'test_cost']]))

    algorithm = GradientDescent(cost=c, parameters=cg.parameters, step_rule=Adam())
    main_loop = MainLoop(algorithm, stream_train, model=Model(c), extensions=extensions)
    main_loop.run()

    return main_loop.log.status['best_test_cost'], main_loop.log.status['epochs_done']


