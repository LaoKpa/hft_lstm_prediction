from __future__ import print_function

import logging
from logging import CRITICAL

from blocks.algorithms import GradientDescent, Adam, RMSProp
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint, load_parameters
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.monitoring.evaluators import DatasetEvaluator
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
from datetime import datetime

from converters import SlidingWindowPandasIterator, build_stream
from cost import AbsolutePercentageError
from custom_bricks import LinearLSTM

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logging.disable(logging.CRITICAL)


def main(save_path, data_path, lstm_dim, columns, batch_size, num_epochs, plot):
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

    columns_dict = {
                    1: ['Close'],  # 1
                    2: ['Close', 'High', 'Low'],  # 2
                    3: ['Close', 'Open'],  # 3
                    4: ['Close', 'High', 'Low', 'Open'],  # 4
                    5: ['Close', 'Qty'],  # 5
                    6: ['Close', 'Open', 'Qty'],  # 6
                    7: ['Close', 'High', 'Low', 'Open', 'Qty'],  # 7
                    8: ['Close', 'Vol'],  # 8
                    9: ['Close', 'High', 'Low', 'Open', 'Vol'],  # 9
                    10: ['Close', 'Qty', 'Vol'],  # 10
                    11: ['Open', 'High', 'Low', 'Close', 'Qty', 'Vol']
    }  # 11

    columns_dict = {columns: columns_dict[columns]}

    report_file.write('starting lstm training for data file {} for batch size {}\n'.format(datafile, batch_size))

    for key, columns in columns_dict.iteritems():
        report_file.write('\n*****************************************************************************\n')
        report_file.write('Training for the following columns {}'.format(columns))

        print('Training columns: {}\n'.format(columns))

        for dimension in lstm_dim:

            avg_test_cost = 0
            number_of_sets = 0

            execution_name = 'lstm_{}_{}_{}_{}'.format(company, key, dimension, batch_size)
            save_file_pre = save_path + execution_name

            report_file.write('\n***********************************\n')
            report_file.write('hidden dimension of {}\n'.format(dimension))

            print('hidden dimension of {}\n'.format(dimension))

            sw_iterator.reset()

            for train, test in sw_iterator:

                start = datetime.now()
                print('started at {}'.format(start))

                number_of_sets += 1

                best_test_cost, epochs_done = train_lstm(train, test, len(columns), dimension, columns, num_epochs,
                                                         save_file_pre + "_" + str(number_of_sets),
                                                         execution_name + "_" + str(number_of_sets),
                                                         batch_size, plot)

                report_file.write('set {}: \t best test cost {} \tepochs done {} \n'.format(number_of_sets,
                                                                                            best_test_cost,
                                                                                            epochs_done))

                print('set {}: \t best test cost {} \t epochs done {} \n'.format(number_of_sets,
                                                                                 best_test_cost,
                                                                                 epochs_done))
                end = datetime.now()
                print('ended at {}'.format(end-start))

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


def train_lstm(train, test, input_dim,
               hidden_dimension, columns, epochs,
               save_file, execution_name, batch_size, plot):
    stream_train = build_stream(train, batch_size, columns)
    stream_test = build_stream(test, batch_size, columns)

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

    c_test = AbsolutePercentageError().apply(y, y_hat)
    c_test.name = 'mape'

    c = SquaredError().apply(y, y_hat)
    c.name = 'cost'

    cg = ComputationGraph(c_test)

    def one_perc_min(current_value, best_value):
        if (1 - best_value / current_value) > 0.01:
            return best_value
        else:
            return current_value

    extensions = []

    extensions.append(DataStreamMonitoring(variables=[c, c_test],
                                           data_stream=stream_test,
                                           prefix='test',
                                           after_epoch=False,
                                           every_n_epochs=100))

    extensions.append(TrainingDataMonitoring(variables=[c_test],
                                             prefix='train',
                                             after_epoch=True))

    extensions.append(FinishAfter(after_n_epochs=epochs))

    # extensions.append(Printing())
    # extensions.append(ProgressBar())

    extensions.append(TrackTheBest('test_mape', choose_best=one_perc_min))
    extensions.append(TrackTheBest('test_cost', choose_best=one_perc_min))
    extensions.append(FinishIfNoImprovementAfter('test_cost_best_so_far', epochs=500))

    # Save only parameters, not the whole main loop and only when best_test_cost is updated
    checkpoint = Checkpoint(save_file, save_main_loop=False, after_training=False)
    checkpoint.add_condition(['after_epoch'], predicate=OnLogRecord('test_cost_best_so_far'))
    extensions.append(checkpoint)

    if BOKEH_AVAILABLE and plot:
        extensions.append(Plot(execution_name, channels=[[  # 'train_cost',
                                                          'test_cost']]))

    step_rule = Adam()

    algorithm = GradientDescent(cost=c_test, parameters=cg.parameters, step_rule=step_rule)
    main_loop = MainLoop(algorithm, stream_train, model=Model(c_test), extensions=extensions)
    main_loop.run()

    test_mape = 0
    if main_loop.log.status.get('best_test_mape', None) is None:
        with open(save_file, 'rb') as f:
            parameters = load_parameters(f)
            model = main_loop.model
            model.set_parameter_values(parameters)
            ev = DatasetEvaluator([c_test])
            test_mape = ev.evaluate(stream_test)['mape']
    else:
        test_mape = main_loop.log.status['best_test_mape']

    return test_mape, main_loop.log.status['epochs_done']


