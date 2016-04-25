from __future__ import print_function


from blocks.bricks import Linear
from blocks.bricks.recurrent import LSTM

from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import Scale
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

    execution_name = 'lstm' + '_' + str(lstm_dim) + '_' + str(batch_size) + '_' + str(num_epochs)

    save_file = save_path + execution_name

    sw_iterator = SlidingWindowPandasIterator(data_path=data_path + 'dados_petr.csv',
                                              normalize=True,
                                              normalize_target=False,
                                              train_size=pd.Timedelta(days=21),
                                              test_size=pd.Timedelta(days=7),
                                              step_size=pd.Timedelta(days=7))

    avg_test_cost = 0
    number_of_sets = 0

    for train, test in sw_iterator:

        stream_train = build_stream(train)
        stream_test = build_stream(test)

        # The train stream will return (TimeSequence, BatchSize, Dimensions) for
        # and the train test will return (TimeSequence, BatchSize, 1)

        x = T.tensor3('x')
        y = T.tensor3('y')

        y = y.reshape((y.shape[1], y.shape[0], y.shape[2]))

        # input_dim = 6
        # output_dim = 1
        linear_lstm = LinearLSTM(6, 1, lstm_dim,
                                 # print_intermediate=True,
                                 print_attrs=['__str__', 'shape'])

        y_hat = linear_lstm.apply(x)
        linear_lstm.initialize()

        c = AbsolutePercentageError().apply(y, y_hat)
        # c = SquaredError().apply(y, y_hat)
        c.name = 'cost'

        cg = ComputationGraph(c)

        extensions = [TrainingDataMonitoring(variables=[c], prefix='train', after_epoch=True),
                      FinishAfter(after_n_epochs=num_epochs),
                      Printing(),
                      ProgressBar(),
                      TrackTheBest('test_cost'),
                      TrackTheBest('train_cost')]

        if BOKEH_AVAILABLE:
            extensions.append(Plot(execution_name, channels=[['train_cost', 'test_cost']]))

        algorithm = GradientDescent(cost=c, parameters=cg.parameters, step_rule=Adam())
        extensions.insert(0, DataStreamMonitoring(variables=[c], data_stream=stream_test, prefix='test'))

        main_loop = MainLoop(algorithm, stream_train, model=Model(c), extensions=extensions)
        main_loop.run()

        avg_test_cost += main_loop.log.status['best_test_cost']
        number_of_sets += 1

        # due to dubious definition of the masters thesis
        if number_of_sets == 35:
            break

        # plot_test(x, y_hat, converter, save_file)

    print("average test cost {}".format(avg_test_cost / float(number_of_sets)))

    return main_loop
