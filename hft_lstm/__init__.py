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

import converters
from cost import AbsolutePercentageError
from custom_bricks import LinearLSTM

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


def find_theano_var_in_list(name, list_to_search):
    return list(ifilter(lambda l: l.name == name, list_to_search))[0]


def main(save_path, data_path, lstm_dim, batch_size, num_epochs):
    # The file that contains the model saved is a concatenation of information passed
    if save_path.endswith('//') is False:
        save_path += '//'

    execution_name = 'lstm' + '_' + str(lstm_dim) + '_' + str(batch_size) + '_' + str(num_epochs)

    save_file = save_path + execution_name

    converter = converters.StreamGenerator(data_path + 'dados_petr.csv')
    converter.load()

    # The train stream will return (TimeSequence, BatchSize, Dimensions) for
    # and the train test will return (TimeSequence, BatchSize, 1)
    stream_train, stream_test = converter.get_streams()

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

    # c = AbsolutePercentageError().apply(y, y_hat)
    c = SquaredError().apply(y, y_hat)
    c.name = 'cost'

    cg = ComputationGraph(c)

    algorithm = GradientDescent(cost=c, parameters=cg.parameters, step_rule=Adam())

    extensions = [DataStreamMonitoring(variables=[c], data_stream=stream_test, prefix='test'),
                  TrainingDataMonitoring(variables=[c], prefix='train', after_epoch=True),
                  FinishAfter(after_n_epochs=num_epochs),
                  Printing(),
                  ProgressBar(),
                  TrackTheBest('test_cost'),
                  TrackTheBest('train_cost')]

    if BOKEH_AVAILABLE:
        extensions.append(Plot(execution_name, channels=[['train_cost', 'test_cost']]))

    main_loop = MainLoop(algorithm, stream_train, model=Model(c),
                         extensions=extensions)
    main_loop.run()

    print('If you reached here, you have a trained LSTM :)')

    return main_loop
