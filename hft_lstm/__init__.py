from __future__ import print_function


from blocks.bricks import Linear
from blocks.bricks.recurrent import LSTM
from blocks.bricks.cost import SquaredError
from blocks.initialization import Constant, IsotropicGaussian
from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.model import Model

import theano.tensor as T
import numpy as np
import pdb
from converters import BatchStreamConverter

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


def main(save_path, data_path, lstm_dim, batch_size, num_epochs):
    # The file that contains the model saved is a concatenation of information passed
    if save_path.endswith('//') is False:
        save_path += '//'

    execution_name = 'lstm' + '_' + str(lstm_dim) + '_' + str(batch_size) + '_' + str(num_epochs)

    save_file = save_path + execution_name

    converter = BatchStreamConverter(data_path + 'dados_petr.csv')
    converter.load()

    stream_train, stream_test = converter.get_streams(batch_size)

    x = T.tensor3('x')
    y = T.tensor3('y')

    # we need to provide data for the LSTM layer of size 4 * lstm_dim, see
    # LSTM layer documentation for the explanation
    x_to_h = Linear(6, lstm_dim * 4, name='x_to_h',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0))
    lstm = LSTM(lstm_dim, name='lstm',
                weights_init=IsotropicGaussian(),
                biases_init=Constant(0.0))
    h_to_o = Linear(lstm_dim, 1, name='h_to_o',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0))

    x_transform = x_to_h.apply(x)
    h, c = lstm.apply(x_transform)
    y_hat = h_to_o.apply(h)
    y_hat.name = 'y_hat'

    lstm.initialize()
    x_to_h.initialize()
    h_to_o.initialize()

    # start of testing
    # f = theano.function([x], y_hat)
    #
    # for data in stream_test.get_epoch_iterator():
    #     print('x shape: {} , y shape: {}'.format(data[0].shape, data[1].shape))
    #     y_test = f(data[0])
    #     print("y_test {}".format(y_test))
    #     print("y shape {}".format(y_test.shape))
    #     return
    # end of testing

    cost = SquaredError().apply(y, y_hat)
    cost.name = 'cost'

    cg = ComputationGraph(cost)

    algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Adam())
    test_monitor = DataStreamMonitoring(variables=[cost], data_stream=stream_test, prefix='test')
    train_monitor = TrainingDataMonitoring(variables=[cost], prefix='train', after_epoch=True)

    if BOKEH_AVAILABLE:
        plot = Plot(execution_name, channels=[['train_cost', 'test_cost']])

    main_loop = MainLoop(algorithm, stream_train, model=Model(cost),
                         extensions=[test_monitor, train_monitor,
                                     FinishAfter(after_n_epochs=num_epochs),
                                     Printing(), ProgressBar(), Checkpoint(save_file), plot])
    main_loop.run()

    print('If you reached here, you have a trained LSTM :)')

    return main_loop
