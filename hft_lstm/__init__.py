from __future__ import print_function

import theano
import theano.tensor as T
from fuel.datasets.base import IndexableDataset

from fuel.streams import DataStream
from fuel.datasets import IndexableDataset
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping

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

import pandas
from collections import OrderedDict
import numpy as np

import pdb

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


def load_data(data_path):
    """
    :param data_path: Path to data
    :return: two DataStreams, the first is the train and the other is the test
    """
    data = pandas.read_csv(data_path + 'dados_petr.csv', sep=';')
    data['Date'] = pandas.to_datetime(data.Date, dayfirst=True)
    data.sort_values('Date', inplace=True)

    # Create the target column, that is the Close value from the next row
    data['CloseTarget'] = pandas.DataFrame(data.Close).drop(0).reset_index(drop=True)

    # Remove the row as it there is no target to point (CloseTarget will be NaN)
    data.drop(len(data) - 1, inplace=True)

    # separate train data from test data and delete column Date (no longer needed)
    loaded_train = data[data.Date.map(lambda x: x.month != 10)].copy().drop(['Date'], axis=1)
    loaded_test = data[data.Date.map(lambda x: x.month == 10)].copy().drop(['Date'], axis=1).reset_index(drop=True)

    return {'train': loaded_train, 'test': loaded_test}


def pandas_to_batch_stream(data_pandas, batch_size):

    features = data_pandas.drop('CloseTarget', axis=1).values.astype(theano.config.floatX)[:, np.newaxis, :]
    targets = data_pandas['CloseTarget'].values.astype(theano.config.floatX)[:, np.newaxis]

    print("features {} targets {}".format(features.shape, targets.shape))

    dataset = IndexableDataset(indexables=OrderedDict([('x', features), ('y', targets)]),
                               axis_labels=OrderedDict([('x',
                                                        tuple(data_pandas.drop('CloseTarget', axis=1).columns)),
                                                        ('y', tuple(['CloseTarget']))]))

    stream = DataStream(dataset=dataset,
                        iteration_scheme=SequentialScheme(examples=dataset.num_examples,
                                                          batch_size=batch_size))

    stream = Mapping(stream, swap_axes_batch)

    return stream


def swap_axes_batch(batch):
    # print('BATCH x shape {} y shape {}'.format(batch[0].shape, batch[1].shape))
    return batch[0].transpose(1, 0, 2), batch[1][np.newaxis, :]


def main(save_path, data_path, lstm_dim, batch_size, num_epochs):
    # The file that contains the model saved is a concatenation of information passed
    if save_path.endswith('//') is False:
        save_path += '//'

    execution_name = 'lstm' + '_' + str(lstm_dim) + '_' + str(batch_size) + '_' + str(num_epochs)

    save_file = save_path + execution_name

    d = load_data(data_path)

    stream_train = pandas_to_batch_stream(d['train'], batch_size)
    stream_test = pandas_to_batch_stream(d['train'], batch_size)

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
