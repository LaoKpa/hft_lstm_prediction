from __future__ import print_function

import random
import numpy as np
import theano
import theano.tensor as T

from fuel.streams import DataStream
from fuel.datasets import IterableDataset

from blocks.bricks import Linear, Logistic
from blocks.bricks.recurrent import LSTM
from blocks.bricks.cost import SquaredError
from blocks.initialization import Constant, IsotropicGaussian
from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.serialization import load
from blocks.model import Model

from itertools import ifilter

import pandas
import numpy

import pdb

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False


def parse_data_into_batches(data, batch_size):
    # The batch's size define how many days it will contain
    grouped = data.groupby(pandas.Grouper(key="Date", freq="1B"))

    close_target = pandas.DataFrame(data.Close).drop(0).reset_index(drop=True)
    data['CloseTarget'] = close_target
    data.drop(len(data) - 1, inplace=True)

    beforegroup = None

    # The very first line of each batch is built from the last line from the latter one
    # if is the very one, then calculate the mean and create a new one
    x = []
    y = []
    x_batch = []
    y_batch = []
    print("batch_size: {}".format(batch_size))

    for i, group in zip(range(len(grouped)), grouped):

        print("i: {}, group: {}, group_len: {}".format(i, group[0], len(group[1])))
        # the day might be a holiday, hence no info will come
        if len(group[1]) == 0:
            continue
        day = group[1]
        y_day = pandas.DataFrame(day.CloseTarget).values.astype(dtype=theano.config.floatX)
        x_day = day.drop(['CloseTarget', 'Date'], axis=1).values.astype(dtype=theano.config.floatX)

        print("shapes = x_day {} y_day {}".format(x_day.shape, y_day.shape))

        # new batch
        if (len(x_batch) + 1) == batch_size or (len(grouped)-1) == i:

            xnp = numpy.zeros((batch_size, x_day.shape[0], x_day.shape[1]), dtype=theano.config.floatX)
            # ynp = numpy.zeros((batch_size, y_day.shape[0]), dtype=theano.config.floatX)
            ynp = numpy.zeros((batch_size, y_day.shape[0], y_day.shape[1]), dtype=theano.config.floatX)

            print("shapes = xnp {} ynp {}".format(xnp.shape, ynp.shape))
            for index, xb, yb in zip(range(len(x_batch)), x_batch, y_batch):
                # this should be removed!! just to by pass the error and try to make it work
                try:
                    xnp[index] = xb
                    # ynp[index] = yb.reshape((yb.shape[0], yb.shape[1]))
                    ynp[index] = yb
                except ValueError:
                    print("An error would happen, this occurs because this day does not have "
                          "as many measures as the others")
                    pass

            xnp = xnp.swapaxes(0, 1)
            ynp = ynp.swapaxes(0, 1)

            print("xnp {} ynp {}".format(xnp.shape, ynp.shape))

            x.append(xnp)
            y.append(ynp)
            x_batch = []
            y_batch = []

        x_batch.append(x_day)
        y_batch.append(y_day)

    # For RNN, batches are formatted in the following way
    # x: (Seq. Length, Batch Size, Char. Dimensions)
    # y: (Batch Size, Output Dimension)

    # it seems that no conversion to numpy obj is necessary
    # as Fuel uses iterable objects (so lists of numpy array are ok)
    # xnp = numpy.zeros((len(x_batch), x_batch[0].shape[0], x_batch[0].shape[1]))
    # for i in xrange(len(x_batch)):
    #     xnp[i] = x_batch[i]
    return {"x": x, "y": y}


def load_data(data_path):
    """
    :param data_path:
    :return: two DataStream, the first is the train and the other is the test
    """
    data = pandas.read_csv(data_path + "dados_petr.csv", sep=";")
    data['Date'] = pandas.to_datetime(data.Date, dayfirst=True)
    data.sort_values('Date')

    # separate train data from test data
    loaded_train = data[data.Date.map(lambda x: x.month != 10)]
    loaded_test = data[data.Date.map(lambda x: x.month == 10)].reset_index(drop=True)

    return {"train": loaded_train, "test": loaded_test}


def main(save_path, data_path, lstm_dim, batch_size, num_epochs):
    # The file that contaitns the model saved is a concatenation of information passed
    if save_path.endswith("//") is False:
        save_path += "//"

    execution_name = "lstm" + "_" + str(lstm_dim) + "_" + str(batch_size) + "_" + str(num_epochs)

    save_file = save_path + execution_name

    # try:
    #     with open(save_file, 'rb') as loaded_file:
    #         main_loop = load(loaded_file)
    #
    #         # import pdb
    #         # pdb.set_trace()
    #
    #         x = list(ifilter(lambda l: l.name == 'x', main_loop.model.inputs))[0]
    #
    #         y = list(ifilter(lambda l: l.name == 'logistic_apply_output', main_loop.model.intermediary_variables))[0]
    #
    #         f = theano.function([x], y)
    #
    #         y = f(np.asarray([
    #             [[0], [1], [1]],
    #             [[0], [1], [0]],
    #             [[0], [1], [0]],
    #         ], dtype=theano.config.floatX))
    #
    #         print(y)
    #
    #         return main_loop
    # except IOError:
    #     print('There was no previous training!')

    d = load_data(data_path)

    data_train = parse_data_into_batches(d["train"], batch_size)
    data_test = parse_data_into_batches(d["test"], batch_size)

    dataset_train = IterableDataset(data_train)
    dataset_test = IterableDataset(data_test)

    stream_train = DataStream(dataset=dataset_train)
    stream_test = DataStream(dataset=dataset_test)

    x = T.tensor3('x')
    # y = T.matrix('y')
    y = T.tensor3('y')

    # we need to provide data for the LSTM layer of size 4 * ltsm_dim, see
    # LSTM layer documentation for the explanation
    x_to_h = Linear(6, 6 * lstm_dim * 4, name='x_to_h',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0))
    lstm = LSTM(6 * lstm_dim, name='lstm',
                weights_init=IsotropicGaussian(),
                biases_init=Constant(0.0), )
    h_to_o = Linear(6 * lstm_dim, 1, name='h_to_o',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0))

    x_transform = x_to_h.apply(x)
    h, c = lstm.apply(x_transform)

    # only values of hidden units of the last timeframe are used for
    # the classification
    # y_hat = h_to_o.apply(h[-1])
    y_hat = h_to_o.apply(h)
    y_hat.name = 'y_hat'

    cost = SquaredError().apply(y, y_hat)
    cost.name = 'cost'

    lstm.initialize()
    x_to_h.initialize()
    h_to_o.initialize()

    cg = ComputationGraph(cost)

    algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Adam())
    test_monitor = DataStreamMonitoring(variables=[cost], data_stream=stream_test, prefix="test")
    train_monitor = TrainingDataMonitoring(variables=[cost], prefix="train", after_epoch=True)

    if BOKEH_AVAILABLE:
        plot = Plot(execution_name, channels=[['train_cost', 'test_cost']])

    main_loop = MainLoop(algorithm, stream_train, model=Model(cost),
                         extensions=[test_monitor, train_monitor,
                                     FinishAfter(after_n_epochs=num_epochs),
                                     Printing(), ProgressBar(), Checkpoint(save_file), plot])
    main_loop.run()

    return main_loop
