from abc import ABCMeta

import pandas
import theano
import numpy as np

from six import add_metaclass
from abc import ABCMeta, abstractmethod

from collections import OrderedDict

from fuel.streams import DataStream
from fuel.datasets import IndexableDataset, IterableDataset
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping


@add_metaclass(ABCMeta)
class PandasStreamsConverter(object):

    def __init__(self, filepath, batch_size=None):
        self.filepath = filepath
        self.loaded_train = None
        self.loaded_test = None
        self.batch_size = batch_size

    def normalize(self, data):
        return (data - data.mean()) / (data.max() - data.min())

    def load(self):
        data = pandas.read_csv(self.filepath, sep=';')
        data['Date'] = pandas.to_datetime(data.Date, dayfirst=True)
        data.sort_values('Date', inplace=True)

        data[data.columns[1:]] = self.normalize(data[data.columns[1:]])

        # Create the target column, that is the Close value from the next row
        data['CloseTarget'] = pandas.DataFrame(data.Close).drop(0).reset_index(drop=True)

        # Remove the row as it there is no target to point (CloseTarget will be NaN)
        data.drop(len(data) - 1, inplace=True)

        # separate train data from test data and delete column Date (no longer needed)
        self.loaded_train = data[data.Date.map(lambda x: x.month != 10)].copy().drop(['Date'], axis=1)
        self.loaded_test = data[data.Date.map(lambda x: x.month == 10)].copy().drop(['Date'],
                                                                                    axis=1).reset_index(drop=True)

    @staticmethod
    def _to_numpy(data_pandas):
        return data_pandas.drop('CloseTarget', axis=1).values.astype(theano.config.floatX), \
               data_pandas['CloseTarget'].values.astype(theano.config.floatX)

    def get_axis_labels(self):
        return OrderedDict([('x', tuple(self.get_dimensions())), ('y', tuple(['CloseTarget']))])

    def get_dimensions(self):
        """

        :return: list of dimensions' names
        """
        return list(self.loaded_train.drop('CloseTarget', axis=1).columns)

    def get_streams(self):

        train = self._parse_to_stream(*PandasStreamsConverter._to_numpy(self.loaded_train))
        test = self._parse_to_stream(*PandasStreamsConverter._to_numpy(self.loaded_test))

        return train, test

    @abstractmethod
    def _parse_to_stream(self, features, targets):
        """
        :param features: numpy.array representing the features of the dataset
        :param targets: numpy.array representing the features of the dataset
        :return: blocks.stream.DataStream
        """


def swap_axes_batch(batch):
    output = batch[0].transpose(1, 0, 2), batch[1][np.newaxis, :]
    # print('BATCH x shape {} y shape {}'.format(output[0].shape, output[1].shape))
    return output


class BatchStreamConverter(PandasStreamsConverter):

    def __init__(self, **kwargs):
        if not kwargs.get('batch_size'):
            raise ValueError('For batch stream, batch size is mandatory')
        super(BatchStreamConverter, self).__init__(**kwargs)

    def get_iteration_scheme(self, dataset):
        return SequentialScheme(examples=dataset.num_examples, batch_size=self.batch_size)

    def _parse_to_stream(self, features, targets):

        # Create new axes, later needed due to recurrent nets' blocks architecture
        # (Batch Size, Sequence Length, Dimensions)
        features = features[:, np.newaxis, :]
        targets = targets[:, np.newaxis]

        print("features {} targets {}".format(features.shape, targets.shape))

        dataset = IndexableDataset(indexables=OrderedDict([('x', features), ('y', targets)]),
                                   axis_labels=self.get_axis_labels())

        stream = DataStream(dataset=dataset,
                            iteration_scheme=self.get_iteration_scheme(dataset))
        # Changes batch from (Mini Batch Size, Sequence Length, Dimensions)
        # to (Sequence Length, Mini Batch Size, Dimensions) needed for blocks recurrent structures
        stream = Mapping(stream, swap_axes_batch)

        return stream


class IterStreamConverter(PandasStreamsConverter):

    def _parse_to_stream(self, features, targets):

        # Create new axes, later needed due to recurrent nets' blocks architecture
        features = features[np.newaxis, :, :]
        targets = targets[np.newaxis, :, np.newaxis]

        print("features {} targets {}".format(features.shape, targets.shape))

        dataset = IterableDataset(iterables=OrderedDict([('x', features), ('y', targets)]),
                                  axis_labels=self.get_axis_labels())

        stream = DataStream(dataset=dataset)
        stream = Mapping(stream, add_axes)
        # stream = Mapping(stream, swap_axes)
        return stream


def add_axes(batch):
    first = batch[0][np.newaxis, :]
    # print('first \n {} \n first reshaped \n {}'.format(first, first.reshape(first.shape[1], first.shape[0], first.shape[2])))
    last = batch[1][np.newaxis, :]
    output = first, last
    print('BATCH x shape {} y shape {}'.format(output[0].shape, output[1].shape))
    return output


def swap_axes(batch):
    first = batch[0][np.newaxis, :]

    # print('first \n {} \n first reshaped \n {}'.format(first, first.reshape(first.shape[1], first.shape[0], first.shape[2])))

    last = batch[1][np.newaxis, :]
    output = first.reshape(first.shape[1], first.shape[0], first.shape[2]), \
             last.reshape(last.shape[1], last.shape[0], last.shape[2])
    print('BATCH x shape {} y shape {}'.format(output[0].shape, output[1].shape))
    return output

