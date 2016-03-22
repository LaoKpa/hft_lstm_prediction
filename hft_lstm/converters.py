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


def swap_axes_batch(batch):
    # print('BATCH x shape {} y shape {}'.format(batch[0].shape, batch[1].shape))
    return batch[0].transpose(1, 0, 2), batch[1][np.newaxis, :]


@add_metaclass(ABCMeta)
class PandasStreamsConverter(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.loaded_train = None
        self.loaded_test = None

    def load(self):
        data = pandas.read_csv(self.filepath, sep=';')
        data['Date'] = pandas.to_datetime(data.Date, dayfirst=True)
        data.sort_values('Date', inplace=True)

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
        return OrderedDict([('x',tuple(self.get_dimensions())), ('y', tuple(['CloseTarget']))])

    def get_dimensions(self):
        """

        :return: list of dimensions' names
        """
        return list(self.loaded_train.drop('CloseTarget', axis=1).columns)

    def get_streams(self, batch_size):

        train = self._parse_to_stream(batch_size, *PandasStreamsConverter._to_numpy(self.loaded_train))
        test = self._parse_to_stream(batch_size, *PandasStreamsConverter._to_numpy(self.loaded_test))

        return train, test

    @abstractmethod
    def _parse_to_stream(self, batch_size, features, targets):
        """
        :param features: numpy.array representing the features of the dataset
        :param targets: numpy.array representing the features of the dataset
        :return: blocks.stream.DataStream
        """


class BatchStreamConverter(PandasStreamsConverter):

    def _parse_to_stream(self, batch_size, features, targets):

        # Create new axes, later needed due to recurrent nets' blocks architecture
        # (Batch Size, Sequence Length, Dimensions)
        features = features[:, np.newaxis, :]
        targets = targets[:, np.newaxis]

        print("features {} targets {}".format(features.shape, targets.shape))

        dataset = IndexableDataset(indexables=OrderedDict([('x', features), ('y', targets)]),
                                   axis_labels=self.get_axis_labels())

        stream = DataStream(dataset=dataset,
                            iteration_scheme=SequentialScheme(examples=dataset.num_examples,
                                                              batch_size=batch_size))
        # Changes batch from (Batch Size, Sequence Length, Dimensions) to (Sequence Length, Batch Size, Dimensions)
        # needed for blocks recurrent structures
        stream = Mapping(stream, swap_axes_batch)

        return stream


class IterStreamConverter(PandasStreamsConverter):

    def _parse_to_stream(self, batch_size, features, targets):

        # Create new axes, later needed due to recurrent nets' blocks architecture
        features = features[:, np.newaxis, :]
        targets = targets[:, np.newaxis]

        print("features {} targets {}".format(features.shape, targets.shape))

        dataset = IterableDataset(iterables=OrderedDict([('x', features), ('y', targets)]),
                                  axis_labels=self.get_axis_labels())

        stream = DataStream(dataset=dataset)

        return stream
