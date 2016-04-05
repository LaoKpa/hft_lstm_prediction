
import numpy as np
import pandas as pd

from fuel.streams import DataStream
from fuel.datasets import IndexableDataset
from fuel.schemes import IterationScheme
from fuel.streams import DataStream
from fuel.transformers import Cast, Mapping

from six import add_metaclass
from abc import ABCMeta, abstractmethod

from collections import OrderedDict

import theano

# copied and adapted from 
# https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
class SlidingWindow(object):
    """Returns iterator that will emit chunks of size 'winSize' each time self.next()
    is called."""
    def __init__(self, sequence, winSize, step=1):
        """Returns iterator that will emit chunks of size 'winSize' and 'step' forward in
        the seq each time self.next() is called."""
 
        # verification code
        if not isinstance(winSize, int) and isinstance(step, int):
            raise Exception("**ERROR** type(winSize) and type(step) must be int.")
        if step > winSize:
            raise Exception("**ERROR** step must not be larger than winSize.")
        if winSize > len(sequence):
            raise Exception("**ERROR** winSize must not be larger than sequence length.")
        self._seq = sequence
        self._step = step
        self._start = 0
        self._stop = winSize
 
    def __iter__(self):
        return self
 
    def next(self):
        """Returns next window chunk or ends iteration if the sequence has ended."""
        try:
            assert self._stop <= len(self._seq), "Not True!"
            chunk = self._seq[self._start:self._stop]
            self._start += self._step
            self._stop  += self._step
            return chunk
        except AssertionError:
            raise StopIteration


@add_metaclass(ABCMeta)
class WindowScheme(IterationScheme):
    requests_examples = False

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        super(IterationScheme, self).__init__(**kwargs)

    def get_request_iterator(self):
        grouped = self.dataset.groupby(pd.Grouper(key="Date", freq="1B"))
        l = sorted(grouped.indices.items(), key=lambda x: x[0])
        l = [x for t, x in l if len(x) > 0]

        for l in SlidingWindow(l, 4):
            yield self.window_filter(l)

    @abstractmethod
    def window_filter(self, l):
        pass


class TrainWindowScheme(WindowScheme):
    def window_filter(self, l):
        return l[0] + l[1] + l[2]


class TestWindowScheme(WindowScheme):
    def window_filter(self, l):
        return l[3]


class StreamGenerator(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.pd_data = None
        self.dataset = None

    def load(self):
        data = pd.read_csv(self.data_path, sep=";")
        data['Date'] = pd.to_datetime(data.Date, dayfirst=True)
        data.sort_values('Date')

        def normalize(pdata):
            return (pdata - pdata.mean()) / (pdata.max() - pdata.min())

        # Except Date
        data[data.columns[1:]] = normalize(data[data.columns[1:]])

        close_target = pd.DataFrame(data.Close).drop(0).reset_index(drop=True)
        close_target.loc[len(close_target)] = data.Close[0]
        data['CloseTarget'] = close_target

        self.pd_data = data

        columns = list(data.columns)[1:]
        columns.remove('CloseTarget')

        self.dataset = IndexableDataset(indexables=OrderedDict([('x', data[columns].values),
                                                                ('y', data['CloseTarget'].values)]))

    def get_streams(self):
        stream_train = self.get_stream(TrainWindowScheme(self.pd_data))
        stream_test = self.get_stream(TestWindowScheme(self.pd_data))

        return stream_train, stream_test

    def get_stream(self, scheme):
        stream = DataStream(dataset=self.dataset,
                            iteration_scheme=scheme)

        stream = Mapping(stream, add_axes)
        stream = Cast(stream, theano.config.floatX)
        return stream


# global to pickle
def add_axes(batch):
    first = batch[0][np.newaxis, :]
    last = batch[1][np.newaxis, :, np.newaxis]
    output = first, last
    return output

