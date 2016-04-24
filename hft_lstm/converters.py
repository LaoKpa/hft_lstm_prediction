
import numpy as np
import pandas as pd

from fuel.streams import DataStream
from fuel.datasets import IndexableDataset
from fuel.schemes import IterationScheme, SequentialScheme
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
    def __init__(self, sequence, win_size, step=1):
        """Returns iterator that will emit chunks of size 'winSize' and 'step' forward in
        the seq each time self.next() is called."""
 
        # verification code
        if not isinstance(win_size, int) and isinstance(step, int):
            raise Exception("**ERROR** type(winSize) and type(step) must be int.")
        if step > win_size:
            raise Exception("**ERROR** step must not be larger than winSize.")
        if win_size > len(sequence):
            raise Exception("**ERROR** winSize must not be larger than sequence length.")
        self._seq = sequence
        self._step = step
        self._start = 0
        self._stop = win_size
 
    def __iter__(self):
        return self
 
    def next(self):
        """Returns next window chunk or ends iteration if the sequence has ended."""
        try:
            assert self._stop <= len(self._seq), "Not True!"
            chunk = self._seq[self._start:self._stop]
            self._start += self._step
            self._stop += self._step
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

    def __init__(self, data_path, normalize=False, normalize_target=False, scheme='window'):
        self.data_path = data_path
        self.pd_data = None
        self.dataset = None
        self.normalize = normalize
        self.normalize_target = normalize_target
        self.mean = 0
        self.maxmin = 0
        self.scheme = scheme

    def load(self):
        data = pd.read_csv(self.data_path, sep=";")
        data['Date'] = pd.to_datetime(data.Date, dayfirst=True)
        data.sort_values('Date')

        close_target = None
        if self.normalize:
            def normalize(pdata):
                mean = pdata.mean()
                maxmin = pdata.max() - pdata.min()
                return (pdata - mean) / maxmin, mean, maxmin

            if self.normalize_target is False:
                close_target = pd.DataFrame(data.Close).drop(0).reset_index(drop=True)
                close_target.loc[len(close_target)] = data.Close[0]
            # Except Date
            data[data.columns[1:]], self.mean, self.maxmin = normalize(data[data.columns[1:]])

        # normalize the target is activated
        if close_target is None:
            close_target = pd.DataFrame(data.Close).drop(0).reset_index(drop=True)
            close_target.loc[len(close_target)] = data.Close[0]

        data['CloseTarget'] = close_target

        data['Time'] = data['Date'].apply(lambda x: x.time())

        self.pd_data = data

        columns = ['Open', 'High', 'Low', 'Close', 'Qty', 'Vol']

        print('data loaded successfully')
        print(data.head())

        self.dataset = IndexableDataset(indexables=OrderedDict([('x', data[columns].values),
                                                                ('y', data['CloseTarget'].values)]))

    def get_streams(self):
        stream_train = self.get_stream(self.get_train_scheme())
        stream_test = self.get_stream(self.get_test_scheme())

        return stream_train, stream_test

    def get_train_scheme(self):
        if self.scheme == 'window':
            return TrainWindowScheme(self.pd_data)
        else:
            return SequentialScheme(examples=range(5150), batch_size=5150)

    def get_test_scheme(self):
        if self.scheme == 'window':
            return TestWindowScheme(self.pd_data)
        else:
            rng = range(5150, 5513)
            return SequentialScheme(examples=rng, batch_size=len(rng))

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

