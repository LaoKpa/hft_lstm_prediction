
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


@add_metaclass(ABCMeta)
class PandasDataIterator(object):

    def __init__(self, data_path, normalize=False, normalize_target=False):

        self.mean = 0
        self.maxmin = 0

        data = pd.read_csv(data_path, sep=";")
        data['Date'] = pd.to_datetime(data.Date, dayfirst=True)
        data.sort_values('Date')

        close_target = None
        if normalize:
            def normalize(pdata):
                mean = pdata.mean()
                maxmin = pdata.max() - pdata.min()
                return (pdata - mean) / maxmin, mean, maxmin

            if normalize_target is False:
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

        self.data = data

        # Only used to stop the iteration
        self.last_elem = data.iloc[-1]

        # Variable responsible to iterate
        self.start_train_begin = pd.to_datetime('2014-01-01')

        self.start_train = pd.to_datetime('2014-01-01')

    def __iter__(self):
        return self

    def reset(self):
        self.start_train = self.start_train_begin

    @abstractmethod
    def next(self):
        pass


class SlidingWindowPandasIterator(PandasDataIterator):

    def __init__(self, train_size, test_size, step_size, **kwargs):
        super(SlidingWindowPandasIterator, self).__init__(**kwargs)

        if not (isinstance(train_size, pd.Timedelta) and
                isinstance(test_size, pd.Timedelta) and
                isinstance(step_size, pd.Timedelta)):
            raise ValueError("train_size and test_size must be pandas Timedelta type")

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def next(self):
        end_train = self.start_train + self.train_size
        start_test = end_train
        end_test = start_test + self.test_size

        train = self.data[(self.start_train <= self.data.Date) & (self.data.Date < end_train)]

        test = self.data[(start_test <= self.data.Date) &
                         (self.data.Date < end_test)]

        self.start_train += self.step_size

        # Time to stop?
        if self.last_elem.Date <= test.iloc[-1].Date:
            raise StopIteration()

        return train, test


def build_stream(data, batch_size, columns=None):
    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Qty', 'Vol']
    dataset = IndexableDataset(indexables=OrderedDict([('x', data[columns].values),
                                                       ('y', data['CloseTarget'].values)]))
    size = len(dataset.indexables[0])
    stream = DataStream(dataset=dataset,
                        iteration_scheme=SequentialScheme(examples=range(size), batch_size=batch_size))

    stream = Mapping(stream, add_axes)
    stream = Cast(stream, theano.config.floatX)

    return stream


# global to pickle
def add_axes(batch):
    first = batch[0][np.newaxis, :]
    last = batch[1][np.newaxis, :, np.newaxis]
    output = first, last
    return output

