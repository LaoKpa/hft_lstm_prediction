from blocks.bricks.recurrent import Linear, LSTM
from blocks.bricks import Initializable
from blocks.bricks.base import application
from blocks.initialization import Constant, IsotropicGaussian

from theano.printing import Print


class LinearLSTM(Initializable):

    def __init__(self, input_dim, output_dim, lstm_dim, print_intermediate=False, print_attrs=['__str__'], **kwargs):
        super(LinearLSTM, self).__init__(**kwargs)

        self.x_to_h = Linear(input_dim, lstm_dim * 4, name='x_to_h',
                             weights_init=IsotropicGaussian(),
                             biases_init=Constant(0.0))
        self.lstm = LSTM(lstm_dim, name='lstm',
                         weights_init=IsotropicGaussian(),
                         biases_init=Constant(0.0))
        self.h_to_o = Linear(lstm_dim, output_dim, name='h_to_o',
                             weights_init=IsotropicGaussian(),
                             biases_init=Constant(0.0))

        self.children = [self.x_to_h, self.lstm, self.h_to_o]

        self.print_intermediate = print_intermediate
        self.print_attrs = print_attrs

    @application
    def apply(self, source):

        x_linear = self.x_to_h.apply(source.reshape((source.shape[1], source.shape[0], source.shape[2])))
        x_linear.name = 'x_linear'
        if self.print_intermediate:
            x_linear = Print(message='x_linear info', attrs=self.print_attrs)(x_linear)

        h, c = self.lstm.apply(x_linear)
        if self.print_intermediate:
            h = Print(message="hidden states info", attrs=self.print_attrs)(h)

        y_hat = self.h_to_o.apply(h)
        y_hat.name = 'y_hat'
        if self.print_intermediate:
            y_hat = Print(message="y_hat info", attrs=self.print_attrs)(y_hat)

        return y_hat

    def initialize(self):
        for child in self.children:
            child.initialize()

    def reset_allocation(self):
        for child in self.children:
            child.allocated = False
