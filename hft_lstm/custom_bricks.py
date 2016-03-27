from blocks.bricks.recurrent import Linear, LSTM
from blocks.bricks import Initializable
from blocks.bricks.base import application
from blocks.initialization import Constant, IsotropicGaussian


class LinearLSTM(Initializable):

    def __init__(self, input_dim, output_dim, lstm_dim, **kwargs):
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

    @application
    def apply(self, source):
        x_linear = self.x_to_h.apply(source)
        x_linear.name = 'x_linear'
        h, c = self.lstm.apply(x_linear)
        y_hat = self.h_to_o.apply(h)
        y_hat.name = 'y_hat'
        return y_hat

    def initialize(self):
        for child in self.children:
            child.initialize()
