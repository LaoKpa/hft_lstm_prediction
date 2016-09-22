# Predicting Stock Value in a High-Frequency Trading scope using Recurrent Neural Networks (LSTM)

This python 2.7 development project was part of an undergradute bachelor thesis in computer science (This is common in Brazil, however is similar to capstone projects in North America and other countries). The work is to predict the price of a stock in an intraday scope (15 minutes), which the focus was aiming high-frequency trading (HFT). 

The main objective was to compare the use of recurrent neural nets. over feedforward ones, which was a master's thesis developed by other student. Due to this, you will find comparisons that seems not so useful for RNN, however they were necessary for comparison.

In the folder [hft_data](hft_data), you will find two datasets extracted from BOVESPA stock market, where [dados_petr.csv](hft_data/dados_petr.csv) is from Petrobrás and [dados_vale.csv](hft_data/dados_vale.csv). Both contains data in the same structure: Samples extracted in a 15 minutes frequency, containing the price of the stocks when the period opened and closed, the highest and the lowest price within this range, the quantity of stocks traded and the volume in Reais of how much was traded.

This project uses Theano and Blocks for modeling the recurrent net. A custom brick called [LinearLSTM](hft_lstm/custom_bricks.py) was created aiming simplicity in the training code, there you will know how the default Bricks are stacked. The training code is done within the ```train_lstm``` method in [init file](hft_lstm/__init__.py).

The neural net minimizes the Squared Error, however it uses the [Mean absolute percentage error](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)(MAPE) in order to compare to the master's thesis.

