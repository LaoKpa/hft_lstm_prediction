import argparse
import logging

from hft_lstm import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser("LSTM trainer and sampler based on HFT data")

    parser.add_argument("--save-path", default="C:\\Users\\tiago\\PycharmProjects\\hft_lstm_prediction\\saved_models\\", nargs="?",
                        help="Path to save the model")
    parser.add_argument("--data-path", default="C:\\Users\\tiago\\PycharmProjects\\hft_lstm_prediction\\hft_data\\", nargs="?",
                        help="Path to HFT data")
    parser.add_argument("--lstm-dim", type=list, default=[1, 5],
                        help="Number of hidden units in the LSTM layer.")
    parser.add_argument("--columns", type=int, default=1,
                        help="Which column set to execute.")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of examples in a single batch.")
    parser.add_argument("--num-epochs", type=int, default=10000,
                        help="Number of epochs to do.")
    parser.add_argument("--plot", type=bool, default=False,
                        help="Plot the test cost.")

    args = parser.parse_args()
    main(**vars(args))
