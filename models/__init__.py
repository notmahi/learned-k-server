from fc import FCModel as FullyConnected
from rnn import LSTMModel as LSTM

model_dict = {
    'lstm': LSTM,
    'fc': FullyConnected
}