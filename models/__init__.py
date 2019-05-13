from .fc import FCModel as FullyConnected
from .fc_q import FCQModel as FullyConnectedQ
from .rnn import LSTMModel as LSTM

model_dict = {
    'lstm': LSTM,
    'fc': FullyConnected,
    'fcq': FullyConnectedQ
}