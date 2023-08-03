try:
    from text.lstm import SimpleLSTM
    from text.rnn import SimpleRNN
    from text.transformer import (TransformerClassificationModel,
                                  TransformerLangModel)
except BaseException:
    pass
try:
    from vision.densenet import DenseNet40
    from vision.lenet import LeNet5
    from vision.CNNCifar import CNNCifar
    from vision.CNNMnist import CNNMnist
    from vision.CNNImageNet import CNNImageNet
except BaseException:
    pass
try:
    from graph.gcn import SimpleGCN
except BaseException:
    pass
