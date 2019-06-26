import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

train_dataset = torchvision.datasets.MNIST(root, train=True, transforms=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root, train=False, transforms=transforms.ToTensor())
# using datasets that pytorch already has

train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size_, shuffle = True)
# shuffle -> per each epoch, the data is shuffled
test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size_, shuffle = False)


# hyper parameter =

length_sentence = 28
input_size = 28
hidden_size = 100
num_layers = 2
# 2 level layer rnn
num_classes = 1
batch_size = 25
num_epochs = 50
learning_rate = 0.01


class RnnTest(nn.Module):
    def __init__(self):
        super(RnnTest, self).__init__()
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, batch_first = True)

    def forward(self, x, hidden_state):
        x = x.reshape(batch_size, length_sentence, input_size)

        out, hidden_state = self.rnn(x, hidden_state)
        # out and hidden vector => two different outputs come out




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # the inputs to an LSTM are (input, (h0, c0))
        # h and c are both 3 dimesions =>  (n_layers, batch, hidden_dim) if they are not specified
        # input: a Tensor containing the values in an iput sequence : (seq_len, batch, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # initial hidden and cell state tensors

        out, _ = self.lstm(x, (h0, c0))
        # (batch, seq_len, hidden_size) as it is output (input : (batch, seq_len, input_size))



        # fc part not understood yet

        return out

model = LSTM(input_size, hidden_size, num_layers, num_classes)

loss_real_predict = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(Cnn_model.parameters(), learning_rate = learning_rate)
