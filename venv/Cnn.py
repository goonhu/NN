import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader


# GPU part skips as this is a test coding and this laptop doesn't have a gpu

batch_size_ = 25
num_epochs = 100
num_classes = 2
learning_rate = 0.1



train_dataset = torchvision.datasets.MNIST(root, train=True, transforms=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root, train=False, transforms=transforms.ToTensor())
# using datasets that pytorch already has

train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size_, shuffle = True)
# shuffle -> per each epoch, the data is shuffled
test_data_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size_, shuffle = False)



class CnnTest(nn.Module):

    def __init__(self, num_classes = 2):
    # binary classificaiton

        super(CnnTest, self).__init__()
#         since image data set => 2d
#         in channel => 1 as it is not a colour image

        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size = 5, stride = 1, padding = 0),
                nn.BatchNorm2d(10),
                nn.Relu(),
                nn.MaxPool2d(kernel_size=2, stride =1))

        self.layer2 = nn.Sequential(
                nn.Conv2d(10, 20, kernel_size = 5, stride = 2, padding = 0),
                nn.BatchNorm2d(),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, strid = 1))

        self.fc = nn.Linear()


    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # out.size(0) => batch size, so the dimension is flattened into the vector apart from the part for batch size
        # and then this result gets through fc network
        out = self.fc(out)


Cnn_model = CnnTest(num_classes)

loss_real_predict = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(Cnn_model.parameters(), learning_rate = learning_rate)


# training part

for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_data_loader):

        output = Cnn_model(image)
        loss_ = loss_real_predict(output, label)
        # forward the model using image and label and get the loss comparing the real label with the predicted output

        optimiser.zero_grad()
        # before backward, need to set the gradient to zero
        loss_.backward()
        # doing backward, need to get gradients for each step
        optimiser.step()
        # step() updates all parameters based on the current gradient

        if (i+1) % 50 == 0:
            print("Epoch {}, Step {}, Loss : {}".format(epoch+1, i+1, loss_.item()))

        # loss.item() => gives the scalar value of loss


# with torch.no_grad():
#         # need to set the gradient to zero again
#
#         # before training