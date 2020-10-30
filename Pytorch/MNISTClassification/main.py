#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
class BasicCNN(nn.Module):
    """Implements a CNN class by inheriting nn.Module
    """
    def __init__(self):
        """Here we define the convolutions and fully connected networks 
        we are going to use
        """
        super(BasicCNN,self).__init__()
        # conv1 takes 1 channel/feature as input and generates 4 feature maps,
        # it is using a 3x3 size for convolution with a stride of one and no padding
        # dilation (https://erogol.com/dilated-convolution/) is kept at 1 which is
        # normal convolution
        self.conv1 = nn.Conv2d(in_cannels=1,out_channels=4,kernel_size=3\
            stride=1,padding=0,dilation=1,groups=1)

        # conv2 takes 4 channel/feature as input and generates 15 feature maps,
        # it is using a 2x2 size for convolution with a stride of 1 and no padding
        # dilation (https://erogol.com/dilated-convolution/) is kept at 1 which is normal
        # convolution
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=15,kernel_size=2\
            stride=1,padding=0,dilation=1,groups=1)

        # fc1 is a fully connected neural network in which 15*3*3 inputs are mapped to 
        # a hidden layer of size 100 here the coise of 3 in (15*3*3) has to be carefully
        # calculated by looking at the network you are constructing befor the FC layers
        # Example:
        #
        #
        #
        self.fc1 = nn.Linear(in_features = 15*3*3,out_features=100)

        # fc1 is a fully connected neural network in which 100 inputs are mapped to 
        # a hidden layer of size 10, this quantity is selected because there are
        # ultimately 10 labels for MNIST dataset
        self.fc2 = nn.Linear(in_features=100,out_features=10)
    
    def forward(self,x):
        # We are designing a network that looks like the following
        # input->conv1->max_pool->conv2->max_pool->flatten->fc1->fc2->output
        # note that we have used ReLu activations here.
        # the thing to note here is that our input x is of the form
        # [<batchsize>,<channels>,<width>,<height>]
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self,x):
        # umber of features after flattening for x, nF = <channels>*<width>*<height>
        features = 1
        for dim in x.size()[1:]:
            features *= dim
        return features


    
# %%
#For classification of MNIST Data, let us create an object of BasicCNN class
mnist_net = BasicCNN()
# %%
#Here is what the network looks like
print(mnist_net)
# %%
#We can actually see the parameters of our network
params = list(mnist_net.parameters())
print(len(params))
print(params[0].size())
# %%
# Let us now bring in some data, we will use MNIST database
from torchvision.datasets import MNIST 
from torchvision import transforms

#This transform is used to normalize the data and cast the images as tensor
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

train_set = MNIST(root = './data',train=True,download=True,transform=transform)

#It would be nice to have the images as an iterable, we are selsctring a batch
# size of 4 and shuffling the data so that the sequence of the data may not create
# bias in our training
trainLoader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4,shuffle=True,\
    num_workers=0)

# Lets us now get the test set
test_set = MNIST(root='./data',train=False,download=True,transform=transform)

# Again we will generate iterable
testLoader = torch.utils.data.DataLoader(dataset=test_set,shuffle=False,batch_size=4,\
    num_workers=0)

#There are 10 classes in MNIST
print("MNIST Classe:{}".format(MNIST.classes))
# %%
# We will now take a look at some images to get a feel of the data
import matplotlib.pyplot as plt 
import numpy as np
import torchvision

def imshow(image):
    np_image = image.numpy()
    plt.imshow(np_image)
    plt.show()

dataiter = iter(trainLoader)
images, labels = dataiter.next()

imshow(images[1,0,:,:])
print(labels[1])
# %%
