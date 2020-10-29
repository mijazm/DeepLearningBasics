#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,4,3)
        self.conv2 = nn.Conv2d(4,15,2)
        self.fc1 = nn.Linear(15*6*6,100)
        self.fc2 = nn.Linear(100,2)
    
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
    
    def num_flat_features(self,x):
        features = 1
        for dim in x.size()[1:]:
            features *= dim
        return features


    
# %%
my_net = BasicCNN()
# %%
print(my_net)
# %%
params = list(my_net.parameters())
print(len(params))
print(params[0].size())