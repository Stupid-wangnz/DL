from pytest import xfail
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from einops.layers.torch import Rearrange
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

batch_size = 32

train_dataset = datasets.MNIST('/codes/DeepLearning/DL/FFN/data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('/codes/DeepLearning/DL/FFN/data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)

class MLPMixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patches, token_dim),
            Rearrange('b d n -> b n d')
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim)
        )

    def forward(self, x):
        x = x + self.token_mlp(x)
        x = x + self.channel_mlp(x)
        return x
        

class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, token_dim, channel_dim):
        super(MLPMixer, self).__init__()
        assert image_size % patch_size == 0

        self.num_patches = (image_size // patch_size) **2

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MLPMixerBlock(dim, self.num_patches, token_dim, channel_dim))
        
        self.ln = nn.LayerNorm(dim)

        self.fc = nn.Linear(dim, 10)
    
    def forward(self, x):
        x = self.patch_embedding(x)

        for block in self.mixer_blocks:
            x = block(x)

        x = self.ln(x)
        x = x.mean(dim=1)
        x = F.log_softmax(self.fc(x), dim=1)
        return x

model = MLPMixer(image_size=28, channels=1, patch_size=7, dim=32, depth=4, token_dim=64, channel_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()  
        
        # Update weights
        optimizer.step()    #  w - alpha * dL / dw
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
    
epochs = 10

if __name__ == "__main__":
    print(model)

    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(lossv, accv)