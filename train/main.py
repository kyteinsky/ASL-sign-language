import sys
sys.path.insert(0,'.')

import torch
import torchvision.transforms as transforms
from model import Net
from config import *


transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.RandomRotation(degrees=5),
     transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

trainset = dataset('sign_mnist_train/sign_mnist_train.csv', ds_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=1)

testset = dataset('sign_mnist_test/sign_mnist_test.csv', ds_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                          shuffle=True, num_workers=1)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'] # no 'J' or 'Z'

if wb:
    wandb.init(project="sign-lang")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps = 0
print_every = 1
running_loss = 0
train_losses, test_losses = [], []


net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(epochs):
    for inputs in trainloader:
        steps += 1
        inputs, labels = (inputs['image']).to(device), (inputs['label']).to(device)
        optimizer.zero_grad()
        out = net(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            net.eval()
            with torch.no_grad():
                for inputs in testloader:
                    inputs, labels = (inputs['image']).to(device), (inputs['label']).to(device)
                    out = net(inputs)
                    batch_loss = criterion(out, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(out)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            if wb:
                wandb.log({'Train Loss':running_loss/print_every, 'Test Loss': test_loss/len(testloader), 'Test Accuracy':accuracy/len(testloader)})

            running_loss = 0
            net.train()
dest = f'sign_lang_lr_{lr}_epo_{epochs}_'
v = 0
while os.path.isfile(dest+str(v)):
    v += 1
torch.save(net, dest+str(v)+'.pth')
