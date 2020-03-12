import time
import PIL
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import apex
from tqdm import tqdm
import multiprocessing
num_threads = 2
BATCH_SIZE = 256
EPOCHS = 10
NN_UNITS = 256
LEARNING_RATE = 0.001
AMP_LEVEL = "O1"
PATH = "./cifar_net.pth"

print(" Configuration ")
print("===============")
print("   Batch size:", BATCH_SIZE)
print("  CPU threads:", num_threads)
print("     NN units:", NN_UNITS)
print("Learning rate:", LEARNING_RATE)
print("    AMP level:", AMP_LEVEL)
print("       Epochs:", EPOCHS)
print("")
print("Model to be saved to", PATH)

print("\n\n")

def return_model(num_class, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    return model

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])
    
def main():
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.RandomAffine(5, scale=(0.9,1.1), resample=PIL.Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root="~/data", train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=num_threads)

    testset = torchvision.datasets.CIFAR10(root="~/data", train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=num_threads)

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    model = Net(units=NN_UNITS).cuda()
    print("Number of parameters:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=LEARNING_RATE)

    model, optimizer = apex.amp.initialize(model, optimizer, opt_level=AMP_LEVEL)

    interval = 50
    train_steps = int(len(trainset)/BATCH_SIZE)
    img_per_epoch = train_steps*BATCH_SIZE
    
    fps_list = []

    for epoch in range(EPOCHS):
        print("\nEpoch: "+str(epoch+1)+"/"+str(EPOCHS))

        running_loss = 0.0
        
        epoch_start = time.time()

        for i, data in tqdm(enumerate(trainloader), total=train_steps):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step = i+1
            if step % interval == 0:
                print(step, running_loss/interval)
                running_loss = 0.0
                
        duration = time.time() - epoch_start
        fps = int(img_per_epoch/duration)
        fps_list.append(fps)
        print("Images/sec:", fps)
    
    print("\n\n")
    print("Finished training!")
    print("==================")
    avg_fps = np.mean(np.asarray(fps_list))
    print("Average images/sec:", int(avg_fps))
    torch.save(model.state_dict(), PATH)
    print("Model saved to", PATH)

    
if __name__ == "__main__":
    st = time.time()
    main()
    et = time.time()
    print("End-to-end time:", int(et-st))
