import os
import datetime
import torch
from engine import train_one_epoch, evaluate

import custom_data
import maskrcnn

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d_%H-%M-%S')
print(f"Time of execution: {now}")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

config = {
        'name': "gdxray",
        'data_dir': os.path.join(PARENT_DIR, "data/gdxray"),
        'metadata': os.path.join(PARENT_DIR, "metadata"),
        'subset': "train",
        'labels': True,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'image_size': (320, 640),
        'learning_rate': 1e-4,
        'batch_size': 4,
        'epochs': 1500,
        'save_dir': os.path.join(PARENT_DIR, "logs/gdxray")
   }

train_dataset = custom_data.GDXrayDataset(config, labels=config['labels'], transform=maskrcnn.get_transform(train=True))
valid_dataset = custom_data.GDXrayDataset(config, labels=config['labels'], transform=maskrcnn.get_transform(train=False))

# split the dataset in train and test set
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                              num_workers=5, pin_memory=False,
                              batch_size=config['batch_size'],
                              shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=valid_dataset,
                              num_workers=5, pin_memory=False,
                              batch_size=config['batch_size'],
                              shuffle=True)

# get the model using our helper function
model = maskrcnn.get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 10 epochs (updated from 2 to 10)
num_epochs = 100

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    if epoch % 100 == 0:
        torch.save(model.state_dict(), os.path.join(config['save_dir'], f"model_{epoch}.pth"))
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed.")

print("That's it!")

# visualize the results

