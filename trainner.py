%load_ext autoreload
%autoreload 2

import os
import datetime
import torch
from mrcnn.engine import train_one_epoch, evaluate

import custom_data
import maskrcnn
import mrcnn.utils as utils

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d_%H-%M-%S')
print(f"Time of execution: {now}")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = CURRENT_DIR

config = {
        'name': "gdxray",
        'data_dir': os.path.join("/home/tuank/data/gdxray"),
        'metadata': os.path.join(PARENT_DIR, "metadata/gdxray"),
        'labels': True,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'image_size': (768, 768),
        'learning_rate': 1e-4,
        'batch_size': 4,
        'epochs': 100,
        'save_dir': os.path.join(PARENT_DIR, "logs/gdxray_" + now),
   }

# config = {
#         'name': "pennfudan",
#         'data_dir': os.path.join("/home/tuank/data/penn"),
#         'device': "cuda" if torch.cuda.is_available() else "cpu",
#         'image_size': (768, 768),
#         'learning_rate': 1e-4,
#         'batch_size': 2,
#         'epochs': 100,
#         'save_dir': os.path.join(PARENT_DIR, "logs/pennfudan_" + now),
# }

train_dataset = custom_data.GDXrayDataset(config, subset="train", labels=config['labels'], transform=custom_data.get_transform(size = config['image_size'], train=True))
valid_dataset = custom_data.GDXrayDataset(config, subset="test", labels=config['labels'], transform=custom_data.get_transform(size = config['image_size'], train=False))

# train_dataset = custom_data.PennFudanDataset(config['data_dir'], transforms=custom_data.get_transform(size=config['image_size'], train=True))
# valid_dataset = custom_data.PennFudanDataset(config['data_dir'], transforms=custom_data.get_transform(size=config['image_size'], train=False))

# indices = torch.randperm(len(train_dataset)).tolist()
# train_dataset = torch.utils.data.Subset(train_dataset,  indices[:-50])
# valid_dataset = torch.utils.data.Subset(valid_dataset, indices[-50:])

# split the dataset in train and test set
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=utils.collate_fn)


# get the model using our helper function
model = maskrcnn.get_model_instance_segmentation(num_classes)

# move model to the right device
device = torch.device(config['device'])
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.0001,
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
num_epochs = config['epochs']

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    if epoch % 10 == 0:
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        torch.save(model.state_dict(), os.path.join(config['save_dir'], f"model_{epoch}.pth"))
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed.")

    if epoch == num_epochs - 1:
        torch.save(model.state_dict(), os.path.join(config['save_dir'], "model_final.pth"))

print("That's it!")


#%%

import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F

image, _ = valid_dataset[5]
image = read_image(config['data_dir'] + "/PNGImages/FudanPed00070.png")
eval_transform = custom_data.get_transform(config['image_size'],train=False)

model.eval()
with torch.no_grad():
    x = eval_transform(image)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]


image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))

# %%
