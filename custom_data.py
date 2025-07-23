import os
import torch
import numpy as np

from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

# Fix load label issue: load as instance segmentation
from torchvision.transforms.v2 import functional as F

import mrcnn.transforms as MT
from mrcnn.utils import visualize_batch

class GDXrayDataset(torch.utils.data.Dataset):
    """
    Dataset of Xray Images

    Images are referred to using their image_id (relative path to image).
    An example image_id is: "Weldings/W0001/W0001_0004.png"
    """

    def __init__(self, config: dict, subset: str, labels: bool, transform):
        super().__init__()
        """
        Args:
            config (dict): Contain nessesary information for the dataset
            config = {
                'name': str, # Name of the dataset
                'data_dir': str, # Path to the data directory
                'subset': str, # 'train' or 'val'
                'metadata': str, # Path to the metadata file
                }
            masked (bool): Whether to load masked
            transform (callable, optional): Transform to be applied to the images.
        """

        self.config = config
        self.subset = subset
        self.labels = labels
        self.transform = transform

        # Initialize the dataset infos
        self.image_info = []
        self.image_indices = {}
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]

        # Add classes
        self.add_class(config["name"], 1, "Defect")

        # Load the dataset in metadata file
        metadata_img = "{}/{}_{}.txt".format(config['metadata'], config["name"], self.subset)

        # Load image ids from key 'image' in dictionary in metadata file
        image_ids = []
        image_ids.extend(self.load_metadata(metadata_img, "image"))

        for i, image_id in enumerate(image_ids):
            img_path = os.path.join(config["data_dir"], image_id)
            if self.labels:
                mask_path = self.get_mask_path(img_path, 0)  # Get the first mask path

                if not os.path.exists(mask_path):
                    print('Mask file does not exist: ', mask_path)
                    print("Skipping ", image_id, " Reason: No mask")

                    continue

            print("Adding image: ", image_id)

            self.add_image(
                source=config["name"],
                subset=self.subset,
                image_id=image_id,
                path=img_path)

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to fetch.

        Returns:
            tuple: (image, label) where both are transformed.
            "boxes": ...,      # Shape: (N, 4)
            "masks": ...,      # Shape: (N, H, W)
            "labels": ...,     # Shape: (N,)
            "area": ...,       # Shape: (N,)
            "image_indice": ..., # Scalar
            "iscrowd": ...     # Shape: (N,)
        """

        image_path = self.image_info[idx]["path"]
        image = read_image(image_path, ImageReadMode.RGB)
        width, height = image.shape[-2], image.shape[-1]
        self.update_info(idx, height_org=height, width_org=width)

        masks = []
        for i in range(100):
            mask_path = self.get_mask_path(image_path, i)
            if not os.path.exists(mask_path):
                break

            mask = read_image(mask_path).squeeze()  # read as grayscale
            mask = mask.to(torch.uint8)  # change to uint8 mask

            # Ensure mask is binary (0 or 1)
            mask = (mask > 0).to(torch.uint8)

            # Skip empty masks
            if mask.sum() == 0:\

                continue

            masks.append(mask)

        masks = torch.stack(masks) if masks else torch.empty((0, height, width), dtype=torch.uint8)

        # get bounding box coordinates for each mask
        bbox = self.extract_bboxes(masks.numpy())
        # bbox = masks_to_boxes(masks)

        boxes = torch.as_tensor(bbox, dtype=torch.float32)

        # get the labels for each mask
        labels = torch.ones((len(masks),), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.ones((len(masks),), dtype=torch.int64)

        # Wrap sample and target into torchvision tv_tensors
        image = tv_tensors.Image(image)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image[0]))
        target["masks"] = tv_tensors.Mask(masks)
        target["area"] = area
        target["labels"] = labels
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target) if self.labels else self.transform(image)

        return image, target

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
            }
        )

    def add_image(self, source, subset, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "subset": subset,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        self.image_indices[image_id] = len(self.image_info) - 1

    def update_info(self, image_id, **kwargs):
        info = self.image_info[image_id]
        info.update(kwargs)
        self.image_info[image_id] = info

    def load_metadata(self, metadata, key):
        """
        metadata file has the following format:
        {
            "image": [<image_id>, <image_id>, ...],
            "label": [<label_id>, <label_id>, ...]
        }

        Args:
            metadata (str): Path to the metadata file
            key (str): Key to load from the metadata file
        """

        image_ids = []
        with open(metadata, "r") as metadata_file:
            image_ids += metadata_file.readlines()
        return [p.rstrip() for p in image_ids]

    def get_mask_path(self, image_path, index=0):
        """Return the path to a mask"""
        series_dir = os.path.dirname(image_path)
        mask_dir = os.path.join(series_dir,"masks")
        mask_name = os.path.basename(image_path).replace(".png","_%i.png"%index)
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        return mask_path

    def extract_bboxes(self, mask):
        """Compute bounding boxes from masks.
        mask: [num_instances, height, width]. Mask pixels are either 1 or 0.

        Returns: bbox array [num_instances, (x1, y1, x2, y2)] in XYXY format.
        """
        boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
        for i in range(mask.shape[0]):
            m = mask[i, :, :]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Use negative values or zeros
                # depending on your downstream processing requirements
                x1, x2, y1, y2 = 0, 1, 0, 1  # or keep as 0, 1, 0, 1
            boxes[i] = np.array([x1, y1, x2, y2])
        return boxes.astype(np.int32)

from torchvision.ops.boxes import masks_to_boxes

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(size, train:bool = False):
    transforms = []
    # transforms.append(T.Resize(size))
    if train:
        transforms.append(T.RandomEqualize())
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        # transforms.append(MT.SimpleCopyPaste())
    # transforms.append(MT.ResizeARwPad(min_dim=256, max_dim=768, padding=True, fill=0))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    # transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = CURRENT_DIR

    config = {
        'name': "gdxray",
        'data_dir': os.path.join("/home/tuank/data/gdxray"),
        'metadata': os.path.join(PARENT_DIR, "metadata/gdxray"),
        'labels': True,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'image_size': (320, 640),
        'learning_rate': 1e-4,
        'batch_size': 4,
        'epochs': 100,
        'save_dir': os.path.join(PARENT_DIR, "logs/gdxray_"),
    }

    xray_dataset = GDXrayDataset(config, 'train', labels=config['labels'],
                                              transform=get_transform(size=config['image_size'],
                                                                                  train=True))

    # Hiển thị 5 mẫu đầu tiên và lưu vào thư mục 'visualizations'
    visualize_batch(xray_dataset, start_idx=0, num_samples=2,
                    save_dir=None)

    config = {
        'name': "pennfudan",
        'data_dir': os.path.join("/home/tuank/data/penn"),
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'image_size': (768, 768),
        'learning_rate': 1e-4,
        'batch_size': 2,
        'epochs': 100,
        'save_dir': os.path.join(PARENT_DIR, "logs/pennfudan_"),
    }

    pennfudan_dataset = PennFudanDataset(config['data_dir'], transforms=get_transform(size=config['image_size'], train=True))

    # Hiển thị 5 mẫu đầu tiên và lưu vào thư mục 'visualizations'
    visualize_batch(pennfudan_dataset, start_idx=0, num_samples=2,
                    save_dir=None)

    xray_image, xray_target = xray_dataset[0]
    print("X-ray Image Shape:", xray_image.shape)
    print("X-ray Target:", xray_target)

    penn_image, penn_target = pennfudan_dataset[0]
    print("PennFudan Image Shape:", penn_image.shape)
    print("PennFudan Target:", penn_target)
