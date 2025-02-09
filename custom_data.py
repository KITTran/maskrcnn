import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class GDXrayDataset(torch.utils.data.Dataset):
    """
    Dataset of Xray Images

    Images are referred to using their image_id (relative path to image).
    An example image_id is: "Weldings/W0001/W0001_0004.png"
    """

    def __init__(self, config: dict, masked: bool, transform):
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
        self.masked = masked
        self.transform = transform

        # Initialize the dataset infos
        self.image_info = []
        self.image_indices = {}
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]

        # Add classes
        self.add_class(config["name"], 1, "Defect")

        # Load the dataset
        metadata_img = "{}/{}_{}.txt".format(config['metadata'], config["name"], "images")
        metadata_label = "{}/{}_{}.txt".format(config['metadata'], config["name"], "labels") if self.masked else None

        # Load image ids from key 'image' in dictionary in metadata file
        image_ids = []
        image_ids.extend(self.load_metadata(metadata_img, "image"))
        if self.masked:
            label_ids = []
            label_ids.extend(self.load_metadata(metadata_label, "label"))

        for i, image_id in enumerate(image_ids):
            img_path = os.path.join(config["data_dir"], image_id)
            mask_path = ""
            if self.masked:
                mask_path = os.path.join(config["data_dir"], label_ids[i])

                if not os.path.exists(mask_path):
                    print("Skipping ", image_id, " Reason: No mask")

                    del self.image_ids[i]
                    del self.label_ids[i]

                    continue

            print("Adding image: ", image_id)

            self.add_image(config["name"], config['subset'], image_id, img_path, mask=mask_path)

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to fetch.

        Returns:
            tuple: (image, label) where both are transformed.
        """

        image_path = self.image_info[idx]["path"]
        image = read_image(image_path)
        width, height = image.shape[-2], image.shape[-1]
        self.update_info(idx, height_org=height, width_org=width)

        mask_path = self.image_info[idx]["mask"]
        mask = read_image(mask_path)  # read as grayscale

        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]  # remove background
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # get the labels for each mask
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_indice = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and target into torchvision tv_tensors
        image = tv_tensors.Image(image)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(width, height))
        target["masks"] = tv_tensors.Mask(masks)
        target["area"] = area
        target["labels"] = labels
        target["image_indice"] = image_indice
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target) if self.masked else self.transform(image)

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
