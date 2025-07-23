# GitHub Copilot Instructions for Mask R-CNN Project

## Project Overview
This is a Mask R-CNN implementation for X-ray defect detection, specifically focused on GDXray dataset processing. The project combines PyTorch's torchvision detection models with custom dataset handling and augmentation transforms.

## Core Architecture

### Dataset System
- **Primary Dataset**: `GDXrayDataset` in `custom_data.py` handles X-ray images with mask annotations
- **Data Structure**: Images stored as `welding/W0001/W0001_0000.png`, masks as `welding/W0001/masks/W0001_0000_0.png`
- **Metadata Files**: Train/test splits defined in `metadata/gdxray/gdxray_train.txt` and `gdxray_test.txt`
- **Key Pattern**: Dataset expects binary masks (0/1 values) and validates box dimensions to prevent NaN losses

```python
# Standard dataset configuration pattern
config = {
    'name': "gdxray",
    'data_dir': "/path/to/gdxray",
    'metadata': "/path/to/metadata/gdxray",
    'labels': True,
    'image_size': (768, 768),
    'batch_size': 4,
}
```

### Model Architecture
- **Entry Point**: `get_model_instance_segmentation()` in `maskrcnn.py`
- **Base**: Pre-trained MaskRCNN ResNet50-FPN from torchvision
- **Customization**: Replaces classification head for 2 classes (background + defect)
- **Key Components**: Custom ROI heads with `FastRCNNPredictor` and `MaskRCNNPredictor`

### Training Pipeline
- **Main Script**: `trainner.py` (note the double 'n')
- **Engine**: Uses `mrcnn.engine.train_one_epoch()` and `evaluate()`
- **Jupyter Integration**: Scripts use `%load_ext autoreload` for notebook-style development
- **Loss Monitoring**: Built-in NaN detection stops training when losses become non-finite

## Critical Workflows

### Training Setup
```python
# Standard training pattern
from mrcnn.engine import train_one_epoch, evaluate
import custom_data
import maskrcnn

# Create datasets with proper validation
train_dataset = custom_data.GDXrayDataset(config, subset="train", labels=True, transform=transform)
model = maskrcnn.get_model_instance_segmentation(num_classes=2)

# Use SGD with specific hyperparameters
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
```

### Data Validation Patterns
- **Box Validation**: Always check `(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])`
- **Mask Validation**: Ensure masks contain only 0/1 values using `torch.unique(masks)`
- **Area Validation**: Filter out zero-area bounding boxes to prevent division by zero
- **Coordinate System**: Uses XYXY format `[x1, y1, x2, y2]` with 0-based indexing

### Transform System
- **Location**: Custom transforms in `mrcnn/transforms.py`
- **Integration**: Uses torchvision v2 transforms with `tv_tensors` for proper bbox/mask handling
- **Key Classes**: `FixedSizeCrop`, `RandomIoUCrop`, `ResizeARwPad` for aspect-ratio preserving resize
- **Coordinate Updates**: All transforms must update both `target["boxes"]` and `target["masks"]`

## Project-Specific Conventions

### Error Handling
- **NaN Protection**: Training automatically stops on non-finite losses with diagnostic output
- **Gradient Clipping**: Use `torch.nn.utils.clip_grad_norm_` to prevent gradient explosion
- **Empty Sample Handling**: Filter out samples with no valid masks/boxes before training
- **Prevent Average Precision (APs) equal 1.0**: This can occur if all predictions are correct, leading to misleading evaluation metrics. Ensure that the dataset has enough variability and that the model is not overfitting.

### File Structure Patterns
- **Config-Driven**: All paths and hyperparameters defined in config dictionaries
- **Modular**: Core functionality split between `maskrcnn.py` (models), `custom_data.py` (datasets), `mrcnn/` (utilities)
- **Logging**: Models saved to timestamped directories: `logs/gdxray_YYYY-MM-DD_HH-MM-SS/`

### Data Loading Specifics
- **Collate Function**: Use `mrcnn.utils.collate_fn` for proper batch handling with variable-size targets
- **Mask Loading**: Searches for masks up to index 100: `W0001_0000_0.png`, `W0001_0000_1.png`, etc.
- **Metadata Format**: Simple text files with one image path per line

### Debugging and Visualization
- **COCO Integration**: Uses `mrcnn.coco_utils` for evaluation metrics compatible with COCO API
- **Jupyter Notebooks**: `penn.ipynb`, `pipeline.ipynb` for interactive development and visualization
- **Visualization Utils**: `mrcnn.utils.visualize_batch()` for dataset inspection

## Common Pitfalls
1. **Canvas Size**: BoundingBoxes require `canvas_size=(height, width)` tuple, not list
2. **Mask Values**: Non-binary masks (values other than 0/1) cause loss explosion
3. **Empty Targets**: Always validate that samples have at least one valid box/mask
4. **Transform Consistency**: Ensure image transforms are applied identically to boxes and masks
5. **Device Placement**: Move all tensor components to device: `{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}`

## Development Environment
- **Python**: Uses conda environment with PyTorch + CUDA
- **Jupyter Integration**: Scripts designed for both standalone and notebook execution
- **Hot Reloading**: `%autoreload 2` enables live code updates during development
