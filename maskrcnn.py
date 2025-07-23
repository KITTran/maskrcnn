import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def torchvisionexample():
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (defect) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    ###### Customize the model to use a different backbone ######
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # ``FasterRCNN`` needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64),),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0),)
    )

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # put the pieces together inside a Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

def get_model_resnet101_fpn(num_classes):
    # Load a pre-trained backbone model and return only the features
    backbone = torchvision.models.resnet101(weights="DEFAULT")
    # Remove the last fully connected layer
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    # Set the number of output channels in the backbone
    backbone.out_channels = 2048


    # Define the anchor generator with specific sizes and aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Define the region of interest (RoI) pooling layer using pyramid ROI align
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    # Put the pieces together inside a Mask R-CNN model
    model = torchvision.models.detection.MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        min_size=256, max_size=768,

    )

    return model

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

if __name__ == "__main__":
    import os
    import mrcnn.utils as utils
    import custom_data
    from mrcnn.coco_eval import CocoEvaluator
    from mrcnn.coco_utils import get_coco_api_from_dataset

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
            'save_dir': os.path.join(PARENT_DIR, "logs/gdxray_"),
    }

    # config = {
    #     'name': "pennfudan",
    #     'data_dir': os.path.join("/home/tuank/data/penn"),
    #     'device': "cuda" if torch.cuda.is_available() else "cpu",
    #     'image_size': (768, 768),
    #     'learning_rate': 1e-4,
    #     'batch_size': 2,
    #     'epochs': 100,
    #     'save_dir': os.path.join(PARENT_DIR, "logs/pennfudan_"),
    # }

    dataset = custom_data.GDXrayDataset(config, subset="train", labels=config['labels'], transform=custom_data.get_transform(size = config['image_size'], train=True))
    # dataset = custom_data.PennFudanDataset(config['data_dir'], transforms=custom_data.get_transform(size=config['image_size'], train=True))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes=2)
    device = torch.device(config['device'])
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    print(output)

    # For inference
    model.eval()
    # x = [torch.rand(3, 768, 768), torch.rand(3, 768, 768)]
    x, _ = dataset[0]
    predictions = model([x])  # Returns predictions
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(predictions[0])

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Visualization of all predictions
    if len(predictions[0]['boxes']) > 0:
        # Convert tensor image to numpy for visualization
        image_np = x.permute(1, 2, 0).cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        # Define colors for different predictions
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta']

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Image with all bounding boxes
        axes[1].imshow(image_np)
        for i in range(len(predictions[0]['boxes'])):
            box = predictions[0]['boxes'][i].detach().numpy()
            score = predictions[0]['scores'][i].detach().item()
            color = colors[i % len(colors)]

            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            axes[1].add_patch(rect)

            # Add score text
            axes[1].text(box[0], box[1]-5, f'{score:.3f}',
                        color=color, fontsize=10, fontweight='bold')

        axes[1].set_title(f'All Bounding Boxes ({len(predictions[0]["boxes"])} predictions)')
        axes[1].axis('off')

        # All masks overlay
        axes[2].imshow(image_np)
        combined_mask_overlay = np.zeros((*image_np.shape[:2], 4))

        # Color mapping for masks (RGB values)
        color_map = [
            [1, 0, 0],      # red
            [0, 0, 1],      # blue
            [0, 1, 0],      # green
            [1, 1, 0],      # yellow
            [1, 0.5, 0],    # orange
            [0.5, 0, 1],    # purple
            [0, 1, 1],      # cyan
            [1, 0, 1]       # magenta
        ]

        for i in range(len(predictions[0]['masks'])):
            mask = predictions[0]['masks'][i][0].detach().numpy()  # First channel
            color_rgb = color_map[i % len(color_map)]

            # Create mask overlay with specific color
            mask_overlay = np.zeros((*mask.shape, 4))
            mask_binary = mask > 0.5
            mask_overlay[mask_binary] = [*color_rgb, 0.6]  # RGB + alpha

            # Combine with existing overlays
            combined_mask_overlay = np.maximum(combined_mask_overlay, mask_overlay)

        axes[2].imshow(combined_mask_overlay)
        axes[2].set_title(f'All Mask Predictions ({len(predictions[0]["masks"])} masks)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Prediction summary:")
        print(f"  Total predictions: {len(predictions[0]['boxes'])}")
        for i in range(len(predictions[0]['boxes'])):
            box = predictions[0]['boxes'][i].detach().numpy()
            score = predictions[0]['scores'][i].detach().item()
            mask = predictions[0]['masks'][i][0].detach().numpy()
            color = colors[i % len(colors)]

            print(f"  Prediction {i+1} ({color}):")
            print(f"    Box (x1, y1, x2, y2): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            print(f"    Score: {score:.4f}")
            print(f"    Mask shape: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}]")
    else:
        print("No predictions found!")

    # For evaluation - step by step debugging
    print("\n=== EVALUATION DEBUG ===")

    # Step 1: Get ground truth data properly
    model.eval()
    with torch.no_grad():
        # Get a single sample with ground truth
        image, target = dataset[0]
        print(f"Step 1 - Ground truth target keys: {target.keys()}")
        print(f"  - image_id: {target['image_id']}")
        print(f"  - boxes shape: {target['boxes'].shape}")
        print(f"  - masks shape: {target['masks'].shape}")
        print(f"  - labels: {target['labels']}")

        # Step 2: Get model prediction for the same image
        prediction = model([image])[0]
        print(f"\nStep 2 - Prediction keys: {prediction.keys()}")
        print(f"  - boxes shape: {prediction['boxes'].shape}")
        print(f"  - masks shape: {prediction['masks'].shape}")
        print(f"  - scores: {prediction['scores']}")
        print(f"  - labels: {prediction['labels']}")

        # Step 3: Filter predictions by confidence threshold
        confidence_threshold = 0.5
        keep = prediction['scores'] >= confidence_threshold
        filtered_prediction = {
            'boxes': prediction['boxes'][keep],
            'masks': prediction['masks'][keep],
            'scores': prediction['scores'][keep],
            'labels': prediction['labels'][keep]
        }
        print(f"\nStep 3 - After filtering (conf >= {confidence_threshold}):")
        print(f"  - Kept {keep.sum().item()} out of {len(keep)} predictions")

        # Step 4: Prepare evaluation data
        if len(filtered_prediction['boxes']) > 0:
            # Create proper evaluation format
            evaluation_target = {
                'image_id': target['image_id'],
                'boxes': target['boxes'],
                'masks': target['masks'],
                'labels': target['labels'],
                'area': (target['boxes'][:, 2] - target['boxes'][:, 0]) * (target['boxes'][:, 3] - target['boxes'][:, 1]),
                'iscrowd': torch.zeros(len(target['boxes']), dtype=torch.uint8)
            }

            evaluation_prediction = {
                'boxes': filtered_prediction['boxes'],
                'masks': filtered_prediction['masks'],
                'scores': filtered_prediction['scores'],
                'labels': filtered_prediction['labels']
            }

            print(f"\nStep 4 - Evaluation format prepared:")
            print(f"  - GT boxes: {evaluation_target['boxes'].shape}")
            print(f"  - Pred boxes: {evaluation_prediction['boxes'].shape}")

            # Step 5: Run evaluation
            try:
                coco = get_coco_api_from_dataset(dataset)
                iou_types = ["bbox", "segm"]
                coco_evaluator = CocoEvaluator(coco, iou_types)

                # Format for COCO evaluator
                res = {evaluation_target["image_id"].item(): evaluation_prediction}
                coco_evaluator.update(res)
                coco_evaluator.synchronize_between_processes()
                coco_evaluator.accumulate()
                coco_evaluator.summarize()

                print("\nStep 5 - Evaluation completed successfully")

            except Exception as e:
                print(f"\nStep 5 - Evaluation failed with error: {e}")
                print("This is expected for single-sample evaluation")

                # Alternative: Manual IoU calculation
                print("\nFallback - Manual IoU calculation:")
                if len(evaluation_target['boxes']) > 0 and len(evaluation_prediction['boxes']) > 0:
                    from torchvision.ops import box_iou
                    ious = box_iou(evaluation_target['boxes'], evaluation_prediction['boxes'])
                    max_iou = torch.max(ious).item()
                    print(f"  - Max IoU between GT and predictions: {max_iou:.4f}")

                    # Count matches at different IoU thresholds
                    for threshold in [0.5, 0.75, 0.9]:
                        matches = (ious > threshold).any(dim=1).sum().item()
                        total_gt = len(evaluation_target['boxes'])
                        print(f"  - Matches at IoU {threshold}: {matches}/{total_gt} ({matches/total_gt:.2%})")
                else:
                    print("  - No valid boxes for comparison")
        else:
            print("\nStep 4 - No predictions above confidence threshold, skipping evaluation")

        # Use coco API for evaluation
        coco_api = get_coco_api_from_dataset(dataset)
        coco_evaluator = CocoEvaluator(coco_api, ["bbox", "segm"])
        # get all image in dataset for evaluation
        predictions = []
        for idx in range(len(dataset)):
            image, _ = dataset[idx]
            # image = image.to(device)
            with torch.no_grad():
                prediction = model([image])[0]
            predictions.append(prediction)
        coco_evaluator.update(predictions)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
