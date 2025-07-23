import datetime
import errno
import os
import time
from collections import defaultdict, deque
import numpy as np

import torch
import torch.distributed as dist

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def create_color_mask(masks):
    """Tạo mask màu với mỗi instance một màu khác nhau"""
    n_instances = len(masks)
    if n_instances == 0:
        return np.zeros((*masks.shape[-2:], 3))

    # Tạo bảng màu ngẫu nhiên cho các instance
    colors = cm.get_cmap('rainbow')(np.linspace(0, 1, n_instances))[:, :3]

    # Khởi tạo mask màu
    colored_mask = np.zeros((*masks.shape[-2:], 3))

    # Tô màu cho từng instance
    for i, mask in enumerate(masks):
        for c in range(3):
            colored_mask[:, :, c] = np.where(mask == 1,
                                             colored_mask[:, :, c] * 0.5 + colors[i][c] * 0.5,
                                             colored_mask[:, :, c])

    return colored_mask

def visualize_sample(dataset, idx=0, save_path=None):
    # Lấy một mẫu từ dataset
    image, target = dataset[idx]

    # Đảo ngược chuẩn hóa
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # image = image * std + mean
    # image = image.clamp(0, 1)

    # Chuyển đổi sang numpy và đổi kênh
    image = image.permute(1, 2, 0).numpy()

    # Tạo figure với 3 subplot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Generate colors for instances
    n_instances = len(target['masks']) if 'masks' in target else 0
    colors = cm.get_cmap('rainbow')(np.linspace(0, 1, n_instances))[:, :3] if n_instances > 0 else []

    # 1. Hiển thị ảnh gốc với bounding boxes (cùng màu với mask)
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')

    if 'boxes' in target and 'masks' in target:
        boxes = target['boxes'].numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = colors[i] if i < len(colors) else 'red'
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, color=color, linewidth=2, linestyle='--', alpha=0.8)
            ax1.add_patch(rect)
    ax1.axis('off')

    # 2. Hiển thị mask tổng hợp (trắng đen)
    if 'masks' in target:
        combined_mask = torch.any(target['masks'], dim=0).numpy()
        ax2.imshow(combined_mask, cmap='gray')
        ax2.set_title('Combined Mask')
        ax2.axis('off')

    # 3. Hiển thị mask nhiều màu với bounding boxes cùng màu
    if 'masks' in target:
        masks = target['masks'].numpy()
        colored_masks = create_color_mask(masks)

        # Kết hợp ảnh gốc với mask màu
        overlay = image.copy()
        mask_pixels = colored_masks.sum(axis=-1) > 0
        overlay[mask_pixels] = image[mask_pixels] * 0.3 + colored_masks[mask_pixels] * 0.7

        ax3.imshow(overlay)

        # Add colored bounding boxes to the overlay
        if 'boxes' in target:
            boxes = target['boxes'].numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                color = colors[i] if i < len(colors) else 'red'
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, color=color, linewidth=2, linestyle='--', alpha=0.9)
                ax3.add_patch(rect)

        ax3.set_title(f'Colored Instances (n={len(masks)})')
        ax3.axis('off')

    plt.suptitle(f'Sample {idx}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_batch(dataset, start_idx=0, num_samples=5, save_dir=None):
    """Hiển thị nhiều mẫu từ dataset"""
    for i in range(start_idx, start_idx + num_samples):
        if i >= len(dataset):
            break
        if save_dir:
            save_path = os.path.join(save_dir, f'sample_{i}.png')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_path = None
        visualize_sample(dataset, i, save_path)
