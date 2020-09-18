from torchvision import transforms
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import skimage
from prettytable import PrettyTable
from gpuinfo import GPUInfo
import cv2
import json

def set_GPU(num_of_GPUs):

    current_memory_gpu = GPUInfo.gpu_usage()[1]
    list_available_gpu = np.where(np.array(current_memory_gpu) < 1500)[0].astype('str').tolist()
    current_available_gpu = ",".join(list_available_gpu)
    if len(list_available_gpu) < num_of_GPUs:
        print("==============Warning==============")
        print("Your process had been terminated")
        print("Please decrease number of gpus you using")
        print(f"number of Devices available:\t{len(list_available_gpu)} gpu(s)")
        print(f"number of Device will use:\t{num_of_GPUs} gpu(s)")
        sys.exit()
    elif len(list_available_gpu) > num_of_GPUs:
        redundant_gpu = len(list_available_gpu) - num_of_GPUs
        list_available_gpu = list_available_gpu[:redundant_gpu]
        current_available_gpu = ",".join(list_available_gpu)
        print("[DEBUG]***********************************************")
        print(f"[DEBUG]You are using GPU(s): {current_available_gpu}")
        print("[DEBUG]***********************************************")
        os.environ["CUDA_VISIBLE_DEVICES"] = current_available_gpu
    else: 
        print("[DEBUG]***********************************************")
        print(f"[DEBUG]You are using GPU(s): {current_available_gpu}")
        print("[DEBUG]***********************************************")
        os.environ["CUDA_VISIBLE_DEVICES"] = current_available_gpu

def preprocess_input(image, advprop=False):
    if advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                        std=[0.229, 0.224, 0.225])
    preprocess_image = transforms.Compose([transforms.ToTensor(), normalize])(image)
    return preprocess_image

def to_onehot(labels, num_of_classes):
    if type(labels) is list:
        labels = [int(label) for label in labels]
        arr = np.array(labels, dtype=np.int)
        onehot = np.zeros((arr.size, num_of_classes))
        onehot[np.arange(arr.size), arr] = 1
    else:
        onehot = np.zeros((num_of_classes,), dtype=np.int)
        onehot[int(labels)] = 1
    return onehot

def multi_threshold(Y, thresholds):
    if Y.shape[-1] != len(thresholds):
        raise ValueError('Mismatching thresholds and output classes')

    thresholds = np.array(thresholds)
    thresholds = thresholds.reshape((1, thresholds.shape[0]))
    keep = Y > thresholds
    score = keep * Y
    class_id = np.argmax(score, axis=-1)
    class_score = np.max(score, axis=-1)
    if class_score == 0:
        return None
    return class_id, class_score

def load_and_crop(image_path, input_size=0, custom_size=None):
    """ Load image and return image with specific crop size

    This function will crop corresponding to json file and will resize respectively input_size

    Input:
        image_path : Ex:Dataset/Train/img01.bmp
        input_size : any specific size
        
    Output:
        image after crop and class gt
    """
    image = cv2.imread(image_path)
    print(image_path)
    json_path = image_path + ".json"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size_image = image.shape

    try :
        with open(json_path, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            box = json_data['box']
            center_x = box['centerX'][0]
            center_y = box['centerY'][0]
            widthBox = box['widthBox'][0]
            heightBox = box['heightBox'][0]
            class_gt = json_data['classId'][0]
    except:
        print(f"Can't find or missing some fields: {json_path}")
        # Crop center image if no json found
        center_x = custom_size[0]
        center_y = custom_size[1]
        widthBox = 0
        heightBox = 0
        class_gt = "Empty"

    new_w = max(widthBox, input_size)
    new_h = max(heightBox, input_size)

    left, right = center_x - new_w / 2, center_x + new_w / 2
    top, bottom = center_y - new_h / 2, center_y + new_h / 2

    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(size_image[1] - 0, right)), round(min(size_image[0] - 0, bottom))

    cropped_image = image[int(top):int(bottom), int(left):int(right)]

    # if input_size > new_w:
    #     changed_image = cv2.resize(cropped_image,(input_size, input_size))
    # else:
    #     changed_image = cropped_image

    return cropped_image, class_gt

def metadata_count(input_dir,classes_name_list, label_list, show_table):
    Table = PrettyTable()
    print(f"[DEBUG] : {input_dir}")
    # print(classes_name_list)
    # print(label_list)
    Table.field_names = ['Defect', 'Number of images']
    unique_label ,count_list = np.unique(label_list, return_counts=True)
    # print(count_list)
    for i in range(len(classes_name_list)):
        for j in range(len(unique_label)):
            if classes_name_list[i] == unique_label[j] :
                Table.add_row([classes_name_list[i], count_list[j]])
    if show_table :
        print(f"[DEBUG] :\n{Table}")
    return classes_name_list, label_list

class FocalLoss(nn.Module):
    # Took from : https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
    # Addition resource : https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    # TODO: clean up FocalLoss class
    def __init__(self, class_weight=1., gamma=2., logits = False, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.class_weight = class_weight
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        # if self.logits:
        #     BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        # else:
        #     BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        # pt = torch.exp(-BCE_loss)
        # F_loss = self.class_weight * (1 - pt)**self.gamma * BCE_loss
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            targets,
            weight = self.class_weight,
            reduction = self.reduction
        )
# TODO: inspect resize_image more carefully
def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    import cv2

    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max


    # Resize image using bilinear interpolation
    if scale != 1:

        # image = cv2.resize(image, (round(h*scale), round(w*scale)), interpolation=cv2.INTER_LINEAR)
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)
    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))

    return image.astype(image_dtype), window, scale, padding, crop

class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))\
                for device_idx in range(len(devices))], [kwargs] * len(devices)

def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate