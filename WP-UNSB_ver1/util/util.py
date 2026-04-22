"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if torch.is_tensor(input_image):
            image_tensor = input_image.detach()
        else:
            return input_image

        # ★バッチ次元だけ落とす（4次元のときだけ）
        if image_tensor.dim() == 4:  # (B,C,H,W)
            image_tensor = image_tensor[0]
        # 3次元 (C,H,W) はそのまま

        image_numpy = image_tensor.cpu().float().numpy()
        # (C,H,W) -> (H,W,C)
        if image_numpy.ndim == 3:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        elif image_numpy.ndim == 2:
            # (H,W) のときはそのままスケーリング（必要なら）
            image_numpy = (image_numpy + 1) / 2.0 * 255.0
        else:
            raise ValueError(f"tensor2im: unexpected ndim={image_numpy.ndim}, shape={image_numpy.shape}")

        image_numpy = np.clip(image_numpy, 0, 255)
        return image_numpy.astype(imtype)
    else:
        return input_image


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    image_numpy: (H,W) or (H,W,1) or (H,W,3) or (H,W,4)
    """
    # dtype を uint8 に寄せる（tensor2imがuint8ならそのままでOK）
    if image_numpy.dtype != np.uint8:
        image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)

    # shape 整形（PILが嫌う (H,W,1) を潰す）
    if image_numpy.ndim == 3 and image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]  # (H,W,1) -> (H,W)

    # PIL生成
    image_pil = Image.fromarray(image_numpy)

    # aspect ratio（※ PILの size は (w,h)）
    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        w, h = image_pil.size
        image_pil = image_pil.resize((int(w * aspect_ratio), h), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        w, h = image_pil.size
        image_pil = image_pil.resize((w, int(h / aspect_ratio)), Image.BICUBIC)

    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)
