# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import tifffile as tiff  
import scipy.io 
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
import h5py
import cv2 
from scipy.linalg import sqrtm
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils_HSI import open_file
import matplotlib.pyplot as plt

DATASETS_CONFIG = {
        'Houston13': {
            'img': 'Houston13.mat',
            'gt': 'Houston13_7gt.mat',
            },
        'Houston18': {
            'img': 'Houston18.mat',
            'gt': 'Houston18_7gt.mat',
            },
        'paviaU': {
            'img': 'paviaU.mat',
            'gt': 'paviaU_7gt.mat',
            },
        'paviaC': {
            'img': 'paviaC.mat',
            'gt': 'paviaC_7gt.mat',
            },
        'Loukia':{
            'img':'Loukia.tif',
            'gt':'Loukia_GT.tif',
        },
        'Dioni':{
            'img':'Dioni.tif',
            'gt':'Dioni_GT.tif',
        },
        'shanghai':{
            'img':'',
            'gt':'',
        },
        'hangzhou':{
            'img':'',
            'gt':'',
        },
        'whu_71':{
            'img':'',
            'gt':'',
        },
        'whu_78':{
            'img':'',
            'gt':'',
        },
    }

# try:
#     from custom_datasets import CUSTOM_DATASETS_CONFIG
#     DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
# except ImportError:
#     pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (173, 216, 230),
    4: (144, 238, 144),
    5: (0, 128, 0),
    6: (50, 205, 50),
    7: (127, 255, 0),
    8: (255, 255, 0),
    9: (255, 165, 0),
    10: (165, 42, 42),
    11: (0, 255, 255),
    12: (128, 0, 128)
}
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder# + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                     reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'Houston13':
        # Load the image
        # img = open_file(folder + 'Houston13.mat')['ori_data']

        # rgb_bands = [13,20,33]

        # gt = open_file(folder + 'Houston13.mat'Houston13_7gt.mat')['map']

        # label_values = ["grass healthy", "grass stressed", "trees",
        #                 "water", "residential buildings",
        #                 "non-residential buildings", "road"]

        # ignored_labels = [0]
        Houston13_data = h5py.File(folder + 'Houston13.mat', 'r')
        img = np.transpose(Houston13_data['ori_data'])

        rgb_bands = [13, 20, 33]

        Houston13_7gt_data = h5py.File(folder + 'Houston13_7gt.mat', 'r')  # 加载原版标签
        gt = np.transpose(Houston13_7gt_data['map'])  # 加载原版标签

        gt = np.int64(gt)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]
        
    elif dataset_name == 'Houston18':
        # Load the image
        # img = open_file(folder + 'Houston18.mat')['ori_data']

        # rgb_bands = [13,20,33]

        # gt = open_file(folder + 'Houston18_7gt.mat')['map']

        # label_values = ["grass healthy", "grass stressed", "trees",
        #                 "water", "residential buildings",
        #                 "non-residential buildings", "road"]
        
        # ignored_labels = [0]
        Houston18_data = h5py.File(folder + 'Houston18.mat', 'r')
        img = np.transpose(Houston18_data['ori_data'])

        rgb_bands = [13, 20, 33]

        Houston18_7gt_data = h5py.File(folder + 'Houston18_7gt.mat', 'r')
        gt = np.transpose(Houston18_7gt_data['map'])
        gt = np.int64(gt)

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]
    elif dataset_name == 'paviaU':

        # -------------------------------------------------
        # Load image
        # Original UP has 103 bands
        # PMGDG paper removes last band
        # Final = 102 bands
        # -------------------------------------------------

        img = open_file(folder + 'paviaU.mat')['paviaU']

        # Remove last spectral band
        img = img[:, :, :-1]

        rgb_bands = [20, 30, 40]

        # -------------------------------------------------
        # Load GT
        # -------------------------------------------------

        gt = open_file(folder + 'paviaU_7gt.mat')['paviaU_gt']

        # -------------------------------------------------
        # Remove NON-COMMON classes
        # Remove:
        # 3 = Gravel
        # 5 = Metal sheets
        # -------------------------------------------------

        gt[(gt == 3) | (gt == 5)] = 0

        # -------------------------------------------------
        # Paper class order
        # -------------------------------------------------

        label_values = [
            "Tree",
            "Asphalt",
            "Brick",
            "Bitumen",
            "Shadow",
            "Meadow",
            "Bare soil"
        ]

        ignored_labels = [0]

        # -------------------------------------------------
        # Correct mapping from original GT labels
        #
        # Original → Paper class
        # 4 → Tree
        # 1 → Asphalt
        # 8 → Brick
        # 7 → Bitumen
        # 9 → Shadow
        # 2 → Meadow
        # 6 → Bare soil
        # -------------------------------------------------

        valid_classes = [4, 1, 8, 7, 9, 2, 6]

        new_gt = np.zeros_like(gt, dtype=np.int64)

        for new_label, old_label in enumerate(valid_classes, start=1):
            new_gt[gt == old_label] = new_label

        gt = new_gt
            
    elif dataset_name == 'paviaC':

        img = open_file(folder + 'paviaC.mat')['pavia']

        if img.shape[-1] > 102:
            img = img[:, :, :102]

        rgb_bands = [20, 30, 40]

        gt = open_file(folder + 'paviaC_7gt.mat')['pavia_gt']

        # Remove non-common classes
        gt[(gt == 1) | (gt == 8)] = 0

        label_values = [
            "Tree",
            "Asphalt",
            "Brick",
            "Bitumen",
            "Shadow",
            "Meadow",
            "Bare soil"
        ]

        ignored_labels = [0]

        # Correct mapping
        valid_classes = [2, 6, 4, 7, 9, 3, 5]

        new_gt = np.zeros_like(gt, dtype=np.int64)

        for new_label, old_label in enumerate(valid_classes, start=1):
            new_gt[gt == old_label] = new_label

        gt = new_gt
    elif dataset_name == 'Dioni':

        img = scipy.io.loadmat(folder + 'Dioni.mat')['ori_data']

        # False-color bands similar to paper visualization
        rgb_bands = [90, 40, 10]

        gt = scipy.io.loadmat(folder + 'Dioni_gt.mat')['map']

        # -------------------------------------------------
        # Re-map Dioni labels to continuous 12-class setup
        # -------------------------------------------------

        valid_classes = [1,2,3,4,5,7,9,10,11,12,13,14]

        new_gt = np.zeros_like(gt, dtype=np.int64)

        for new_label, old_label in enumerate(valid_classes, start=1):
            new_gt[gt == old_label] = new_label

        gt = new_gt

        # -------------------------------------------------
        # Common class names (same as paper)
        # -------------------------------------------------

        label_values = [
            "Dense Urban Fabric",
            "Mineral Extraction Sites",
            "Non Irrigated Arable Land",
            "Fruit Trees",
            "Olive Groves",
            "Coniferous Forest",
            "Dense Sclerophyllous Vegetation",
            "Sparse Sclerophyllous Vegetation",
            "Sparsely Vegetated Areas",
            "Rocks and Sand",
            "Water",
            "Coastal Water"
        ]

        ignored_labels = [0]
    elif dataset_name == 'Loukia':

        img = scipy.io.loadmat(folder + 'Loukia.mat')['ori_data']

        rgb_bands = [90, 40, 10]

        gt = scipy.io.loadmat(folder + 'Loukia_gt.mat')['map']

        # -------------------------------------------------
        # Remove non-common classes
        # -------------------------------------------------

        gt[(gt == 6) | (gt == 7)] = 0

        # -------------------------------------------------
        # Re-map Loukia labels to common 12-class setup
        # -------------------------------------------------

        valid_classes = [1,2,3,4,5,8,9,10,11,12,13,14]

        new_gt = np.zeros_like(gt, dtype=np.int64)

        for new_label, old_label in enumerate(valid_classes, start=1):
            new_gt[gt == old_label] = new_label

        gt = new_gt

        label_values = [
            "Dense Urban Fabric",
            "Mineral Extraction Sites",
            "Non Irrigated Arable Land",
            "Fruit Trees",
            "Olive Groves",
            "Coniferous Forest",
            "Dense Sclerophyllous Vegetation",
            "Sparse Sclerophyllous Vegetation",
            "Sparsely Vegetated Areas",
            "Rocks and Sand",
            "Water",
            "Coastal Water"
        ]

        ignored_labels = [0]
        
    elif dataset_name == 'shanghai':
        cube = open_file(folder+'DataCube.mat')
        img = cube['DataCube1']
        gt = cube['gt1']
        rgb_bands = [20,30,40]
        label_values = ['water','land/building', 'plant']
        ignored_labels = [0]
    elif dataset_name == 'hangzhou':
        cube = open_file(folder+'DataCube.mat')
        img = cube['DataCube2']
        gt = cube['gt2']
        rgb_bands = [20,30,40]
        label_values = ['water','land/building', 'plant']
        ignored_labels = [0]
    elif dataset_name == 'whu_71':
        img = np.array(tiff.imread(folder + 'O1_0071.tif'))
        rgb_bands = [20,30,40]
        gt = np.array(tiff.imread(folder + 'O1_0071_label.tif'))
        gt = np.where((gt == 6) | (gt == 12), 0, gt)
        mapping = {2: 1, 10: 2, 15: 3, 16: 4, 17: 5}
        for old_value, new_value in mapping.items():
            gt[gt == old_value] = new_value
        label_values = ['Dry farm','River canal', '	Urban built-up', 'Rural settlement', 'Other construction land']
        ignored_labels = [0]
    elif dataset_name == 'whu_78':
        img = np.array(tiff.imread(folder + 'O1_0078.tif'))
        rgb_bands = [20,30,40]
        gt = np.array(tiff.imread(folder + 'O1_0078_label.tif'))
        mapping = {2: 1, 10: 2, 15: 3, 16: 4, 17: 5}
        for old_value, new_value in mapping.items():
            gt[gt == old_value] = new_value
        label_values = ['Dry farm','River canal', '	Urban built-up', 'Rural settlement', 'Other construction land']
        ignored_labels = [0]

    # else:
    #     # Custom dataset
    #     img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    
    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img= img.reshape((m*n,-1))
    img = img/img.max()
    img_temp = np.sqrt(np.asarray((img**2).sum(1)))
    img_temp = np.expand_dims(img_temp,axis=1)
    img_temp = img_temp.repeat(d,axis=1)
    img_temp[img_temp==0]=1
    img = img/img_temp
    img = np.reshape(img,(m,n,-1))
    

    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        
        state = np.random.get_state()
        np.random.shuffle(self.indices)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
                data = self.mixture_noise(data, label)


        # if self.flip_augmentation and self.patch_size > 1:
        #     # Perform data augmentation (only on 2D patches)
        #     data, label = self.flip(data, label)
        # if self.radiation_augmentation and np.random.random() < 0.1:
        #         data = self.radiation_noise(data)
        # if self.mixture_augmentation and np.random.random() < 0.5:
        #         data = self.mixture_noise(data, label)



        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]
            
        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        # plt.imshow(data[[10,23,23],:,:].permute(1,2,0))
        # plt.show()
        return data, label


class HyperX_test(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX_test, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.r = int(self.patch_size / 2) + 1

        self.center_pixel = hyperparams['center_pixel']

        # Fully supervised : use all pixels with label not ignored

        mask = np.ones_like(gt)

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        # r = self.r

        # self.data = np.pad(self.data, ((r, r), (r, r), (0, 0)), 'symmetric')

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]
        return data, label



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data, self.label = next(self.loader)

        except StopIteration:
            self.next_input = None

            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        label = self.label

        self.preload()
        return data, label