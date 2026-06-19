# -*- coding: utf-8 -*-
import math
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import itertools
import spectral
import matplotlib.pyplot as plt
from scipy import io
import imageio
import logging
import os
import re
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                # data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device

def adjust_learning_rate(optimizer, epoch, lr, total_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr*(1 - math.sqrt(epoch/total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))


def human_readable_bytes(num_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def safe_count_nonzero(array):
    if isinstance(array, (list, tuple)):
        return len(array)
    return int(np.count_nonzero(array))


def choose_rgb_bands(rgb_bands, n_bands):
    if rgb_bands is not None and len(rgb_bands) >= 3:
        return [min(max(int(b), 0), n_bands - 1) for b in rgb_bands[:3]]
    return [0, min(1, n_bands - 1), min(2, n_bands - 1)]


def get_visualization_directory(visualizations_dir=None):
    if visualizations_dir is None:
        visualizations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    return visualizations_dir


def sanitize_dataset_name(dataset_name):
    return ''.join(c if c.isalnum() or c in '_-' else '_' for c in str(dataset_name))


def save_figure(fig, filename, visualizations_dir=None):
    visualizations_dir = get_visualization_directory(visualizations_dir)
    filepath = os.path.join(visualizations_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    return filepath


def visualize_pca_scatter(img, gt, dataset_name, sample_size=5000, ignored_labels=None, visualizations_dir=None):
    flattened = img.reshape(-1, img.shape[-1])
    labels = gt.reshape(-1)
    mask = np.isfinite(flattened).all(axis=1)
    flattened = flattened[mask]
    labels = labels[mask]
    if ignored_labels is not None:
        excluded = set(ignored_labels)
        label_mask = np.array([int(l) not in excluded for l in labels])
        flattened = flattened[label_mask]
        labels = labels[label_mask]
    if flattened.shape[0] == 0:
        return
    if flattened.shape[0] > sample_size:
        indices = np.random.choice(flattened.shape[0], sample_size, replace=False)
        flattened = flattened[indices]
        labels = labels[indices]

    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(np.nan_to_num(flattened))
    unique_labels = np.unique(labels)
    fig = plt.figure(figsize=(8, 6))
    for label in unique_labels:
        label_mask = labels == label
        plt.scatter(pca_points[label_mask, 0], pca_points[label_mask, 1],
                    s=10, alpha=0.6, label=str(int(label)))
    plt.title(f"{dataset_name} PCA scatter plot")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    filename = f"{sanitize_dataset_name(dataset_name)}_pca_scatter.png"
    save_figure(fig, filename, visualizations_dir=visualizations_dir)
    plt.show()
    plt.close(fig)


def visualize_class_distribution(gt, dataset_name, ignored_labels=None, visualizations_dir=None):
    values, counts = np.unique(gt, return_counts=True)
    if ignored_labels is None:
        ignored_labels = []
    filtered = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) not in ignored_labels]
    if len(filtered) == 0:
        return
    class_values, class_counts = zip(*filtered)
    fig = plt.figure(figsize=(8, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_values)))
    plt.bar([str(v) for v in class_values], class_counts, color=colors)
    plt.title(f"{dataset_name} class distribution")
    plt.xlabel("Class label")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    filename = f"{sanitize_dataset_name(dataset_name)}_class_distribution.png"
    save_figure(fig, filename, visualizations_dir=visualizations_dir)
    plt.show()
    plt.close(fig)


def visualize_pca_variance(img, dataset_name, n_components=10, visualizations_dir=None):
    flattened = img.reshape(-1, img.shape[-1])
    flattened = np.nan_to_num(flattened)
    n_components = min(n_components, img.shape[-1])
    pca = PCA(n_components=n_components)
    pca.fit(flattened)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    fig = plt.figure(figsize=(8, 5))
    plt.bar(np.arange(1, len(ratios) + 1), ratios, alpha=0.6, label='Explained variance')
    plt.step(np.arange(1, len(cumulative) + 1), cumulative, where='mid', label='Cumulative variance')
    plt.xlabel('PCA component')
    plt.ylabel('Variance ratio')
    plt.title(f"{dataset_name} PCA explained variance")
    plt.xticks(np.arange(1, len(ratios) + 1))
    plt.legend()
    plt.tight_layout()
    filename = f"{sanitize_dataset_name(dataset_name)}_pca_variance.png"
    save_figure(fig, filename, visualizations_dir=visualizations_dir)
    plt.show()
    plt.close(fig)


def visualize_preprocessed_data(img, gt, rgb_bands, dataset_name, pca_components=3, ignored_labels=None, visualizations_dir=None):
    height, width, n_bands = img.shape
    sel_bands = choose_rgb_bands(rgb_bands, n_bands)
    rgb = img[..., sel_bands].astype('float32')
    rgb = rgb - np.nanmin(rgb)
    if np.nanmax(rgb) != 0:
        rgb = rgb / np.nanmax(rgb)
    rgb = np.clip(rgb, 0, 1)
    rgb_img = np.asarray(255 * rgb, dtype='uint8')

    fig = plt.figure(figsize=(6, 6))
    plt.title(f"{dataset_name} RGB visualization")
    plt.imshow(rgb_img)
    plt.axis('off')
    filename = f"{sanitize_dataset_name(dataset_name)}_rgb.png"
    save_figure(fig, filename, visualizations_dir=visualizations_dir)
    plt.show()
    plt.close(fig)

    
    
    # =========================================================
# Dataset-specific ground truth visualization
# =========================================================

    # =========================================================
# Dataset-specific ground truth visualization
# =========================================================

    if dataset_name in ['Dioni', 'Loukia']:

    # HyRANK palette

        colors = [
            '#000000',
            '#800080',
            '#0000FF',
            '#ADD8E6',
            '#90EE90',
            '#006400',
            '#32CD32',
            '#7FFF00',
            '#FFFF00',
            '#FFA500',
            '#FF0000',
            '#FF1493',
            '#8B0000'
        ]

        class_names = [
            'BG',
            'Dense Urban Fabric',
            'Mineral Extraction Sites',
            'Non Irrigated Arable Land',
            'Fruit Trees',
            'Olive Groves',
            'Coniferous Forest',
            'Dense Sclerophyllous Vegetation',
            'Sparse Sclerophyllous Vegetation',
            'Sparsely Vegetated Areas',
            'Rocks and Sand',
            'Water',
            'Coastal Water'
        ]

        vmax_value = 12

        fig_size = (14, 4)

    elif dataset_name in ['Houston13', 'Houston18']:

    # Houston palette

        colors = [
            '#000000',
            '#0000FF',
            '#66FF66',
            '#008000',
            '#FFFF00',
            '#FF0000',
            '#FFA500',
            '#FF1493'
        ]

        class_names = [
            'BG',
            'Grass healthy',
            'Grass stressed',
            'Trees',
            'Water',
            'Residential buildings',
            'Non-residential buildings',
            'Road'
        ]

        vmax_value = 7

        fig_size = (10, 6)

    else:

    # Pavia palette

        colors = [
            'black',
            'blue',
            'limegreen',
            'yellow',
            'orange',
            'purple',
            'cyan',
            'red'
        ]

        class_names = [
            'BG',
            'Tree',
            'Asphalt',
            'Brick',
            'Bitumen',
            'Shadow',
            'Meadow',
            'Bare soil'
        ]

        vmax_value = 7

        fig_size = (8, 8)

    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=fig_size)

    plt.title(f"{dataset_name} Ground truth map")

    plt.imshow(
        gt,
        cmap=cmap,
        interpolation='nearest',
        vmin=0,
        vmax=vmax_value
    )

    plt.axis('off')

    cbar = plt.colorbar(
        ticks=np.arange(0, vmax_value + 1)
    )

    cbar.set_ticklabels(class_names)

    filename = f"{sanitize_dataset_name(dataset_name)}_ground_truth_paper_style.png"

    save_figure(
        fig,
        filename,
        visualizations_dir=visualizations_dir
    )

    plt.show()

    plt.close(fig)

    pca_components = min(pca_components, n_bands)
    flattened = img.reshape(-1, n_bands)
    flattened = np.nan_to_num(flattened)
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(flattened)
    if pca_components < 3:
        pca_result = np.pad(pca_result, ((0, 0), (0, 3 - pca_components)), mode='constant')
    pca_img = pca_result.reshape(height, width, 3)
    pca_img = pca_img - np.min(pca_img)
    if np.max(pca_img) != 0:
        pca_img = pca_img / np.max(pca_img)
    pca_img = np.asarray(255 * np.clip(pca_img, 0, 1), dtype='uint8')

    fig = plt.figure(figsize=(6, 6))
    plt.title(f"{dataset_name} PCA visualization ({pca_components} components)")
    plt.imshow(pca_img)
    plt.axis('off')
    filename = f"{sanitize_dataset_name(dataset_name)}_pca_visualization.png"
    save_figure(fig, filename, visualizations_dir=visualizations_dir)
    plt.show()
    plt.close(fig)

    visualize_pca_variance(img, dataset_name, visualizations_dir=visualizations_dir)
    visualize_pca_scatter(img, gt, dataset_name, ignored_labels=ignored_labels, visualizations_dir=visualizations_dir)
    visualize_class_distribution(gt, dataset_name, ignored_labels=ignored_labels, visualizations_dir=visualizations_dir)


def preprocess_dataset(dataset_name, img, gt, patch_size, training_sample_ratio, ignored_labels, rgb_bands=None, palette=None, visualize=False, pca_components=3):
    n_bands = img.shape[-1]
    # Number of classes after filtering/remapping (ignored labels excluded)
    unique_labels = np.unique(gt)
    n_classes = len([l for l in unique_labels if int(l) not in ignored_labels and int(l) != 0])
    dataset_shape = img.shape
    data_dtype = img.dtype
    data_min = float(np.nanmin(img))
    data_max = float(np.nanmax(img))
    data_mean = float(np.nanmean(img))
    data_std = float(np.nanstd(img))
    zero_values = int(np.sum(img == 0))
    nan_values = int(np.isnan(img).sum())
    inf_values = int(np.isinf(img).sum())
    pad = int(patch_size / 2) + 1
    padded_shape = (dataset_shape[0] + 2 * pad, dataset_shape[1] + 2 * pad, n_bands)
    train_gt, test_gt, _, _ = sample_gt(gt, training_sample_ratio, mode='random')
    training_count = safe_count_nonzero(train_gt)
    testing_count = safe_count_nonzero(test_gt)
    class_values, class_counts = np.unique(gt, return_counts=True)
    class_distribution = {int(label): int(count) for label, count in zip(class_values, class_counts) if int(label) not in ignored_labels}
    total_bytes = img.nbytes + gt.nbytes
    patch_shape = f"{patch_size} × {patch_size} × {n_bands}"
    patch_shape_pca = None
    if pca_components and pca_components < n_bands:
        patch_shape_pca = f"{patch_size} × {patch_size} × {min(pca_components, n_bands)}"

    print('\n' + '=' * 72)
    print(f"Preprocessing summary for dataset: {dataset_name}")
    print('=' * 72)
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset shape: {dataset_shape[0]} × {dataset_shape[1]} × {dataset_shape[2]}")
    print(f"Number of spectral bands: {n_bands}")
    # Print number of classes (should be 7 for Pavia UP/PC after remapping)
    print(f"Number of classes: {n_classes}")
    print(f"Data type: {data_dtype}")
    print(f"Normalization min/max: {data_min:.6f} / {data_max:.6f}")
    print(f"Mean / Std: {data_mean:.6f} / {data_std:.6f}")
    print(f"Zero values: {zero_values}")
    print(f"NaN values: {nan_values}")
    print(f"Infinite values: {inf_values}")
    print(f"Padding output shape: {padded_shape[0]} × {padded_shape[1]} × {padded_shape[2]}")
    print(f"Training samples: {training_count}")
    print(f"Testing samples: {testing_count}")
    print(f"Patch/tensor shape after patch extraction: {patch_shape}")
    if patch_shape_pca is not None:
        print(f"Patch/tensor shape after PCA: {patch_shape_pca}")
    print("Class distribution per class:")
    for label, count in sorted(class_distribution.items()):
        print(f"    {label}: {count}")
    print(f"Memory usage estimate (arrays only): {human_readable_bytes(total_bytes)}")
    if patch_shape_pca is not None:
        print(f"PCA component count: {min(pca_components, n_bands)}")
    print('=' * 72 + '\n')

    if visualize:
        visualize_preprocessed_data(img, gt, rgb_bands, dataset_name,
                                    pca_components=pca_components,
                                    ignored_labels=ignored_labels)

    return {
        'dataset_name': dataset_name,
        'dataset_shape': dataset_shape,
        'n_bands': n_bands,
        'n_classes': n_classes,
        'dtype': data_dtype,
        'min': data_min,
        'max': data_max,
        'mean': data_mean,
        'std': data_std,
        'zero_values': zero_values,
        'nan_values': nan_values,
        'inf_values': inf_values,
        'padded_shape': padded_shape,
        'training_count': training_count,
        'testing_count': testing_count,
        'patch_shape': patch_shape,
        'patch_shape_pca': patch_shape_pca,
        'class_distribution': class_distribution,
        'memory_estimate': total_bytes,
        'pca_components': min(pca_components, n_bands) if patch_shape_pca is not None else None
    }


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})

def display_dataset(img, gt, bands, labels, palette, vis):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})

def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    # ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    # for l in ignored_labels:
    #     ignored_mask[target == l] = True
    # ignored_mask = ~ignored_mask
    # target = target[ignored_mask] -1
    # target = target[ignored_mask]
    # prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion_matrix"] = cm

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    results["TPR"] = TPR
    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1_scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    results["prediction"] = prediction
    results["label"] = target

    return results


def show_results(results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1_scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion_matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion_matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1_scores"]
        kappa = results["Kappa"]

    #label_values = label_values[1:]
    vis.heatmap(cm, opts={'title': "Confusion_matrix", 
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    text += "Confusion_matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1_scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print(text)


def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random':
        if train_size == 1:
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0],test_indices[1],test_label))

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt, train_set, test_set


def sample_gt_fixed(gt, train_size_list, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)

    train_label = []
    test_label = []
    print("Sampling {} with train size = {}".format(mode, train_size_list))
    train_indices, test_indices = [], []
    train_label = []
    test_label = []
    for c in np.unique(gt):
        if c == 0:
            continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices))  # x,y features

        train, test = sklearn.model_selection.train_test_split(
            X, train_size=train_size_list[c-1], random_state=23)
        train_indices += train
        test_indices += test
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]

    [train_label.append(i) for i in gt[train_indices]]
    train_set = np.column_stack(
        (train_indices[0], train_indices[1], train_label))
    [test_label.append(i) for i in gt[test_indices]]
    test_set = np.column_stack((test_indices[0], test_indices[1], test_label))

    return train_gt, test_gt, train_set, test_set

def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients 
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def set_up_logger(logs_path, log_file_name):
        # logging settings
        logger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(logs_path, log_file_name), mode="w")
        consoleHandler = logging.StreamHandler()
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.info("Created " + log_file_name)
        return logger

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def mixup_data(x1, x2, y, alpha=1.0, use_cuda=False, lam=None):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    if not lam:

        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index,:]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index,:]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_data1(x, y, alpha=1.0, use_cuda=False, lam=None):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    if not lam:

        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x= lam * x + (1 - lam) * x[index,:]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

