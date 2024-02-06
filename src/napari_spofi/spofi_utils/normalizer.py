# some utils for using StarDist
import numpy as np
import h5py

# cs12oct2023


def get_normalization_limits(path=None, file=None, channels=None, norm_ranges={'min': 1.0, 'max': 99.8},
                             norm_type='full'):
    """
    get normalization levels (percentile based) for each channel
    input images are expected to have gone only through median filtering
    use either h5 type file or image dictionary
    """

    # check suffix
    file = str(file)
    if file.endswith('.h5'):
        pass
    else:
        file = file + '.h5'

    levels = dict()
    for range_key in norm_ranges.keys():
        with h5py.File(path / file, 'r') as f:
            _levels = []
            for channel in channels:
                img = f[channel][:].astype(np.uint16)
                if norm_type.endswith("nozero"):
                    img = img[img > 0]
                if norm_type.startswith("full"):
                    _levels.append(np.percentile(img, norm_ranges[range_key]).astype(np.uint16))
                levels[range_key] = _levels
    return levels


def normalize_to_range(img, min_max):
    """
    define normalization function
    """

    img = img.astype(np.float32)
    eps = np.finfo('float64').eps
    for idx in range(img.shape[-1]):
        img[..., idx] = (img[..., idx] - min_max['min'][idx]) / (min_max['max'][idx] - min_max['min'][idx] + eps)

    return img
