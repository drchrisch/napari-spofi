"""
some helper functions for napari spot it training
"""

import numpy as np
from pathlib import Path
from tifffile import imread
from tqdm.notebook import tqdm
from pprint import pprint
# from tensorflow.config import get_visible_devices
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
from functools import lru_cache


class FileData(Sequence):
    """
    handle image files for training
    """

    def __init__(self, filenames):
        super().__init__()
        self._filenames = filenames

    def __len__(self):
        return len(self._filenames)

    @lru_cache
    def __getitem__(self, i):
        x = imread(self._filenames[i])
        return x


class TrainingUtils:
    """
    some functions for StarDist processing
    """
    def __init__(self, parent):
        self.data = parent.data
        self.training_widget = parent.widget
        self.model_dir = Path(self.data.model_dir)

        # check for sd_ directories, prepare data directories
        self.sd_images = Path(self.data.sd_images)
        self.sd_masks = Path(self.data.sd_masks)
        self.validation_fraction = self.data.validation_fraction
        self.train_patch_size = self.data.train_patch_size
        self.number_of_epochs = self.data.number_of_epochs
        self.stop_training = False

        # check gpu
        from tensorflow.config import get_visible_devices
        print(get_visible_devices())

        # load image and label data
        self.files = [x.stem for x in list(self.sd_images.glob(('img*' + '.tif')))]
        assert len(self.files) > 1, "not enough training data"
        print(f"image files: {self.files}")
        self.imgs = FileData([self.sd_images / (f + '.tif') for f in self.files])
        self.masks = FileData([self.sd_masks / (f + '.tif') for f in self.files])
        # report image and label shape
        print(f'image shape: {self.imgs[0].shape}, label shape: {self.masks[0].shape}')
        # split data
        self.split_data()

        # update steps_per_epoch value
        self.steps_per_epoch = np.max([5, int(len(self.training_files) / 2)])

        # calculate image anisotropy (from labels), estimate number of rays
        """
        we skip this for 'typical' images
        self.get_spot_anisotropy()
        print('empirical anisotropy of labeled objects = %s' % str(self.anisotropy))
        self.get_number_of_rays()
        """
        self.anisotropy = (1, 1, 1)
        self.scores_iso = None
        self.scores_aniso = None
        # set n_rays parameter
        # n_rays = 96 # is a good default choice (but see calculation above)
        self.n_rays = self.data.n_rays  # default 96

        # gather configs
        self.model = self.get_model()

    def split_data(self):
        # get random file indices
        self.random_index = np.random.permutation(len(self.files))
        self.validation_n = max(1, int(round(self.validation_fraction * len(self.random_index))))
        print(f"{self.random_index=}")

        # keep names of training and validation files
        self.training_files = [self.files[i] for i in self.random_index[:-self.validation_n]]
        self.validation_files = [self.files[i] for i in self.random_index[-self.validation_n:]]

        print('number of images: %3d' % len(self.files))
        print('- training:       %3d' % len(self.training_files))
        print('- validation:     %3d' % len(self.validation_files))

        # make lists from image and label input
        self.imgs_trn = [self.imgs[idx] for idx in self.random_index[:-self.validation_n]]
        self.masks_trn = [self.masks[idx] for idx in self.random_index[:-self.validation_n]]
        self.imgs_val = [self.imgs[idx] for idx in self.random_index[-self.validation_n:]]
        self.masks_val = [self.masks[idx] for idx in self.random_index[-self.validation_n:]]

    @staticmethod
    def random_intensity_change(img):
        img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
        return img

    @staticmethod
    def random_flipper(img, mask):
        # augmentation for two channel (do not touch channel order) input
        # find out what to do
        yes_or_no = np.random.randint(0, 6, 1)
        # flip updown
        if yes_or_no == 0:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)
        # flip leftright
        elif yes_or_no == 1:
            img = np.flip(img, axis=(2, 1))
            mask = np.flip(mask, axis=(2, 1))
        # transpose
        elif yes_or_no == 2:
            img = np.transpose(img, axes=(0, 2, 1, 3))
            mask = np.transpose(mask, axes=(0, 2, 1))
        # rotate
        elif yes_or_no == 3:
            img = np.rot90(img, axes=(1, 2))
            mask = np.rot90(mask, axes=(1, 2))
        # rotate
        elif yes_or_no == 4:
            img = np.rot90(img, k=3, axes=(1, 2))
            mask = np.rot90(mask, k=3, axes=(1, 2))
        # do nothing
        else:
            pass
        return img, mask

    def augmenter(self, x, y):
        """Augmentation of a single input/label image pair.
        x is an input image
        y is the corresponding ground-truth label image
        """
        x, y = self.random_flipper(x, y)
        x = self.random_intensity_change(x)
        # add some gaussian noise
        sig = 0.02 * np.random.uniform(0, 1)
        x = x + sig * np.random.normal(0, 1, x.shape)
        return x, y

    def get_model(self):
        from stardist import gputools_available
        from stardist import Rays_GoldenSpiral
        from stardist.models import Config3D, StarDist3D
        from stardist import calculate_extents

        # Use OpenCL-based computations for data generator during training (requires 'gputools')
        use_gpu = True and gputools_available()
        if use_gpu:
            from csbdeep.utils.tf import limit_gpu_memory
            limit_gpu_memory(None, allow_growth=True)

        # check if new model should be started with new configs or existing configs should be used
        if (self.model_dir / "config.json").exists():
            # create new model using existing conf
            model = StarDist3D(None, name=self.model_dir.name, basedir=self.model_dir.parent)
        else:
            # create new model using new configs
            # check number of channels
            n_channel = 1 if self.imgs[0].ndim == 3 else self.imgs[0].shape[-1]
            print(f'number of channels in input image: {n_channel}')
            # Predict on subsampled grid for increased efficiency and larger field of view
            grid = tuple(1 if a > 1.5 else 2 for a in self.anisotropy)
            # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
            rays = Rays_GoldenSpiral(self.n_rays, anisotropy=self.anisotropy)
            # adjust for your data below (make patch size as large as possible)
            conf = Config3D(
                rays=rays,
                grid=grid,
                anisotropy=self.anisotropy,
                use_gpu=use_gpu,
                n_channel_in=n_channel,
                n_classes=None,
                train_patch_size=self.train_patch_size,
                train_batch_size=1,
                unet_n_depth=3,
                train_sample_cache=False,
            )
            print(conf)

            model = StarDist3D(conf, name=self.model_dir.name, basedir=self.model_dir.parent)

        # add custom callback to model
        model.prepare_for_training()
        training_stop_callback = self.StopCallback(self)
        report_callback = self.ReportCallback(self)
        model.callbacks.append(report_callback)
        model.callbacks.append(training_stop_callback)
        print(model.callbacks)

        # check object size and field of view
        median_size = calculate_extents(self.masks, np.median)
        fov = np.array(model._axes_tile_overlap('ZYX'))
        print(f"median object size:      {median_size}")
        print(f"network field of view :  {fov}")
        if any(median_size > fov):
            print("WARNING: median object size larger than field of view of the neural network.")

        return model

    def get_number_of_rays(self):
        n_rays = [8, 16, 32, 64, 96, 128]
        self.scores_iso = self.reconstruction_scores(n_rays, anisotropy=None)
        self.scores_aniso = self.reconstruction_scores(n_rays, anisotropy=self.anisotropy)

        """
        plt.figure(figsize=(8,5))
        plt.plot(n_rays, self.scores_iso,   'o-', label='Isotropic')
        plt.plot(n_rays, self.scores_aniso, 'o-', label='Anisotropic')
        plt.xlabel('Number of rays for star-convex polyhedra')
        plt.ylabel('Reconstruction score (mean intersection over union)')
        plt.legend();
        """
        pprint(dict({'n_rays': n_rays, 'scores iso': self.scores_iso, 'scores aniso': self.scores_aniso}),
               sort_dicts=False)

    def get_spot_anisotropy(self):
        # calculate anisotropy (from labels)
        from stardist import calculate_extents
        extents = calculate_extents(self.masks)
        self.anisotropy = tuple(np.max(extents) / extents)
        return self.anisotropy

    def reconstruction_scores(self, n_rays, anisotropy=None):
        # estimate anisotropy effect
        from stardist import Rays_GoldenSpiral
        from stardist import relabel_image_stardist3D
        from stardist.matching import matching_dataset
        scores = []
        for r in tqdm(n_rays):
            rays = Rays_GoldenSpiral(r, anisotropy=anisotropy)
            masks_reconstructed = [relabel_image_stardist3D(lbl, rays) for lbl in self.masks]
            mean_iou = matching_dataset(self.masks, masks_reconstructed, thresh=0, show_progress=False).mean_true_score
            scores.append(mean_iou)
        return scores

    def train(self):
        # print(f"Training start, {self.data.training_widget.losses.value}")
        # start StarDist model training
        self.model.train(self.imgs_trn, self.masks_trn,
                         validation_data=(self.imgs_val, self.masks_val),
                         augmenter=self.augmenter,
                         epochs=self.number_of_epochs,  # 200 epochs seem to be enough for synthetic demo dataset
                         steps_per_epoch=self.steps_per_epoch,
                         )

    def optimize_thresholds(self):
        self.model.optimize_thresholds(self.imgs_val, self.masks_val, nms_threshs=[0.01, ], )

    class StopCallback(Callback):
        def __init__(self, args):
            super().__init__()
            self.args = args

        def on_epoch_end(self, epoch, logs={}):
            if self.args.stop_training:
                print(f"training stopped after epoch {epoch + 1}")
                self.model.stop_training = True

    class ReportCallback(Callback):
        def __init__(self, args):
            super().__init__()
            self.args = args

        def on_epoch_end(self, epoch, logs={}):
            loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            print(f"\t***** training: {epoch}, {loss}, validation: {val_loss} *****")
            self.args.training_widget.losses.value = f"{epoch=:4n} * {loss=:.2f} * {val_loss=:.2f}"
            self.args.data.losses['training'].append(loss)
            self.args.data.losses['validation'].append(val_loss)
