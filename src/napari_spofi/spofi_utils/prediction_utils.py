"""
some helper functions for napari-SpotIt prediction

cs 25jan2024
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import ndimage as ndi
from skimage.exposure import match_histograms
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from .normalizer import get_normalization_limits
from .normalizer import normalize_to_range
#from tensorflow.config import get_visible_devices


class PredictionUtils:
    """
    some functions for StarDist processing
    """
    def __init__(self, parent):
        self.data = parent.data
        self.model_dir = Path(parent.model_dir)
        self.model_name = None
        self.model = None
        self.img_dir = Path(self.data.img_dir)
        self.img_file = self.data.img_file
        self.ref_file = None
        self.threshold = parent.threshold
        self.predictions_dir = Path(self.data.sd_predictions)
        self.spot_class = "A"

        # check gpu
        from tensorflow.config import get_visible_devices
        print(get_visible_devices())

    def predict(self):
        from stardist.models import Config3D, StarDist3D
        start_time = time.process_time()

        # load model
        self.model_name = self.model_dir.name
        self.model = StarDist3D(None, name=self.model_name, basedir=self.model_dir.parent)

        # check prediction dir for current img dir
        prediction_dir_current = self.predictions_dir / self.img_dir.stem
        if not prediction_dir_current.exists():
            Path.mkdir(prediction_dir_current)
        predictions = [x for x in self.predictions_dir.glob("*") if x.is_dir]
        for prediction in predictions:
            if prediction.name == self.img_dir.name:
                prediction_files = [f.name for f in prediction.iterdir() if (f.suffix == ".h5")
                                    & (self.model_name in f.name)]
        prediction_file = [f for f in prediction_files if Path(self.img_file).name in f]
        if len(prediction_file):
            print(f"prediction found {prediction_file}")
            self.parent.prediction_file = prediction_file
            return

        # check normalization data
        df = self.data.annotation_data["true"].copy()
        normalization_values = df.loc[(df["img_dir"] == self.data.img_dir)
                                      & (df["img_file"] == self.data.img_file), "normalization_values"]
        if not len(normalization_values):
            print(f"predict, calculating normalization values: {self.img_dir=}, {self.img_file=}")
            channels = [self.data.fg_channel, self.data.bg_channel]
            normalization_values = get_normalization_limits(
                path=Path(self.img_dir),
                file=self.img_file,
                channels=channels,
                norm_ranges=self.data.normalization_range,
                norm_type=self.data.normalization_type,
            )
        else:
            normalization_values = normalization_values.to_numpy()[0]
        self.normalization_values = normalization_values

        # load reference image, mind active slice
        if not isinstance(self.ref_file, type(None)):
            with h5py.File(self.img_dir / self.ref_file, 'r') as f:
                ref_img = np.stack([f[channel][eval(str(self.data.img_slice))] for channel in [self.data.fg_channel, self.data.bg_channel]], axis=-1)

        # load image, mind active slice
        with h5py.File(self.img_dir / self.img_file, 'r') as f:
            img = np.stack([f[channel][eval(str(self.data.img_slice))] for channel in [self.data.fg_channel, self.data.bg_channel]], axis=-1)

        # match image intensities to reference image
        if not isinstance(self.ref_file, type(None)):
            print('match histograms')
            img = self.histogram_matcher(
                ref_f=(self.img_dir / self.ref_file),
                img_f=(self.img_dir / self.img_file),
                main_channel=self.data.fg_channel,
                bgnd_channel=self.data.bg_channel,
                channel_type='',
            )

        # normalize intensity
        img = normalize_to_range(img, normalization_values)

        labels, res = self.model.predict_instances(img,
                                                   n_tiles=self.model._guess_n_tiles(img),
                                                   prob_thresh=self.threshold,
                                                   nms_thresh=0.01,
                                                   show_tile_progress=False)

        print(f"spot prediction: found {np.max(labels)} spots")
        print(f"spot prediction: calculation time={time.process_time() - start_time} s")

        # get spot properties
        # load slice from image
        with h5py.File(self.img_dir / self.img_file, 'r') as f:
            intensity_img = np.stack([f[channel][eval(str(self.data.img_slice))] for channel in [self.data.fg_channel, self.data.bg_channel]], axis=-1)

        spot_props = regionprops_table(labels,
                                       intensity_image=intensity_img,
                                       properties=['label', 'area', 'extent',
                                                   'intensity_max', 'intensity_min', 'intensity_mean', ],
                                       extra_properties=(self.intensity_median, self.intensity_std,))
        spot_props = pd.DataFrame(spot_props)
        col_names = spot_props.columns
        for idx, channel in enumerate([self.data.fg_channel, self.data.bg_channel]):
            col_names = [x.replace(f"-{idx}", f"_{channel}") if x.endswith(f"-{idx}") else x for x in col_names]
        spot_props.columns = col_names

        # add spot positions
        spot_props[['z', 'y', 'x']] = res['points']
        # add prediction probability
        spot_props[['prob']] = res['prob'][:, np.newaxis]
        # modify label
        frame_id = 0
        spot_props['label'] = spot_props['label'].apply(lambda x: f'{frame_id}_{str(x)}')
        # add frame
        spot_props['frame'] = frame_id
        # add spot class
        spot_props['spot_class'] = self.spot_class
        # add filename
        spot_props['img_file'] = self.img_file

        # save spot table
        spot_props_file = self.predictions_dir / self.img_dir.name / f"{self.img_file.replace('.h5', '')}_spot_table_{self.model_dir.name}.csv"
        if spot_props_file.exists():
            spot_props_file.unlink()
        spot_props.to_csv(spot_props_file, index=False)

        # prepare prediction output file
        img_out = self.predictions_dir / self.img_dir.name / (self.img_file.replace(".h5", "") + "_prediction_" + self.model_dir.name + ".h5")
        if img_out.exists():
            img_out.unlink()

        # load image, save together with StarDist predictions
        chunks = (64, np.min((256, img.shape[1])), np.min((256, img.shape[2])))
        with h5py.File(img_out, 'a') as f_out:
            with h5py.File(self.img_dir / self.img_file, 'r') as f_in:
                for channel in [self.data.fg_channel, self.data.bg_channel]:
                    f_out.create_dataset(channel,
                                         data=f_in[channel][eval(str(self.data.img_slice))].astype(np.uint16),
                                         dtype=np.uint16,
                                         chunks=chunks,
                                         compression=5,
                                         )
            f_out.create_dataset(
                f"StarDist_labels_{self.spot_class}",
                data=labels.astype(np.uint16),
                dtype=np.uint16,
                chunks=chunks,
                compression=5,
            )
            # make dummy spots
            if len(res["points"]) > 0:
                labels = np.zeros_like(labels, dtype=np.uint16)
                labels[tuple(res['points'].T)] = 1
                labels, _ = ndi.label(labels)
                labels = expand_labels(labels,
                                       distance=np.sqrt(np.sum([x ** 2 for x in self.data.spot_display_radius]))).astype(np.uint16)
            else:
                labels = np.zeros(self.data.img_shape, dtype=np.uint16)
            f_out.create_dataset(
                f"StarDist_spot_dummy_{self.spot_class}",
                data=labels,
                dtype=np.uint16,
                chunks=chunks,
                compression=5,
            )

    @staticmethod
    def intensity_median(regionmask, intensity):
        return np.median(intensity[regionmask])

    @staticmethod
    def intensity_std(regionmask, intensity):
        return np.std(intensity[regionmask])

    @staticmethod
    def histogram_matcher(ref_f=None, img_f=None, main_channel="ch1", bgnd_channel="ch2", channel_type=""):
        # get reference image for proper image intensity adjustment
        with h5py.File(ref_f, "r") as f1, h5py.File(img_f, "r") as f2:
            img_ref = np.stack((f1[main_channel][:], f1[bgnd_channel][:]), axis=3)
            img = np.stack(
                (
                    f2[(main_channel + channel_type)][:],
                    f2[(bgnd_channel + channel_type)][:],
                ),
                axis=3,
            )
        img = match_histograms(img, img_ref, channel_axis=-1)
        return img
