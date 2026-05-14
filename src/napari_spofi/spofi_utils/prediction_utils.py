"""
some helper functions for napari-spofi prediction

cs2024
"""

import time
import pprint
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.optimize import curve_fit
from skimage.exposure import match_histograms
from skimage.measure import regionprops_table
from .normalizer import get_normalization_limits
from .normalizer import normalize_to_range
from tensorflow.config import get_visible_devices
from stardist.models import StarDist3D


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
        self.spot_class = self.data.spot_class
        self.frame_id = self.data.frame_id

        # check gpu
        print(get_visible_devices())

    def predict(self):
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

        # save predictions for training tiles
        tile_data = df.loc[(df["img_dir"] == self.data.img_dir)
                           & (df["img_file"] == self.data.img_file)
                           & (df["active_tile"] == 1), ("img_id", "img_slice", "tile_id", "tile_slice")]
        tile_data = tile_data.drop_duplicates(subset=("img_id", "tile_id", "tile_slice"))
        print(f"prediction tiles: {tile_data['tile_id'].unique()}")
        for ind in tile_data.index:
            p_tile_file = f"p_img_{tile_data['img_id'][ind]:.0f}_{tile_data['tile_id'][ind]:05.0f}_prediction_{self.model_dir.name}"
            p_tile_path = self.predictions_dir / self.img_dir.name
            _p_tile_path = self.predictions_dir / self.img_dir.name / (p_tile_file + ".tif")
            if _p_tile_path.exists():
                _p_tile_path.unlink()
            p_tile = labels[eval(tile_data["tile_slice"][ind])]
            self.data._write_mask_tile(p_tile, p_tile_path, p_tile_file)

        # get spot properties
        # load image, mind active slice
        with h5py.File(self.img_dir / self.img_file, 'r') as f:
            img = np.stack([f[channel][eval(str(self.data.img_slice))] for channel in [self.data.fg_channel, self.data.bg_channel]], axis=-1)

        spot_props = regionprops_table(labels,
                                       intensity_image=img,
                                       properties=['label', 'area', 'extent', "coords",
                                                   'intensity_max', 'intensity_min', 'intensity_mean',
                                                   'equivalent_diameter_area',
                                                   "bbox", 'centroid_weighted', ],
                                       extra_properties=(self.intensity_sum, self.intensity_median, self.intensity_std,),
                                       cache=True, separator='-', )
        spot_props = pd.DataFrame(spot_props)

        # add/modify some columns
        spot_props["bbox"] = [pprint.pformat(x.tolist()) for x in
                              np.squeeze([spot_props[f"bbox-{idx}"] for idx in range(6)]).T]
        spot_props.drop(columns=[f"bbox-{idx}" for idx in range(6)], inplace=True)

        for c_idx, channel in enumerate([self.data.fg_channel, self.data.bg_channel]):
            spot_props[f"centroid_weighted_{channel}"] = [pprint.pformat(x.tolist()) for x in np.squeeze(
                [spot_props[f"centroid_weighted-{p_idx}-{c_idx}"] for p_idx in range(3)]).T]
            spot_props.drop(columns=[f"centroid_weighted-{p_idx}-{c_idx}" for p_idx in range(3)], inplace=True)

        for lbl in spot_props["label"]:
            coords = spot_props.loc[spot_props["label"] == lbl, "coords"].values
            coords = np.ravel_multi_index(coords[0].T.tolist(), labels.shape)
            spot_props.loc[spot_props["label"] == lbl, "coords"] = pprint.pformat(coords.tolist())

        # estimate "characteristic" spot intensity (fit histogram, 64 bins)
        spot_props[["fit_amp", "fit_mu", "fit_sigma", "fwhm"]] = ""
        for idx in spot_props.index:
            coords = eval(spot_props["coords"].iloc[idx])
            coords = np.unravel_index(coords, img.shape[:-1])
            spot = img[..., 0][coords] # take fg channel
            hist, bins = np.histogram(spot, bins=np.linspace(np.min(spot), np.max(spot), 64 + 1), density=False)
            #hist, bins = np.histogram(spot, bins=np.linspace(0, np.max(spot), 64 + 1), density=False)
            bins = bins + ((bins[1] - bins[0]) / 2)
            bins = bins[:-1]
            initial_guess = [1. * np.max(hist), bins[np.argmax(hist)], 15]
            bounds = (
            (0.25 * np.max(hist), bins[int(0.01 * len(bins))], 1), (3 * np.max(hist), bins[int(0.99 * len(bins))], 250))
            try:
                popt, _ = curve_fit(self._gaussian_1d, xdata=bins, ydata=hist, bounds=bounds, p0=initial_guess,
                                maxfev=int(1e7), ftol=2.3e-16, xtol=2.3e-16)
                # Calculate FWHM
                # fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
                fwhm = 2.355 * popt[2]
            except Exception as excptn:
                print(f"intensity fitting error: {excptn}")
                popt = [0, 0, 0]
                fwhm = 0
            spot_props.loc[spot_props["label"] == (idx + 1), ["fit_amp", "fit_mu", "fit_sigma", "fwhm"]] = [*popt, fwhm]

        spot_props["equivalent_radius"] = np.round(spot_props['equivalent_diameter_area'] / 2, 2)
        spot_props[['z', 'y', 'x']] = res['points']
        spot_props[['prob']] = res['prob'][:, np.newaxis]
        spot_props['label'] = spot_props['label'].apply(lambda x: f'{self.frame_id}_{str(x)}')
        spot_props['frame'] = self.frame_id
        spot_props['spot_class'] = self.spot_class
        spot_props['img_dir'] = self.img_dir
        spot_props['img_file'] = self.img_file
        spot_props["img_slice"] = pprint.pformat(self.data.img_slice)

        # update columns names
        col_names = spot_props.columns
        for idx, channel in enumerate([self.data.fg_channel, self.data.bg_channel]):
            col_names = [x.replace(f"-{idx}", f"_{channel}") if x.endswith(f"-{idx}") else x for x in col_names]
        spot_props.columns = col_names

        # save spot table
        spot_props_file = self.predictions_dir / self.img_dir.name / f"{self.img_file.replace('.h5', '')}_spot_table_{self.model_dir.name}.csv"
        if spot_props_file.exists():
            spot_props_file.unlink()
        spot_props.to_csv(spot_props_file, index=False)

        # prepare prediction output file
        img_out = self.predictions_dir / self.img_dir.name / (self.img_file.replace(".h5", "") + "_prediction_" + self.model_dir.name + ".h5")
        if img_out.exists():
            img_out.unlink()

        # load input image, save together with StarDist predictions
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

    @staticmethod
    def intensity_median(regionmask, intensity):
        return np.median(intensity[regionmask])

    @staticmethod
    def intensity_std(regionmask, intensity):
        return np.std(intensity[regionmask])

    @staticmethod
    def intensity_sum(regionmask, intensity):
        return np.sum(intensity[regionmask])

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

    @staticmethod
    def _gaussian_1d(x, amplitude, x0, sigma_x):
        res = amplitude * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2)))
        return res
