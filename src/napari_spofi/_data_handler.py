import json
import pickle
import time
import warnings
import traceback
import pprint
from pathlib import Path

import napari
from napari.layers import Points
import numpy as np
import pandas as pd
import h5py
from PIL import Image as PImage, ImageDraw
from tifffile import imwrite
from scipy import ndimage as ndi
from skimage.feature import blob_log
from skimage.filters import difference_of_gaussians
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from skimage.segmentation import watershed
from qtpy.QtWidgets import QWidget, QMessageBox
from .spofi_utils.directory_maker import directory_maker
from .spofi_utils.normalizer import get_normalization_limits
from .spofi_utils.normalizer import normalize_to_range


# check use of gpu
try:
    from stardist import gputools_available
    if gputools_available():
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.8, allow_growth=False, total_memory=6000)
    import pyclesperanto as cle
    print("Available OpenCL devices:" + str(cle.available_device_names()))
    cle.select_device()  # defaults to None
    print("Using OpenCL device: " + cle.get_device().name)
except Exception as excptn:
    print(excptn, traceback.format_exc())


class DataHandler:
    """
    prepare spofi data container, load defaults from file
    """
    def __init__(self, viewer: napari.Viewer, defaults_file: str):
        self.viewer = viewer
        self.start_time = time.strftime("%Y%m%d-%H%M%S")

        if defaults_file:
            self.defaults_file = defaults_file
        else:
            self.defaults_file = (Path(__file__).parent / "resources" / "spofi_defaults.json").as_posix()
        print(f"startup: {self.defaults_file=}")
        self.load_defaults()

        new_args = ["annotation_dir", "annotation_file",
                    "img_dir", "img_files", "img_file",
                    "img_slice", "raw_img_shape",
                    "fg_channels", "fg_channel", "bg_channels", "bg_channel",
                    "annotation_data",
                    "sd_images", "sd_masks", "sd_predictions", "sd_models",
                    "model_dir", "losses", "losses_file", ]
        self.new_args = {_: None for _ in new_args}
        self.set_new_args()
        self.__dict__.update((k, v) for (k, v) in self.new_args.items())
        self.tile_ids = dict()
        self.model = None

    def set_new_args(self):
        self.new_args["annotation_data"] = {k: self.make_spots_df() for k in ("true", "predicted")}
        self.new_args["annotation_dir"] = (Path(self.default_annotation_dir)).as_posix()
        self.new_args["img_dir"] = (Path(self.default_img_dir) / "None").as_posix()
        self.new_args["model_dir"] = (Path(self.default_model_dir) / "None").as_posix()
        self.new_args["fg_channel"] = self.default_fg_channel
        self.new_args["bg_channel"] = self.default_bg_channel
        self.new_args["losses"] = {"training": [], "validation": []}
        self.new_args["losses_file"] = None

    def load_defaults(self) -> None:
        try:
            with open(self.defaults_file) as f:
                data_dict = json.load(f)
            self.__dict__.update((k, v) for (k, v) in data_dict.items() if not k.startswith("_"))
        except FileNotFoundError as err:
            print(err, traceback.format_exc())

    def make_checkerboard(self):
        """
        make checkerboard from default image shape and tile shape
        """
        img_size = self.img_shape[1:]
        tile_size = self.tile_shape[1]
        img = np.zeros(img_size, dtype=np.uint8)
        for start_i in range(2):
            for r_i in range(start_i * tile_size, img_size[0], (tile_size * 2)):
                for c_i in range(
                        start_i * tile_size, img_size[0], (tile_size * 2)
                ):
                    img[r_i:(r_i + tile_size), c_i:(c_i + tile_size)] = 255
        return img

    @staticmethod
    def make_spots_df():
        df = pd.DataFrame(
            {
                key: np.empty(0, dtype=object)
                for key in [
                    "img_dir",
                    "img_file",
                    "img_id",
                    "img_slice",
                    "tile_id",
                    "tile_slice",
                    "active_tile",
                    "fg_channel",
                    "bg_channel",
                    "spot_pos",
                    "normalization_type",
                    "normalization_range",
                    "normalization_values",
                    "timestamp",
                ]
            }
        )
        return df

    def check_tile(self, cursor_pos):
        tile_pos, tile_slice, tile_idx, ul_lr = self.cursor_pos_to_tile(cursor_pos=cursor_pos)
        tile_slice = tile_slice[0]
        tile_idx = tile_idx[0]
        ul = ul_lr[0][0]
        lr = ul_lr[0][1]

        # check tile list
        if tile_idx in self.tile_ids.keys():
            if self.tile_ids[tile_idx]["is_active"] > 0:
                qm = QMessageBox
                ret = qm.question(QWidget(), "", f"deactivate tile {tile_idx}?", qm.Yes | qm.No,)
                if ret == qm.Yes:
                    self.tile_ids[tile_idx]["is_active"] = -1
                    tile_color = self.tile_color[0]
                else:
                    return
            else:
                qm = QMessageBox
                ret = qm.question(QWidget(), "", f"activate tile {tile_idx}?", qm.Yes | qm.No,)
                if ret == qm.Yes:
                    self.tile_ids[tile_idx]["is_active"] = 1
                    tile_color = self.tile_color[1]
                else:
                    return
        else:
            self.tile_ids[tile_idx] = dict(is_active=1, tile_slice=tile_slice)
            tile_color = self.tile_color[1]

        # get 'tiles' layer data
        tiles_img = self.viewer.layers["tiles"].data
        tiles_img = PImage.fromarray(tiles_img)
        draw = ImageDraw.Draw(tiles_img)
        draw.rectangle(xy=(ul, lr), fill=0, outline=tile_color, width=1)
        self.viewer.layers["tiles"].data = np.array(tiles_img)
        self.viewer.layers["tiles"].interpolation2d = "nearest"
        self.viewer.layers["tiles"].interpolation3d = "nearest"

    def add_predicted_spots_to_tile(self, cursor_pos):
        print("one could imagine to add predicted spots from this tile to 'true' spots...")

    def load_img(self):
        # check for new start (self.annotation_file still None)
        if isinstance(self.annotation_file, type(None)):
            self.annotation_dir = self.new_args["annotation_dir"]
            self.annotation_file = self.new_args["annotation_file"]
            self.img_dir = self.new_args["img_dir"]
            self.img_file = self.new_args["img_file"]
            # add know annotation data
            annotation_files = sorted([f.stem for f in Path(self.annotation_dir).glob(("annotation_data*" + ".pickle"))], reverse=True)
            print(f"load img\n\tannotation dir: {self.annotation_dir}\n\tannotation files: {annotation_files}")
            if len(annotation_files):
                self.load_annotation_data(annotation_dir=self.annotation_dir,
                                          annotation_file=(annotation_files[0] + ".pickle"))

            self.dump_data(spot_type="true")

            # prepare data directories (for new annotation or skip)
            self.prepare_sd_data_dirs(overwrite=False, verbose=False)

            # update self args with 'new' args
            for (k, v) in self.new_args.items():
                setattr(self, k, v)

            # get slice info for new image
            self.calc_img_slice()

            # load image from file
            channels = [self.fg_channel, self.bg_channel]
            with h5py.File(Path(self.img_dir) / self.img_file, "r", ) as f:
                for idx in range(len(channels)):
                    img = f[channels[idx]][self.img_slice]
                    img = self.blur_it(img)
                    self.viewer.add_image(
                        data=img,
                        name=channels[idx],
                        colormap=self.viewer_colors[idx],
                        blending="additive",
                        interpolation2d="nearest",
                        interpolation3d="nearest",
                        )

            self.add_spots_layer(name="true", symbol="disc",
                                 border_color=self.border_colors[0],
                                 face_color=self.face_colors[0],
                                 editable=True
                                 )

            # add points layer, add know points from annotation data
            annotation_files = sorted([f.stem for f in Path(self.annotation_dir).glob(("annotation_data*" + ".pickle"))], reverse=True)
            if len(annotation_files):
                self.load_annotation_data(annotation_dir=self.annotation_dir,
                                          annotation_file=(annotation_files[0] + ".pickle"))

            self.dump_data(spot_type="true")

            self.add_known_spots_tiles()

            # set viewer to z-projection
            self.viewer.dims.ndisplay = 3
            return

        # update annotation data
        self.update_annotation_data()
        self.dump_data(spot_type="true")

        # reset viewer
        self.reset_viewer()

        # prepare data directories (for new annotation or skip)
        self.prepare_sd_data_dirs(overwrite=False, verbose=False)

        # update self args with 'new' args
        for (k, v) in self.new_args.items():
            setattr(self, k, v)

        # get slice info for new image
        self.calc_img_slice()

        # load image from file
        channels = [self.fg_channel, self.bg_channel]
        with h5py.File(Path(self.img_dir) / self.img_file, "r", ) as f:
            for idx in range(len(channels)):
                img = f[channels[idx]][self.img_slice]
                self.viewer.add_image(
                    data=img,
                    name=channels[idx],
                    colormap=self.viewer_colors[idx],
                    blending="additive",
                    interpolation2d="nearest",
                    interpolation3d="nearest",
                    )

        self.add_spots_layer(name="true", symbol="disc",
                             border_color=self.border_colors[0],
                             face_color=self.face_colors[0],
                             editable=True,
                             )

        # add points layer, add know points from annotation data
        annotation_files = sorted([f.stem for f in Path(self.annotation_dir).glob(("annotation_data*" + ".pickle"))], reverse=True)
        if len(annotation_files):
            self.load_annotation_data(annotation_dir=self.annotation_dir,
                                      annotation_file=(annotation_files[0] + ".pickle"))

        self.dump_data(spot_type="true")

        self.add_known_spots_tiles()

        # set viewer to z-projection
        self.viewer.dims.ndisplay = 3

    def blur_it(self, img):
        """blur image with given sigma, use pyclesperanto"""
        sigmas = self.display_blur_sigmas
        if len(sigmas) == 1:
            sigmas = np.repeat(sigmas, 3)
        try:
            img = cle.pull(cle.gaussian_blur(cle.push(img), sigma_z=sigmas[0], sigma_y=sigmas[1], sigma_x=sigmas[2]))
        except Exception as excptn:
            print(f"pyclesperanto gaussian_blur error: {excptn}")
        return img

    def get_spots(self):
        # check spots in 'true' layer
        spot_pos = {"true": []}
        for layer in self.viewer.layers:
            if isinstance(layer, Points) & (layer.name == "true"):
                spot_pos["true"] = layer.data[[np.sum(x) > 0 for x in layer.data]]  # pos(0,0,0) special!
        return spot_pos

    def calc_img_slice(self):
        # get image raw shape and calculate slice matching default img shape
        with h5py.File(Path(self.img_dir) / self.img_file, "r") as f:
            raw_img_shape = f[self.fg_channel].shape
        if raw_img_shape[0] < self.img_shape[0]:
            # raise Exception("image z axis < 64")
            warnings.warn("image z axis < 64")
        if raw_img_shape[0] >= self.img_shape[0]:
            if isinstance(self.slice_mode, str):
                if self.slice_mode == "center":
                    img_slice_z = slice(
                        int(raw_img_shape[0] / 2 - int(self.img_shape[0] / 2)),
                        int(raw_img_shape[0] / 2 + int(self.img_shape[0] / 2)),
                    )
                if self.slice_mode == "top":
                    img_slice_z = slice(0, self.img_shape[0])
        if raw_img_shape[1] >= self.img_shape[1]:
            img_slice_xy = slice(
                int(raw_img_shape[1] / 2 - int(self.img_shape[1] / 2)),
                int(raw_img_shape[1] / 2 + int(self.img_shape[1] / 2)),
            )
        else:
            img_slice_xy = slice(0, self.img_shape[1])
        img_slice = (
            img_slice_z,
            img_slice_xy,
            img_slice_xy,
            )
        print(
            f"\tinput image size: {raw_img_shape}, actually used region: {img_slice} to match default size {self.img_shape}\n"
        )

        self.img_slice = img_slice
        self.raw_img_shape = raw_img_shape

    def prepare_sd_data_dirs(self, overwrite=False, verbose=False):
        for name in ["images", "masks", "predictions", "models"]:
            new_dir = Path(self.new_args["annotation_dir"]) / "SD_data" / name
            directory_maker(new_dir, overwrite=overwrite, verbose=verbose)
            self.new_args[f"sd_{name}"] = new_dir.as_posix()

    def reset_viewer(self):
        # cleanup napari viewer, remove existing image layers
        try:
            for layer in self.viewer.layers[:]:
                if layer.name not in ["checkerboard", "tiles"]:
                    self.viewer.layers.remove(layer)
            self.viewer.layers["tiles"].data = np.zeros(self.img_shape[1:])
            self.viewer.dims.ndisplay = 3
            self.viewer.layers["tiles"].interpolation2d = "nearest"
            self.viewer.layers["tiles"].interpolation3d = "nearest"
        except Exception as e:
            print(e, traceback.format_exc())

    def dump_data(self, spot_type="true"):
        """
        dump annotation data
        """
        with open(self.annotation_file, "wb") as f:
            pickle.dump(self.annotation_data[spot_type], f, pickle.HIGHEST_PROTOCOL)

    def load_annotation_data(self, annotation_dir=None, annotation_file=None):
        # load annotation data from file
        try:
            with open(Path(annotation_dir) / annotation_file, 'rb') as f:
                annotation_data = pickle.load(f)
                self.annotation_data["true"] = annotation_data
        except FileNotFoundError as err:
            print(err, traceback.format_exc())

    @staticmethod
    def read_annotation_img_files(annotation_dir=None, annotation_file=None, img_dir=None):
        # read used img files from annotation data
        try:
            with open(Path(annotation_dir) / annotation_file, 'rb') as f:
                annotation_data = pickle.load(f)
        except FileNotFoundError as err:
            print(err, traceback.format_exc())

        used_img_files = []
        if len(annotation_data):
            # extract args
            used_img_files = annotation_data.loc[annotation_data["img_dir"] == img_dir, "img_file"].unique()
        return used_img_files

    def read_annotation_data(self, annotation_dir=None, annotation_file=None):
        # read annotation data from file
        try:
            with open(Path(annotation_dir) / annotation_file, 'rb') as f:
                annotation_data = pickle.load(f)
        except FileNotFoundError as err:
            print(err, traceback.format_exc())

        if not len(annotation_data):
            warnings.warn("Ooops, annotation data file is empty!")
            return

        # extract params
        img_dir = annotation_data["img_dir"].unique()[0]
        self.new_args["img_dir"] = img_dir
        available_img_files = sorted([x.name for x in Path(img_dir).glob(("*" + ".h5"))])
        # filter available h5 files, get names matching typical ScanImage output names
        # filenames = [f for f in filenames if re.search(r"(.*\d{5}\.h5)", f)]
        used_img_files = annotation_data.loc[annotation_data["img_dir"] == img_dir, "img_file"].unique()
        available_img_files = np.concatenate((used_img_files, sorted(set(available_img_files) - set(used_img_files))))
        self.new_args["img_files"] = available_img_files
        self.new_args["img_file"] = available_img_files[0]

        fg_channels = annotation_data.loc[(annotation_data["img_dir"] == img_dir)
                                          & (annotation_data["img_file"] == used_img_files[0]), "fg_channel"].unique()
        bg_channels = annotation_data.loc[(annotation_data["img_dir"] == img_dir)
                                          & (annotation_data["img_file"] == used_img_files[0]), "bg_channel"].unique()
        self.new_args["fg_channels"] = fg_channels
        self.new_args["fg_channel"] = fg_channels[0]
        self.new_args["bg_channels"] = bg_channels
        self.new_args["bg_channel"] = bg_channels[0]

        # look for normalization type, check for consistency
        normalization_type_from_file = annotation_data["normalization_type"].iloc[0]
        if not self.normalization_type == normalization_type_from_file:
            qm = QMessageBox
            ret = qm.question(QWidget(), "", f"Different normalization type found! Continue?", qm.Yes | qm.No,)
            if ret == qm.Yes:
                # remove normalization data!
                self.normalization_type = normalization_type_from_file
                print(f"new normalization type: {normalization_type_from_file}")
            else:
                warnings.warn("Check normalization type settings!")
                return

    def add_spots_layer(self, data=np.zeros((1, 3)), name="true", symbol="disc",
                        size=6,
                        border_color="green",
                        face_color="red",
                        editable=True,
                        ):
        self.viewer.add_points(
            data=data,
            name=name,
            ndim=3,
            n_dimensional=True,
            symbol=symbol,
            size=size,
            border_width=0.1,
            border_color=border_color,
            face_color=face_color,
            blending="additive",
            opacity=0.8,
            visible=True,
        )
        self.viewer.layers[name].editable=editable

    def update_annotation_data(self):
        # get spots from active image
        spot_pos = self.get_spots()

        # get new img id
        img_id_new = self.get_img_id()

        # get normalization values
        normalization_values = self.get_normalization_values()

        # get active tiles
        active_tiles = [idx for idx in self.tile_ids.keys() if self.tile_ids[idx]["is_active"] > 0]

        # calculate tile positions, prepare spots dataframes
        spot_types = ["true", ]
        spots_df = {key: self.make_spots_df() for key in spot_types}
        for spot_type in spot_types:
            if len(spot_pos[spot_type]) > 0:
                spots_df[spot_type]["spot_pos"] = [x for x in spot_pos[spot_type]]
                tile_pos, tile_slices, tile_ids, ul_lr = self.cursor_pos_to_tile(cursor_pos=spot_pos[spot_type])
                spots_df[spot_type]["img_dir"] = self.img_dir
                spots_df[spot_type]["img_file"] = self.img_file
                spots_df[spot_type]["img_id"] = img_id_new
                spots_df[spot_type]["img_slice"] = pprint.pformat(self.img_slice)
                spots_df[spot_type]["tile_id"] = tile_ids
                spots_df[spot_type]["tile_slice"] = [pprint.pformat(x) for x in tile_slices]
                spots_df[spot_type]["active_tile"] = [1 if tile_id in active_tiles else -1 for tile_id in tile_ids]
                spots_df[spot_type]["fg_channel"] = self.fg_channel
                spots_df[spot_type]["bg_channel"] = self.bg_channel
                spots_df[spot_type]["normalization_type"] = self.normalization_type
                for idx in spots_df[spot_type].index:
                    spots_df[spot_type].at[idx, "normalization_range"] = self.normalization_range
                    spots_df[spot_type].at[idx, "normalization_values"] = normalization_values
                spots_df[spot_type]["timestamp"] = pd.Timestamp("now")

        # get known/loaded 'true' annotation data (from other images)
        current_annotation_data = self.annotation_data["true"].copy()
        current_annotation_data = current_annotation_data.loc[
            (current_annotation_data["img_dir"] != self.img_dir)
            | ((current_annotation_data["img_dir"] == self.img_dir)
               & (current_annotation_data["img_file"] != self.img_file))]

        if np.all([len(x) > 0 for x in [current_annotation_data, spots_df["true"]]]):
            df_ = pd.concat([current_annotation_data, spots_df["true"]], sort=False, ignore_index=True)
            self.annotation_data["true"] = df_
            return
        if (len(current_annotation_data) == 0) & (len(spots_df["true"]) > 0):
            df_ = spots_df["true"]
            self.annotation_data["true"] = df_
            return
        if (len(current_annotation_data) > 0) & (len(spots_df["true"]) == 0):
            df_ = current_annotation_data
            self.annotation_data["true"] = df_
            return
        return

    def get_img_id(self):
        # get image id
        img_id_new = 0
        df = self.annotation_data["true"]
        img_ids = df.img_id.unique()
        if len(img_ids):
            img_id_current = df.loc[(df.img_dir == self.img_dir)
                                    & (df.img_file == self.img_file), "img_id"].unique()
            if len(img_id_current):
                img_id_new = np.max(img_id_current)
            else:
                img_ids_max = np.max(img_ids)
                img_id_new = img_ids_max + 1
        return img_id_new

    def get_normalization_values(self):
        # get normalization_values
        df = self.annotation_data["true"]
        df = df.loc[(df.img_dir == self.img_dir)
                    & (df.img_file == self.img_file), ["normalization_values"]]
        normalization_values = np.empty(0, dtype=object)
        for idx in range(len(df)):
            if isinstance(df.iloc[idx]["normalization_values"], dict):
                normalization_values = df.iloc[idx]["normalization_values"]
                break
        return normalization_values

    def cursor_pos_to_tile(self, cursor_pos=None):
        """
        convert spot pos/mouse double click position to tile slice and tile idx
        """
        if isinstance(cursor_pos, tuple):
            if len(cursor_pos) == 2:
                cursor_pos = [0, *cursor_pos]
            cursor_pos = list((cursor_pos,))

        tiles = [list(self._to_tiles(x)) for x in cursor_pos]

        # calculate rectangle edges to display tile
        ul_lr = [self._get_ul_lr(x) for x in tiles]

        slices = [
            (
                slice(0, self.tile_shape[0]),
                slice(x[1] * self.tile_shape[1], (x[1] + 1) * self.tile_shape[1]),
                slice(x[2] * self.tile_shape[2], (x[2] + 1) * self.tile_shape[2]),
            )
            for x in tiles
        ]

        # get tile index
        tile_ids = [int(t[1] * self.img_shape[1] / self.tile_shape[1] + t[2]) for t in tiles]

        return tiles, slices, tile_ids, ul_lr

    def add_known_spots_tiles(self):
        # filter current annotation data for matching image dir & file
        df = self.annotation_data["true"]
        df = df.loc[
            (df["img_dir"] == self.img_dir)
            & (df["img_file"] == self.img_file)
            & (df["fg_channel"] == self.fg_channel)
            & (df["bg_channel"] == self.bg_channel)
            ]
        if len(df) == 0:
            spots = []  # np.zeros((1, 3))
        else:
            spots = list(df["spot_pos"].to_numpy())

        if np.any([layer.name == "true" for layer in self.viewer.layers]):
            self.viewer.layers["true"].data = spots
        else:
            self.add_spots_layer(data=spots, name="true", symbol="disc",
                                 border_color=self.border_colors[0],
                                 face_color=self.face_colors[0],
                                 editable=True,
                                 )

        # reset current tile idx list
        self.tile_ids = dict()
        if len(df) == 0:
            # print("add known spots: nothing to add")
            return

        # get 'tiles' layer data
        tiles_img = self.viewer.layers["tiles"].data
        tiles_img = PImage.fromarray(tiles_img)
        draw = ImageDraw.Draw(tiles_img)
        # get tile id from spot pos
        tile_pos, tile_slice, tile_idx, ul_lr = self.cursor_pos_to_tile(cursor_pos=df["spot_pos"])
        tiles_uni, idx_uni = np.unique(tile_idx, return_index=True)
        for tile_id, idx in zip(tiles_uni, idx_uni):
            self.tile_ids[tile_id] = dict()
            self.tile_ids[tile_id]["tile_slice"] = tile_slice[idx]
            is_active = df.loc[(df["tile_id"] == tile_id), "active_tile"].iloc[0]
            self.tile_ids[tile_id]["is_active"] = is_active
            if is_active > 0:
                tile_color = self.tile_color[1]
            else:
                tile_color = self.tile_color[0]
            ul = ul_lr[idx][0]
            lr = ul_lr[idx][1]
            draw.rectangle(xy=(ul, lr), fill=0, outline=tile_color, width=1)

        self.viewer.layers["tiles"].data = np.array(tiles_img)
        self.viewer.layers["tiles"].interpolation2d = "nearest"
        self.viewer.layers["tiles"].interpolation3d = "nearest"

    def _get_ul_lr(self, pos):
        ul = (int(pos[2] * self.tile_shape[2] + 0), int(pos[1] * self.tile_shape[1] + 0))
        lr = (
            int((pos[2] + 1) * self.tile_shape[2] - 1),
            int((pos[1] + 1) * self.tile_shape[1] - 1),
        )
        return ul, lr

    def _to_tiles(self, x):
        # Sort each sublist
        return [int(y // self.tile_shape[i]) for (i, y) in enumerate(x)]

    def suggest_spots(self):
        """
        suggest spots for active tiles based on Laplacian of Gaussian, add suggested spots to "suggested" layer
        """

        print("suggesting spots")

        # get image, use foreground channel (id==0) to find spots
        with h5py.File(Path(self.new_args['img_dir']) / self.new_args['img_file'], "r") as f:
            img = f[self.fg_channel][self.img_slice]

        # pad img in xy
        t_pad = self.tile_shape[0] // 2
        img_padded = np.pad(img, ((0, 0), (t_pad, t_pad), (t_pad, t_pad)), mode="constant", constant_values=0)

        # iterate over tiles
        print(f"{self.new_args['img_dir']}, {self.new_args['img_file']}, {self.tile_ids=}")
        spot_pos = []
        for tile_id in self.tile_ids.keys():
            tile = self.tile_ids[tile_id]
            if tile['is_active'] == -1:
                continue
            # increase tile positions in xy
            tile_slice = tile['tile_slice']
            starts = [(x.start - 0) for (x, p) in zip(tile_slice, [0, t_pad, t_pad])]
            stops = [(x.stop + 2*p) for (x, p) in zip(tile_slice, [0, t_pad, t_pad])]
            tile_slice_padded = (slice(starts[0], stops[0]),
                                 slice(starts[1], stops[1]),
                                 slice(starts[2], stops[2]))

            img_tile_padded = img_padded[tile_slice_padded]

            # find spots, add spot positions to dummy array
            blobs = blob_log(img_tile_padded, min_sigma=0., max_sigma=1.5, num_sigma=2, threshold=None, overlap=0.5, log_scale=False, threshold_rel=0.1, exclude_border=True)
            if len(blobs) > 0:
                # get absolute positions
                spot_pos_ = blobs[:, :3]
                spot_pos_[:, 1] = spot_pos_[:, 1] + starts[1] - t_pad
                spot_pos_[:, 2] = spot_pos_[:, 2] + starts[2] - t_pad
                spot_pos.append(spot_pos_)
        if len(spot_pos):
            spot_pos = np.vstack(spot_pos)
            spot_pos = np.unique(spot_pos, axis=0)

        # replace existing suggested spot layers
        if np.any([layer.name == "suggested" for layer in self.viewer.layers]):
            self.viewer.layers["suggested"].data = spot_pos
        else:
            self.add_spots_layer(data=spot_pos, name="suggested", symbol="x",
                                 size=8,
                                 border_color=self.border_colors[2],
                                 face_color=self.face_colors[2],
                                 editable=False,
                                 )

    def extract_spots(self):
        # update annotation data
        self.update_annotation_data()
        self.dump_data(spot_type="true")

        # check image normalization data
        df = self.annotation_data["true"].copy()
        for img_dir in df.img_dir.unique():
            for img_file in df.loc[df["img_dir"] == img_dir, 'img_file'].unique():
                _df = df.loc[(df["img_dir"] == img_dir)
                             & (df["img_file"] == img_file)
                             & (df["fg_channel"] == self.fg_channel)
                             & (df["bg_channel"] == self.bg_channel)
                             & (df["normalization_type"] == self.normalization_type), ]

                normalization_values = np.empty(0, dtype=object)
                for idx in range(len(_df)):
                    if isinstance(_df.iloc[idx]["normalization_values"], dict):
                        normalization_values = _df.iloc[idx]["normalization_values"]
                        break
                if not isinstance(normalization_values, dict):
                    # print(f"{img_dir}, {img_file}, calculating normalization values")
                    channels = [self.fg_channel, self.bg_channel]
                    normalization_values = get_normalization_limits(
                        path=Path(img_dir),
                        file=img_file,
                        channels=channels,
                        norm_ranges=self.normalization_range,
                        norm_type=self.normalization_type,
                    )

                ids = self.annotation_data["true"].loc[
                    (self.annotation_data["true"]["img_dir"] == img_dir)
                    & (self.annotation_data["true"]["img_file"] == img_file)].index
                for idx in ids:
                    self.annotation_data["true"].at[idx, "normalization_values"] = normalization_values

        self.dump_data()
        self.make_training_tiles()

    @staticmethod
    def get_props(mask, img):
        spot_props = regionprops_table(mask, intensity_image=img,
                                       properties=('label', 'centroid_weighted', 'area', 'intensity_max', 'equivalent_diameter_area'),
                                       extra_properties=None, cache=True, separator='-', )
        spot_props = pd.DataFrame(spot_props)
        spot_props.rename(columns={"centroid_weighted-0": "centroid_z", "centroid_weighted-1": "centroid_y", "centroid_weighted-2": "centroid_x"}, inplace=True)
        return spot_props

    def make_training_tiles(self):
        """
        make img and mask tiles for training
        """
        if self.spot_extraction_procedure == 1:
            print(f"create spot mask with procedure: {self.spot_extraction_procedure}, using spot intensity threshold: {self.spot_intensity_threshold}")
        if self.spot_extraction_procedure == 2:
            print(f"create spot mask with procedure: {self.spot_extraction_procedure} (DOG + watershed)")

        # delete existing files in "images" and "masks" directory
        for file in sorted(Path(self.sd_images).rglob("*.tif")):
            (Path(self.sd_images) / file).unlink()
        for file in sorted(Path(self.sd_masks).rglob("*.tif")):
            (Path(self.sd_masks) / file).unlink()

        # grasp "true" annotation data
        spot_type = "true"
        df = self.annotation_data[spot_type].copy()

        t_pad = self.tile_shape[0] // 2
        tile_end_offset = (0, self.tile_shape[0], self.tile_shape[0])
        tile_padded_offset = (0, t_pad, t_pad)
        spot_expand_distance = (3 * self.spot_radius**2)**0.5

        # iterate over all annotated images
        for img_dir in df['img_dir'].unique():
            for img_file in df.loc[df['img_dir'] == img_dir, 'img_file'].unique():
                spot_data = df.loc[(df['img_dir'] == img_dir)
                                    & (df["img_file"] == img_file)
                                    & (df["active_tile"] == 1),
                ["img_id", "img_slice", "tile_id", "tile_slice", "spot_pos", "active_tile", "normalization_values"]].copy()

                if not len(spot_data):
                    continue

                spot_data.reset_index(inplace=True)

                normalization_values = spot_data.iloc[0]["normalization_values"]
                img_id = spot_data.iloc[0]["img_id"]
                img_slice = spot_data.iloc[0]["img_slice"]

                with h5py.File(Path(img_dir) / img_file, "r") as f:
                    img = np.stack([f[channel][eval(img_slice)] for channel in [self.fg_channel, self.bg_channel]],
                                   axis=-1)
                # pad in xy
                img_padded = np.stack([np.pad(img[..., i], ((0, 0), (t_pad, t_pad), (t_pad, t_pad)), mode="constant", constant_values=0) for i in range(2)], axis=-1)

                # iterate over tiles
                for tile_id, grp in spot_data.groupby("tile_id"):
                    center_tile_slice = eval(grp.iloc[0].tile_slice)
                    img_tile = np.stack([img[..., channel_id][center_tile_slice]
                                        for channel_id in range(2)],
                                        axis=-1)
                    # normalize
                    img_tile = normalize_to_range(img_tile, normalization_values)

                    # write image tile (multichannel, normalized data) to file
                    file = f'img_{img_id:.0f}_{tile_id:05.0f}'
                    self._write_img_tile(img_tile=img_tile,
                                         path=Path(self.sd_images),
                                         file=file,
                                         )

                    # increase tile in xy
                    starts = [x.start for x in center_tile_slice]
                    ends = [x.stop for x in center_tile_slice]
                    ends = [(e + to) for (e, to) in zip(ends, tile_end_offset)]
                    center_tile_slice_padded = (slice(starts[0], ends[0]),
                                                slice(starts[1], ends[1]),
                                                slice(starts[2], ends[2]))

                    # use foreground channel (id==0) to create mask
                    img_tile_padded = img_padded[..., 0][center_tile_slice_padded]

                    # iterate over spots, add to dummy array
                    spot_pos_array = np.zeros_like(img_tile_padded, dtype=np.uint16)
                    for idx in spot_data.index:
                        _spot_data = spot_data.iloc[idx][["tile_id", "tile_slice", "spot_pos"]]
                        pos_coords_padded = [(sp - ts.start + tpo).astype(np.uint16) for (sp, ts, tpo) in
                                             zip(_spot_data["spot_pos"], center_tile_slice_padded, tile_padded_offset)]
                        # add spot positions to dummy array if in padded tile
                        if np.all([x < 128 for x in pos_coords_padded]):
                            spot_pos_array[tuple(pos_coords_padded)] = int(idx + 1)

                    # extend spot positions using expected spot radius
                    spot_pos_mask = expand_labels(spot_pos_array, distance=spot_expand_distance)

                    # get smoothed version of foreground channel
                    img_tile_padded_smoothed = ndi.gaussian_filter(img_tile_padded.astype(np.float32), 1.0, order=0,
                                                                   mode='constant', cval=0.0,
                                                                   truncate=4.0, radius=None, axes=None)
                    
                    # get spot properties
                    spot_props = self.get_props(spot_pos_mask, img_tile_padded_smoothed)

                    # recenter spot positions, create spot position array for all spots, overwrite array
                    spot_pos_array = np.zeros_like(img_tile_padded, dtype=np.uint16)
                    for lbl, z, y, x in spot_props[['label', 'centroid_z', 'centroid_y', 'centroid_x']].to_numpy():
                        pos_ = [round(value) for value in [z, y, x]]
                        spot_pos_array[tuple(pos_)] = lbl

                    # extend spot positions using expected spot radius at updated positions
                    spot_pos_mask = expand_labels(spot_pos_array, distance=spot_expand_distance)

                    if self.spot_extraction_procedure == 1:
                        # procedure 1: expand spot positions considering relative intensity following simple Gaussian filter
                        spot_pos_mask_new = np.zeros_like(spot_pos_mask, dtype=np.uint16)
                        spot_props.sort_values(by=['intensity_max', 'area', ], ascending=[False, False, ], inplace=True)
                        for lbl, int_max in spot_props[['label', 'intensity_max']].to_numpy():
                            lbl = int(lbl)
                            _mask = np.where(spot_pos_mask == lbl, 1, 0)
                            _mask = np.where(img_tile_padded_smoothed >= (self.spot_intensity_threshold * int_max), _mask, 0)
                            _mask = ndi.binary_closing(_mask, structure=ndi.generate_binary_structure(3, 3).astype(np.uint8), iterations=1, output=None, origin=0, mask=None, border_value=0, brute_force=False)
                            spot_pos_mask_new = np.where(_mask == 1, lbl, spot_pos_mask_new)
                        spot_pos_mask = spot_pos_mask_new

                    if self.spot_extraction_procedure == 2:
                        # procedure 2: use watershed to describe spots
                        img_tile_padded_dog = difference_of_gaussians(img_tile_padded_smoothed, 0.0, high_sigma=1.0, mode='constant', cval=0, channel_axis=None, truncate=4.0)
                        spot_pos_mask = watershed(-img_tile_padded_dog, markers=spot_pos_array, connectivity=1, offset=None, mask=spot_pos_mask, compactness=1.0, watershed_line=True)

                    # remove padded parts
                    mask_tile = spot_pos_mask[(slice(0, self.tile_shape[0]), slice(t_pad, (self.tile_shape[1] + t_pad)), slice(t_pad, (self.tile_shape[2] + t_pad)))]
                    
                    # write mask tile to file
                    self._write_mask_tile(
                        mask_tile=mask_tile,
                        path=Path(self.sd_masks),
                        file=file,
                    )

    @staticmethod
    def _write_img_tile(img_tile, path, file):
        """
        write image tile (multichannel) to file
        """
        imwrite(
            path / (file + ".tif"),
            img_tile.astype(np.float32),
            imagej=False,
            bigtiff=False,
            photometric="minisblack",
            resolution=(1.0, 1.0),
            metadata={"axes": "ZYXC", "spacing": 1.0, "unit": "pixel"},
        )

        # save for imagej (optional)
        save4ij = True
        if save4ij:
            # swap axes to reach TZCYX format
            img_tile = np.transpose(img_tile, (0, 3, 2, 1))
            imwrite(
                path / (f"x_{file}" + ".tif"),
                img_tile.astype(np.float32),
                imagej=True,
                bigtiff=False,
                photometric="minisblack",
                resolution=(1.0, 1.0),
                metadata={"axes": "ZCYX", "spacing": 1.0, "unit": "pixel"},
            )

    @staticmethod
    def _write_mask_tile(mask_tile, path, file):
        """
        prepare label image
        """
        imwrite(
            path / (file + ".tif"),
            mask_tile.astype(np.uint16),
            imagej=False,
            bigtiff=False,
            photometric="minisblack",
            resolution=(1.0, 1.0),
            metadata={"axes": "ZYX", "spacing": 1.0, "unit": "pixel"},
        )
