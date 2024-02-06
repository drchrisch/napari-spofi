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
import pyclesperanto as cle
from PIL import Image as PImage, ImageDraw
from tifffile import imwrite
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
from qtpy.QtWidgets import QWidget, QMessageBox
from .spofi_utils.directory_maker import directory_maker
from .spofi_utils.normalizer import get_normalization_limits
from .spofi_utils.normalizer import normalize_to_range

try:
    print("Available OpenCL devices:" + str(cle.available_device_names()))
    cle.select_device()  # defaults to None
    print("Using OpenCL device: " + cle.get_device().name)
except Exception as e:
    print(e, traceback.format_exc())


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
        self.new_args["annotation_data"] = {k: self.make_spots_df() for k in ("true", "edited", "predicted")}
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
        # self.viewer.layers["tiles"].interpolation = "linear"
        # self.viewer.layers["tiles"].interpolation2d = "linear"
        # self.viewer.layers["tiles"].interpolation3d = "linear"

        #self.update()

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
                        # interpolation3d="nearest",
                        # interpolation="nearest",
                        )

            self.add_spots_layer()

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
                img = self.blur_it(img)
                self.viewer.add_image(
                    data=img,
                    name=channels[idx],
                    colormap=self.viewer_colors[idx],
                    blending="additive",
                    # interpolation3d="nearest",
                    # interpolation="nearest",
                    )

        self.add_spots_layer()

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
        img = cle.pull(cle.gaussian_blur(img, sigma_z=sigmas[0], sigma_y=sigmas[1], sigma_x=sigmas[2]))
        return img

    def get_spots(self):
        # check spots in 'true' and 'edited' layer
        spot_types = ("true", "edited")
        spot_pos = {key: [] for key in spot_types}
        for spot_type in spot_types:
            for layer in self.viewer.layers:
                if isinstance(layer, Points) & (layer.name == spot_type):
                    # get spots from layer
                    spot_pos[spot_type] = layer.data[[np.sum(x) > 0 for x in layer.data]]  # pos(0,0,0) special!
        return spot_pos

    def calc_img_slice(self):
        # get image raw shape and calculate slice matching default img shape
        with h5py.File(Path(self.img_dir) / self.img_file, "r") as f:
            raw_img_shape = f[self.fg_channel].shape
        if raw_img_shape[0] < self.img_shape[0]:
            # raise Exception("image z axis < 64")
            warnings.warn("image z axis < 64")
        if raw_img_shape[0] > self.img_shape[0] or raw_img_shape[1] > self.img_shape[1]:
            img_slice_z = slice(
                int(raw_img_shape[0] / 2 - int(self.img_shape[0] / 2)),
                int(raw_img_shape[0] / 2 + int(self.img_shape[0] / 2)),
            )
            img_slice_xy = slice(
                int(raw_img_shape[1] / 2 - int(self.img_shape[1] / 2)),
                int(raw_img_shape[1] / 2 + int(self.img_shape[1] / 2)),
            )
            img_slice = (
                img_slice_z,
                img_slice_xy,
                img_slice_xy,
            )
        else:
            img_slice = slice(0, self.img_shape[0])
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
            # self.viewer.layers["tiles"].interpolation = "linear"
            # self.viewer.layers["tiles"].interpolation2d = "linear"
            # self.viewer.layers["tiles"].interpolation3d = "linear"
        except Exception as e:
            print(e, traceback.format_exc())

    def dump_data(self, spot_type="true"):
        """
        dump annotation data
        """
        # print(f"dump data: {self.annotation_file}")
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
                """
                for idx in range(len(annotation_data)):
                    annotation_data.at[idx, "normalization_type"] = normalization_type_from_file
                    annotation_data.at[idx, "normalization_values"] = np.empty(0, dtype=object)
                """
            else:
                warnings.warn("Check normalization type settings!")
                return

    def add_spots_layer(self):
        spots = np.zeros((1, 3))
        self.viewer.add_points(
            data=spots,
            name="true",
            ndim=3,
            n_dimensional=True,
            symbol="disc",
            size=6,
            edge_width=0.1,
            edge_color=self.edge_colors[0],
            face_color=self.face_colors[0],
            blending="additive",
            opacity=0.8,
            visible=True,
        )

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
        spot_types = ["true", "edited"]
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

        # check 'edited' spots in active tiles
        # consider only spots in active tiles and move those to 'true'
        # replace 'true' spots that are (too) close to 'edited' spots by 'edited' spots
        spot_pos_true = spots_df["true"].copy()
        spot_pos_true_data = spot_pos_true.spot_pos.to_numpy()
        spot_pos_edited = spots_df["edited"].loc[(spots_df["edited"]["active_tile"] != -1), ].copy()
        spot_pos_edited_data = spot_pos_edited.spot_pos.to_numpy()
        if spot_pos_edited_data.shape[0] > 0:
            # get voxel size to account for non-isotropic image dimension
            voxel_size = self.voxel_size / np.min(self.voxel_size)
            voxel_size = voxel_size[::-1]  # xyz -> zyx

            # calculate 'true' <-> 'edited' spot distances
            spot_pos_true_data = [np.multiply(voxel_size, x) for x in spot_pos_true_data]
            spot_pos_edited_data = [np.multiply(voxel_size, x) for x in spot_pos_edited_data]

            # identify 'true' spots where distance to 'edited' spot is less than min_dist (=5) 'units'
            min_dist = 5
            valid_true_spot_ids = []
            for pos_true in spot_pos_true_data:
                pp_diff = [(pos_true - pos_edited) for pos_edited in spot_pos_edited_data]
                pp_dist = [np.sqrt(np.sum(x ** 2)) for x in pp_diff]
                valid_true_spot_ids.append(np.all([x >= min_dist for x in pp_dist]))
            valid_true_spot_ids = np.where(valid_true_spot_ids)

            # update 'true' spots
            spots_df["true"] = spots_df["true"].iloc[valid_true_spot_ids]
            spots_df["true"] = pd.concat([spots_df["true"], spot_pos_edited], sort=False, ignore_index=True)

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
            spots = np.zeros((1, 3))
        else:
            spots = list(df["spot_pos"].to_numpy())

        if np.any([layer.name == "true" for layer in self.viewer.layers]):
            self.viewer.layers["true"].data = spots
        else:
            self.viewer.add_points(
                data=spots,
                name="true",
                ndim=3,
                n_dimensional=True,
                symbol="disc",
                size=6,
                edge_width=0.1,
                edge_color=self.edge_colors[0],
                face_color=self.face_colors[0],
                blending="additive",
                opacity=0.8,
                visible=True,
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
        # self.viewer.spot_layers["tiles"].interpolation = "linear"
        # self.viewer.spot_layers["tiles"].interpolation2d = "linear"
        # self.viewer.spot_layers["tiles"].interpolation3d = "linear"

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
                    print(f"{img_dir}, {img_file}, calculating normalization values")
                    channels = [self.fg_channel, self.bg_channel]
                    normalization_values = get_normalization_limits(
                        path=Path(img_dir),
                        file=img_file,
                        channels=channels,
                        norm_ranges=self.normalization_range,
                        norm_type=self.normalization_type,
                    )

                idxs = self.annotation_data["true"].loc[
                    (self.annotation_data["true"]["img_dir"] == img_dir)
                    & (self.annotation_data["true"]["img_file"] == img_file)].index
                for idx in idxs:
                    self.annotation_data["true"].at[idx, "normalization_values"] = normalization_values

        self.dump_data()
        self.make_training_tiles()

    def make_training_tiles(self):
        """
        make img and mask tiles for training
        """
        # delete existing files in "images" and "masks" directory
        for file in sorted(Path(self.sd_images).rglob("*.tif")):
            (Path(self.sd_images) / file).unlink()
        for file in sorted(Path(self.sd_masks).rglob("*.tif")):
            (Path(self.sd_masks) / file).unlink()

        # iterate over "true" annotation data in active tiles
        spot_type = "true"
        df = self.annotation_data[spot_type].copy()
        df = df.loc[(df["active_tile"] > 0)]

        for img_dir in df.img_dir.unique():
            for img_file in df.loc[df["img_dir"] == img_dir, 'img_file'].unique():
                _df = df.loc[
                    (df["img_dir"] == img_dir)
                    & (df["img_file"] == img_file)
                    & (df["fg_channel"] == self.fg_channel)
                    & (df["bg_channel"] == self.bg_channel),
                ]
                if len(_df) > 0:
                    normalization_values = _df.iloc[0]["normalization_values"]
                    img_id = _df.img_id.unique()[0]
                    img_slice, fg_channel, bg_channel = _df.iloc[0][["img_slice", "fg_channel", "bg_channel"]]
                    # iterate over tiles
                    tile_ids = _df["tile_id"].unique()
                    for tile_id in tile_ids:
                        tile_slice = _df.loc[_df.tile_id == tile_id, "tile_slice"].unique()[0]
                        # prepare image tile and mask tile file name
                        file = f'img_{img_id:.0f}_{tile_id:05.0f}'
                        # get tile from image file, prepare as (multichannel) image
                        with h5py.File(Path(img_dir) / img_file, "r") as f:
                            img_tile = np.stack([f[channel][eval(img_slice)][eval(tile_slice)]
                                                 for channel in [fg_channel, bg_channel]],
                                                axis=-1)

                        # ensure that z tile shape matches default shape, pad with zeros
                        tile_shape_raw = img_tile.shape
                        img_tile = np.pad(img_tile,
                                          ((0, self.tile_shape[0] - tile_shape_raw[0]), (0, 0), (0, 0), (0, 0)),
                                          mode="constant",
                                          constant_values=0,
                                          )

                        # normalize
                        img_tile = normalize_to_range(img_tile, normalization_values)

                        # write image tile (multichannel) to file
                        self._write_img_tile(img_tile=img_tile,
                                             path=Path(self.sd_images),
                                             file=file,
                                             )

                        # prepare mask
                        # get smoothed version of foreground channel
                        img_tile_smoothed = ndi.gaussian_filter(img_tile[..., 0].astype(np.float32), 1, order=0,
                                                                mode='constant', cval=0.0,
                                                                truncate=4.0, radius=None, axes=None)

                        # get conversion from tile to considered image slice
                        starts = [x.start for x in eval(tile_slice)]

                        # create spot position array for all spots
                        spot_pos_array = np.zeros_like(img_tile_smoothed, dtype=np.uint8)

                        for p_idx, spot_pos in enumerate(_df.loc[_df.tile_id == tile_id, "spot_pos"], start=1):
                            pos_coords = [(sp - s).astype(np.uint8) for (sp, s) in zip(spot_pos, starts)]
                            spot_pos_array[tuple(pos_coords)] = p_idx
                        del pos_coords

                        # expand spot positions
                        _spot_radius = self.spot_radius
                        spot_expand_distance = np.sqrt(
                            np.sum([_spot_radius[key] ** 2 for key in _spot_radius.keys()]))
                        del _spot_radius
                        spot_pos_mask = expand_labels(spot_pos_array, distance=spot_expand_distance).astype(np.uint8)

                        # check spot center position
                        spot_props = regionprops_table(spot_pos_mask, intensity_image=img_tile_smoothed,
                                                       properties=('label', 'centroid_weighted',
                                                                   'intensity_max', 'equivalent_diameter_area'),
                                                       cache=True, separator='-', extra_properties=None, spacing=None)
                        pos_coords = []
                        spot_radii = []
                        spot_max = []
                        for p_idx, lbl in enumerate(spot_props['label']):
                            pos_coords = np.append(pos_coords, np.squeeze(
                                [(spot_props[f"centroid_weighted-{idx}"][p_idx]).astype(np.uint16) for idx in range(3)]))
                            # spot_radius = np.round(((np.sum(spot_pos_mask==p_idx) * 3 / np.pi / 4)**(1/3)), 2)
                            spot_radius = np.round(spot_props['equivalent_diameter_area'][p_idx] / 2, 2)
                            spot_radii = np.append(spot_radii, spot_radius)
                            spot_max = np.append(spot_max, spot_props['intensity_max'])
                        pos_coords = np.reshape(pos_coords, (-1, 3)).astype(np.uint16)
                        del spot_radius

                        # create spot position array for all spots
                        spot_pos_array = np.zeros_like(img_tile_smoothed, dtype=np.uint8)
                        for p_idx, spot_pos in enumerate(pos_coords, start=1):
                            spot_pos_array[tuple(spot_pos)] = p_idx

                        # expand spot positions (use median radius from regionprops)
                        spot_expand_distance = (3 * (np.median(spot_radii) / 2) ** 2) ** 0.5
                        mask_tile = expand_labels(spot_pos_array, distance=spot_expand_distance)

                        # filter by image intensity threshold
                        # get histogram of pixel intensities, correct bin edges to get "x"
                        counts, counts_x = np.histogram(img_tile_smoothed, bins=1024, density=True)
                        counts_x = (counts_x + ((counts_x[1] - counts_x[0]) / 2))[:-1]
                        """
                        # get intensity value for max counts
                        background = counts_x[np.argmax(counts)]
                        """
                        # get intensity value for max counts (5% of bins from low intensity edge)
                        background = counts_x[np.argmax(counts[:int(1024 * 0.05)])]

                        for p_idx, lbl in enumerate(spot_props['label']):
                            mask_tile = np.where((mask_tile == lbl) &
                                                 (img_tile_smoothed < np.max(
                                                     ((0.2 * spot_max[p_idx]), (3 * background)))),
                                                 0,
                                                 mask_tile)

                        """
                        # close holes
                        for p_idx, lbl in enumerate(spot_props['label']):
                            mask_tile_ = remove_small_holes(np.where(mask_tile==lbl, 1, 0), area_threshold=27, connectivity=1, out=None)
                            mask_tile = np.where(mask_tile_, lbl, mask_tile)
                        """

                        # write mask tile to file
                        self._write_mask_tile(
                            mask_tile=mask_tile,
                            path=Path(self.sd_masks),
                            file=file,
                        )

        return

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
            # compression="zlib",
            # compressionargs={"level": 8},
            resolution=(1.0, 1.0),
            metadata={"axes": "ZYXC", "spacing": 1.0, "unit": "pixel"},
        )

        # save for imagej (optional)
        # """
        # swap axes to reach TZCYX format
        img_tile = np.swapaxes(img_tile, 3, 1)
        img_tile = np.swapaxes(img_tile, 2, 3)
        imwrite(
            path / (f"x_{file}" + ".tif"),
            img_tile.astype(np.float32),
            imagej=True,
            bigtiff=False,
            photometric="minisblack",
            # compression=("zlib", 8),
            resolution=(1.0, 1.0),
            metadata={"axes": "ZCYX", "spacing": 1.0, "unit": "pixel"},
        )
        # """

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
            # compression="zlib",
            # compressionargs={"level": 1},
            resolution=(1.0, 1.0),
            metadata={"axes": "ZYX", "spacing": 1.0, "unit": "pixel"},
        )
