from magicgui import magic_factory
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import pprint
from napari import Viewer
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QWidget
from .spofi_utils.prediction_utils import PredictionUtils


class Prediction(QWidget):
    """widget to start prediction"""

    def __init__(self, parent):
        super().__init__(parent)
        self.viewer = parent.viewer
        self.data = parent.data
        global _data
        _data = self.data
        self.widget = self._make_widget()
        self.threshold = None
        self.model_dir = self.data.default_model_dir
        self.worker = []
        self.prediction_file = None

        @self.widget.predict_button.clicked.connect
        def predict():
            # check if prediction is running,
            if self.widget.predict_button.text == "Predict":
                # find model directory, load, prepare training, start worker
                print("Start prediction...!")
                self.prediction_utils = PredictionUtils(self)
                self.worker = self.predict()
                self.worker.finished.connect(self.end_prediction)
                self.worker.start()
                self.widget.predict_button.text = "*** prediction running ***"
            else:
                return

        @self.widget.model_dir.changed.connect
        def model_dir_change(model_dir):
            if isinstance(model_dir, type(None)):
                print(f"model_dir, None: {model_dir=}")
                return

            # check that model directory has thresholds file
            model_dir = Path(model_dir)
            if not (model_dir / "thresholds.json").exists():
                print("check model")
                self.widget.model_dir.value = self.data.model_dir
                return
            else:
                model_name = model_dir.parts[-1]
                print(f"model_dir, model found: {model_name}")
                self.model_dir = model_dir

                with open(model_dir / "thresholds.json") as f:
                    thresholds = json.load(f)
                    self.threshold = thresholds["prob"]
                    self.widget.threshold.value = self.threshold

        @self.widget.load_prediction_button.changed.connect
        def load_prediction():
            """
            load prediction file
            """
            # check if prediction file exists
            img_dir_name = Path(self.data.img_dir).stem
            img_file = self.data.img_file
            model_dir = Path(self.model_dir)
            model_name = model_dir.name

            if (isinstance(img_dir_name, type(None))) or (isinstance(img_file, type(None))):
                print("set image dir/file")
                return

            # make full path
            prediction_file = (Path(self.data.annotation_dir) / "SD_data" / "predictions" / img_dir_name /
                               f"{img_file.replace('.h5', '')}_prediction_{model_name}.h5")

            print(f"load prediction: {prediction_file=}")

            # check that prediction channel matches annotation channel
            with h5py.File(prediction_file, 'r') as f:
                if np.all((self.data.fg_channel in f.keys(),
                           self.data.bg_channel in f.keys())):
                    pass
                else:
                    print("annotation and prediction channel do not match")
                    return

            spot_table_file = (Path(self.data.annotation_dir) / "SD_data" / "predictions" / img_dir_name /
                               f"{img_file.replace('.h5', '')}_spot_table_{model_name}.csv")

            if isinstance(prediction_file, type(None)):
                print("set image file")
                return

            if not np.all([f.exists() for f in [prediction_file, spot_table_file]]):
                print("no prediction data found")
                return

            # load prediction
            spot_table = pd.read_csv(spot_table_file)

            # add prediction to "predicted" annotation_data
            # get known "predicted" annotation data (from other images)
            df_old = self.data.annotation_data["predicted"].copy()
            df_old = df_old.loc[(df_old["img_dir"] != self.data.img_dir)
                                | ((df_old["img_dir"] == self.data.img_dir)
                                   & (df_old["img_file"] != self.data.img_file))]

            # calculate tile positions, prepare spots dataframes
            spots_df = self.data.make_spots_df()
            if len(spot_table) > 0:
                spot_pos = list(spot_table[['z', 'y', 'x']].to_numpy())
                tile_pos, tile_slices, tile_ids, ul_lr = self.data.cursor_pos_to_tile(cursor_pos=spot_pos)
                spots_df["spot_pos"] = [x for x in spot_pos]
                spots_df["img_dir"] = self.data.img_dir
                spots_df["img_file"] = self.data.img_file
                spots_df["img_slice"] = pprint.pformat(self.data.img_slice)
                spots_df["tile_id"] = tile_ids
                spots_df["tile_slice"] = [pprint.pformat(x) for x in tile_slices]
                spots_df["active_tile"] = -1
                spots_df["fg_channel"] = self.data.fg_channel
                spots_df["bg_channel"] = self.data.bg_channel
                spots_df["normalization_type"] = self.data.normalization_type
                for idx in spots_df.index:
                    spots_df.at[idx, "normalization_range"] = self.data.normalization_range
                spots_df["timestamp"] = pd.Timestamp("now")

            if (len(df_old) > 0) & (len(spots_df) > 0):
                self.data.annotation_data["predicted"] = pd.concat([df_old, spots_df], sort=False, ignore_index=True)
            elif (len(df_old) > 0) & (len(spots_df) == 0):
                self.data.annotation_data["predicted"] = df_old
            else:
                self.data.annotation_data["predicted"] = spots_df

            for idx, spot_type, editable in zip(range(1, 3), ["edited", "predicted"], [True, False]):
                name = spot_type
                df_ = spot_table
                if len(df_) == 0:
                    spot_pos = np.zeros((1, 3))
                else:
                    # prepare points layer
                    spot_pos = spot_table.loc[:, ["z", "y", "x"]].to_numpy()
                    if spot_pos.shape[0] == 0:
                        spot_pos = np.zeros((1, 3))
                    else:
                        spot_pos = list(spot_pos)
                # replace existing spot layers
                if np.any([layer.name == name for layer in self.viewer.layers]):
                    self.viewer.layers[name].data = spot_pos
                else:
                    self.viewer.add_points(
                        data=spot_pos,
                        name=name,
                        ndim=3,
                        n_dimensional=True,
                        symbol="diamond",
                        size=6,
                        edge_width=0.1,
                        edge_color=self.data.edge_colors[idx],
                        face_color=self.data.face_colors[idx],
                        blending="additive",
                        opacity=0.8,
                        visible=True,
                    )
                self.viewer.layers[name].editable = editable

        @self.widget.threshold.changed.connect
        def threshold_change(threshold):
            self.threshold = threshold

    def _on_init(widget):
        # initial settings
        path = Path(__file__).parent
        widget.native.setStyleSheet(open(path / "resources" / "styles.qss", 'r').read())
        widget.model_dir.value = _data.default_model_dir

    @magic_factory(
        widget_init=_on_init,
        model_dir=dict(widget_type="FileEdit", visible=True,
                       label="model dir", mode="d",
                       value=None,
                       ),
        predict_button=dict(widget_type="PushButton", text="Predict"),
        load_prediction_button=dict(widget_type="PushButton", text="Load prediction"),
        threshold=dict(widget_type="FloatSlider", min=0.01, max=1.0, step=0.05, value=0.5, visible=True),
        layout="vertical",
        persist=False,
        call_button=False,
    )
    def _make_widget(viewer: Viewer,
                     model_dir: Path,
                     threshold: float,
                     predict_button,
                     load_prediction_button,
                     ) -> None:
        ...

    @thread_worker
    def predict(self):
        self.prediction_utils.predict()

    def end_prediction(self):
        self.worker.quit()
        self.widget.predict_button.text = "Predict"
