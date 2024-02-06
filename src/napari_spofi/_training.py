import numpy as np
import pandas as pd
import shutil
import re
from pathlib import Path
from magicgui import magic_factory
from napari import Viewer
from qtpy.QtWidgets import QWidget, QSizePolicy
from napari.qt.threading import thread_worker
from .spofi_utils.training_utils import TrainingUtils


class Training(QWidget):
    """widget to run training"""

    def __init__(self, parent):
        super().__init__(parent)
        self.viewer = parent.viewer
        self.data = parent.data
        global _data
        _data = self.data
        self.widget = self._make_widget()
        self.retrain_model = False
        self.model_to_retrain = None
        self.worker = []

        @self.widget.start_stop_button.clicked.connect
        def start_stop_training():
            # check if training is running,
            if self.widget.start_stop_button.text == "START":
                # find model directory, load, prepare training, start worker
                print("Start training...!")
                self.get_model_directory()
                self.data.losses = {"training": [], "validation": []}
                self.data.losses_file = self.data.model_dir / "losses.csv"
                self.training_utils = TrainingUtils(self)
                self.worker = self.train()
                self.worker.finished.connect(self.end_training)
                self.worker.start()
                self.widget.start_stop_button.text = "STOP"
            else:
                # call stop_training, stop worker
                print("Got 'STOP' command")
                self.worker.quit()
                self.training_utils.stop_training = True
                print("...training stopped")
                self.widget.start_stop_button.text = "START"

        @self.widget.retrain.changed.connect
        def retrain_evnt(retrain_check):
            # toggles model_dir FileEdit
            if retrain_check:
                self.retrain_model = True
                self.widget.model_names.value = "_ -> _"
                self.widget.model_dir.visible = True
            else:
                self.retrain_model = False
                self.widget.model_names.value = "_ -> _"
                self.widget.model_dir.visible = False

        @self.widget.model_dir.changed.connect
        def model_dir_change(model_dir):
            self.model_to_retrain = Path(model_dir)

        @self.widget.number_of_epochs.changed.connect
        def number_of_epochs(value):
            self.data.number_of_epochs = value

    def _on_init(widget):
        # initial settings
        path = Path(__file__).parent
        widget.native.setStyleSheet(open(path / "resources" / "styles.qss", 'r').read())
        widget.model_names.native.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        widget.model_dir.value = Path(_data.default_model_dir)
        widget.model_dir.native.setToolTip("chose model directory")

    @magic_factory(
        widget_init=_on_init,
        retrain=dict(widget_type="Checkbox", value=False),
        model_dir=dict(widget_type="FileEdit",
                       label="model dir", mode="d",
                       value=None, visible=False,
                       ),
        model_names=dict(widget_type="Label", label="model", value="_ -> _", visible=True),
        number_of_epochs=dict(widget_type="Slider", min=10, max=5e3, step=10, value=100, visible=True),
        start_stop_button=dict(widget_type="PushButton", text="START"),
        losses=dict(widget_type="LineEdit", label="losses", value="epoch loss val_loss"),
        layout="vertical",
        persist=False,
        call_button=False,
    )
    def _make_widget(viewer: Viewer,
                     start_stop_button,
                     model_names,
                     model_dir: Path,
                     losses: str,
                     number_of_epochs: int,
                     retrain: bool = False,
                     ) -> None:
        ...

    def get_model_directory(self):
        # prepare for training, get model directory, screen dir
        if self.retrain_model:
            # retrain given model
            print("retrain given model")
            # check for models subindex
            model_name = self.model_to_retrain.name
            models = sorted(self.model_to_retrain.parent.rglob(f"**\{model_name.split('.')[0]}*"),
                            key=lambda x: x.name)
            models = [x for x in models if x.is_dir()]
            print(f"models found: {models}")
            idx_list = [re.match('(.*)(\.)(\d{1,})', model.name) for model in models]
            idx_list = [idx for idx in idx_list if not isinstance(idx, type(None))]
            new_idx = 1
            if idx_list:
                # check for existing subindex
                new_idx += np.max([int(idx.group(3)) for idx in idx_list if not isinstance(idx, type(None))])
            new_model_name = f"{model_name.split('.')[0]}.{new_idx}"
            print(new_model_name)

            # copy model configs
            from_dir = self.model_to_retrain.parent / model_name
            to_dir = self.model_to_retrain.parent / new_model_name
            self.data.model_dir = to_dir
            shutil.copytree(from_dir, to_dir)
            # delete logs
            for log_dir in (self.data.model_dir / "logs").iterdir():
                for f in log_dir.iterdir():
                    f.unlink()
            self.widget.model_names.value = f"{from_dir.name} -> {to_dir.name}"
        else:
            models = sorted(Path(self.data.sd_models).rglob(f"**\model_*"), key=lambda x: float(x.name.split("_")[1]))
            # create new model
            print("create new model")
            new_idx = 1
            if models:
                most_recent_idx = int(np.floor(sorted([float(x.name.split("_")[1]) for x in models if x.is_dir()])[-1]))
                new_idx += most_recent_idx
            new_model_name = f"model_{new_idx}"
            self.data.model_dir = Path(self.data.sd_models) / new_model_name
            self.data.model_dir.mkdir(exist_ok=True)
            self.widget.model_names.value = f"new model: {new_model_name}"

    @thread_worker
    def train(self):
        self.training_utils.train()

    def end_training(self):
        self.training_utils.optimize_thresholds()
        if not isinstance(self.data.losses, type(None)):
            df = pd.DataFrame.from_dict(self.data.losses)
            df.to_csv(self.data.losses_file, index=False)
        self.worker.quit()
        self.data.model = self.training_utils.model
        self.widget.start_stop_button.text = "START"
