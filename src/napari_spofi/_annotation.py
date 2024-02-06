import ast
import warnings
import h5py
from typing import List
from pathlib import Path
from magicgui import magic_factory
from napari import Viewer
from qtpy.QtWidgets import QWidget, QSizePolicy


class Annotation(QWidget):
    """
    annotate images in widget
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.viewer = parent.viewer
        self.data = parent.data
        global _data
        _data = self.data
        self.widget = self._make_widget()

        @self.widget.annotation_dir.changed.connect
        def annotation_dir_change(annotation_dir):
            self.set_button_state(button=self.widget.load_img_button.native, state="not ok")
            self.data.set_new_args()
            self.data.new_args["annotation_dir"] = annotation_dir.as_posix()

            # create new annotation data filename
            self.data.new_args["annotation_file"] = f"{annotation_dir.as_posix()}/annotation_data_{self.data.start_time}.pickle"

            # check for existing annotation files
            annotation_files = sorted([f.stem for f in annotation_dir.glob(("annotation_data*" + ".pickle"))], reverse=True)

            if len(annotation_files) == 0:
                # start new annotation
                print(f"annotation dir change, start new annotation: {annotation_dir=}")
                self.widget.img_dir.value = self.data.new_args["img_dir"]
                self.widget.img_files.choices = [None, ]
                self.widget.img_files.value = None
            else:
                print(f"annotation dir change, load annotation: {annotation_dir=}, {annotation_files[0]=}")
                # load most recent existing annotation data (checks for normalization_type)
                # suggest first entry from listed img dirs
                self.data.read_annotation_data(annotation_dir=annotation_dir.as_posix(),
                                               annotation_file=(annotation_files[0] + ".pickle"),
                                               )
                if isinstance(self.data.new_args["img_file"], type(None)):
                    self.widget.img_dir.value = self.data.new_args["img_dir"]
                    self.widget.img_files.choices = [None, ]
                    self.widget.img_files.value = None
                else:
                    self.widget.img_dir.value = self.data.new_args["img_dir"]
                    self.widget.img_files.choices = self.data.new_args["img_files"]
                    self.widget.img_files.value = self.data.new_args["img_file"]

            self.widget.fg_channels.choices = [self.data.new_args["fg_channel"], ]
            self.widget.fg_channels.value = self.data.new_args["fg_channel"]
            self.widget.bg_channels.choices = [self.data.new_args["bg_channel"], ]
            self.widget.bg_channels.value = self.data.new_args["bg_channel"]

        @self.widget.img_dir.changed.connect
        def img_dir_change(img_dir):
            if img_dir.name == "None":
                return
            self.set_button_state(button=self.widget.load_img_button.native, state="not ok")
            self.data.new_args["img_dir"] = img_dir.as_posix()
            # get image directory, screen image directory for h5 type files
            available_img_files = sorted([x.name for x in img_dir.glob(("*" + ".h5"))])
            # filter available h5 files, get names matching typical ScanImage output names
            # filenames = [f for f in filenames if re.search(r"(.*\d{5}\.h5)", f)]
            self.data.new_args["img_files"] = available_img_files

            # check for existing annotation files
            annotation_dir = self.widget.annotation_dir.value
            annotation_files = sorted([f.stem for f in annotation_dir.glob(("annotation_data*" + ".pickle"))], reverse=True)
            used_img_files = []
            if len(annotation_files):
                used_img_files = self.data.read_annotation_img_files(annotation_dir=annotation_dir.as_posix(),
                                                                     annotation_file=(annotation_files[0] + ".pickle"),
                                                                     img_dir=img_dir.as_posix())
                used_img_files = sorted(used_img_files)

            self.widget.img_files.choices = available_img_files
            if len(used_img_files):
                self.widget.img_files.value = used_img_files[0]
                self.data.new_args["img_file"] = used_img_files[0]
                used_img_files_ = "\n".join([str(f) for f in used_img_files])
                self.widget.img_files.tooltip = f"img files in annotation data:\n{used_img_files_}"
            else:
                self.widget.img_files.tooltip = "no img file in annotation data!"
                self.widget.img_files.value = available_img_files[0]
                self.data.new_args["img_file"] = available_img_files[0]

        @self.widget.img_files.changed.connect
        def img_file_change(img_file):
            if isinstance(img_file, type(None)):
                return
            if img_file == "None":
                return
            self.set_button_state(button=self.widget.load_img_button.native, state="not ok")
            self.data.new_args["img_file"] = img_file

            # check for existing annotation files
            annotation_dir = self.widget.annotation_dir.value
            annotation_files = sorted([f.stem for f in annotation_dir.glob(("annotation_data*" + ".pickle"))], reverse=True)
            used_img_files = []
            if len(annotation_files):
                used_img_files = self.data.read_annotation_img_files(annotation_dir=annotation_dir.as_posix(),
                                                                     annotation_file=(annotation_files[0] + ".pickle"),
                                                                     img_dir=self.widget.img_dir.value.as_posix())
            if len(used_img_files):
                used_img_files = sorted(used_img_files)
                used_img_files_ = "\n".join([str(f) for f in used_img_files])
                self.widget.img_files.tooltip = f"img files in annotation data:\n{used_img_files_}"
            else:
                self.widget.img_files.tooltip = "no img file in annotation data!"

            # check h5 data, check fg_channels and bg_channels
            with h5py.File(Path(self.widget.img_dir.value) / img_file, "r", ) as f:
                channels = [c for c in f.keys()]
            available_fg_channels = sorted([channel for channel in channels if channel.startswith(self.data.default_fg_channel)])
            available_bg_channels = sorted([channel for channel in channels if channel.startswith(self.data.default_bg_channel)])

            if self.data.new_args["fg_channel"] in available_fg_channels:
                fg_channel = self.data.new_args["fg_channel"]
                bg_channel = self.data.new_args["bg_channel"]
            else:
                fg_channel = available_fg_channels[0]
                bg_channel = available_bg_channels[0]
                self.data.new_args["fg_channel"] = fg_channel
                self.data.new_args["bg_channel"] = bg_channel
            self.data.new_args["fg_channels"] = available_fg_channels
            self.data.new_args["bg_channels"] = available_bg_channels

            self.widget.fg_channels.choices = available_fg_channels
            self.widget.bg_channels.choices = available_bg_channels
            self.widget.fg_channels.value = fg_channel
            self.widget.bg_channels.value = bg_channel

        @self.widget.fg_channels.changed.connect
        def fg_channel_change(fg_channel):
            self.data.new_args["fg_channel"] = fg_channel
            self.set_button_state(button=self.widget.load_img_button.native, state="not ok")

        @self.widget.bg_channels.changed.connect
        def bg_channel_change(bg_channel):
            self.data.new_args["bg_channel"] = bg_channel
            self.set_button_state(button=self.widget.load_img_button.native, state="not ok")

        @self.widget.spot_radius.changed.connect
        def spot_radius_event(event):
            try:
                spot_radius = ast.literal_eval(event)
                self.set_button_state(button=self.widget.load_img_button.native, state="not ok")
            except SyntaxError:
                return
            self.data.spot_radius = spot_radius

        @self.widget.load_img_button.changed.connect
        def load_img():
            """
            load selected channels from image file, apply smoothing (just for display)
            """
            # check for annotation dir
            if self.widget.annotation_dir.value == self.data.default_annotation_dir:
                warnings.warn("oops! set annotation dir", UserWarning)
                return
            # check fg & bg channels
            fg = self.widget.fg_channels.value
            bg = self.widget.bg_channels.value
            if not fg.split('_')[1:] == bg.split('_')[1:]:
                warnings.warn("oops! channels not matching", UserWarning)
                return

            # load channels, apply filter
            self.data.load_img()
            self.widget.raw_img_shape.value = f"{self.data.raw_img_shape}"
            self.set_button_state(button=self.widget.load_img_button.native, state="ok")

        @self.widget.extract_spots_button.changed.connect
        def extract_spots():
            """
            extract spots and associated data
            add spots from current image to true spots layer
            """
            self.data.extract_spots()

    def _on_init(widget):
        # initial settings
        path = Path(__file__).parent
        widget.native.setStyleSheet(open(path / "resources" / "styles.qss", 'r').read())
        widget.raw_img_shape.native.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        widget.spot_radius.native.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        widget.annotation_dir.value = _data.annotation_dir
        widget.annotation_dir.native.setToolTip("chose annotation directory")
        widget.img_dir.value = _data.img_dir
        widget.img_dir.native.setToolTip("chose image directory")
        widget.fg_channels.choices = [_data.default_fg_channel, ]
        widget.fg_channels.value = _data.default_fg_channel
        widget.bg_channels.choices = [_data.default_bg_channel, ]
        widget.bg_channels.value = _data.default_bg_channel
        widget.spot_radius.value = _data.spot_radius
        widget.spot_radius.native.setToolTip("set characteristic spot size (zyx, px), run 'extract spots' if changed!")

    @magic_factory(
        widget_init=_on_init,
        annotation_dir=dict(widget_type="FileEdit", visible=True,
                            label="annotation dir", mode="d",
                            value=None,
                            ),
        img_dir=dict(widget_type="FileEdit", visible=True,
                     label="image dir", mode="d",
                     value=None,
                     ),
        img_files=dict(label="image file", choices=[None, ],
                       value=None, nullable=False,
                       ),
        fg_channels=dict(widget_type="ComboBox", label="fg channel",
                         choices=[None, ],
                         value=None, nullable=False,
                         ),
        bg_channels=dict(widget_type="ComboBox", label="bg channel",
                         choices=[None, ],
                         value=None, nullable=False,
                         ),
        spot_radius=dict(widget_type="LiteralEvalLineEdit", label="spot radius",
                         value=None, nullable=False,
                         ),
        raw_img_shape=dict(widget_type="Label", label="image shape", value="(0, 0, 0)"),
        load_img_button=dict(widget_type="PushButton", text="load image"),
        extract_spots_button=dict(widget_type="PushButton", text="extract spots"),
        layout="vertical",
        persist=False,
        call_button=False,
    )
    def _make_widget(self,
                     viewer: Viewer,
                     annotation_dir: Path,
                     img_dir: Path,
                     img_files: List,
                     fg_channels,
                     bg_channels,
                     spot_radius,
                     raw_img_shape,
                     load_img_button,
                     extract_spots_button,
                     ) -> None:
        ...

    @staticmethod
    def set_button_state(button=None, state="ok"):
        if state == "not ok":
            button.setStyleSheet(
                "background-color: qlineargradient(x1: 1, y1: 1, x2: 0, y2: 0, stop: 0 green, stop: .1 red);")
        if state == "ok":
            button.setStyleSheet(
                "background-color: qlineargradient(x1: 1, y1: 1, x2: 0, y2: 0, stop: 0 red, stop: .1 green);")
