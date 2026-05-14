"""
napari spofi code

cs2024
"""

import os
import platform
import numpy as np

from superqt import QCollapsible
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QScrollArea
from qtpy.QtWidgets import QWidget

from napari_spofi._data_handler import DataHandler
from napari_spofi._header import Header
from napari_spofi._annotation import Annotation
from napari_spofi._training import Training
from napari_spofi._prediction import Prediction


# %% define SpotFinderWidget class
class SpotFinderWidget(QWidget):
    def __init__(self, napari_viewer: object):
        super().__init__()

        # add viewer variable
        self.viewer = napari_viewer

        # instantiate SpotFinderData
        self.data = DataHandler(viewer=self.viewer, defaults_file=None)

        # startup, get username and computer name
        self.data.username = os.getlogin()
        self.data.node = platform.node()
        print(f"\tHello {self.data.username} at {self.data.node}!")
        print(f"\ttime: {self.data.start_time}")

        # set viewer mouse double click callback, will add/remove tile from list of considered tiles
        self.viewer.mouse_double_click_callbacks.append(self.viewer_double_click_cb)

        # add widgets
        if self.data.show_extra_widget == "YES":
            widgets = [Header, Annotation, Training, Prediction]
            widget_names = ["header", "annotation", "training", "prediction"]
        else:
            widgets = [Annotation, Training, Prediction]
            widget_names = ["annotation", "training", "prediction"]

        # Create layout and widgets
        widget = QWidget()
        layout = QVBoxLayout()
        collapsibles = []
        for w, w_name in zip(widgets, widget_names):
            w_native = w(self).widget.native
            w_native.layout().setContentsMargins(0, 0, 0, 0)
            c = QCollapsible(w_name)
            c.addWidget(w_native)
            layout.addWidget(c)

            setattr(self.data, f"{w_name}_widget", w_native)
            collapsibles.append(c)
        layout.addStretch()

        for c in collapsibles:
            c.layout().setSpacing(0)
            c.layout().setContentsMargins(0, 0, 0, 0)

        # prepare scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidget(widget)

        widget.setLayout(layout)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

        # add image layers
        self.add_checkerboard()
        self.add_tiles_layer()

        print("\tStart processing ...\n")

    def add_checkerboard(self):
        # add checkerboard
        self.viewer.add_image(
            data=[self.data.make_checkerboard()],
            name="checkerboard",
            colormap="gray",
            blending="additive",
            interpolation2d="nearest",
            interpolation3d="nearest",
            opacity=0.1,
            visible=False,
        )

    def add_tiles_layer(self):
        # add tiles layer (indicates tiles in annotation list)

        self.viewer.add_image(
            data=[np.zeros(self.data.img_shape[1:], dtype=np.uint8)],
            name="tiles",
            colormap=self.make_colormap(),
            blending="additive",
            interpolation2d="nearest",
            interpolation3d="nearest",
            opacity=0.8,
            gamma=1.0,
            visible=True,
        )
        self.viewer.layers["tiles"].editable = False

    def viewer_double_click_cb(self, viewer, evnt):
        """
        viewer double click callback function (only active in 'tiles' layer)
        """
        layer = self.viewer.layers.selection.active
        if layer.name == "tiles":
            # convert mouse position to tile position, calculate rectangle edges to display tile
            self.data.check_tile(self.viewer.cursor.position)
        elif layer.name == "predicted":
            # add predicted spot positions to true spots
            self.data.add_predicted_spots_to_tile(self.viewer.cursor.position)
        else:
            return

    @staticmethod
    def make_colormap():
        # light yellow, yellow, light blue, blue
        colors = [[0, 0, 0, 0],
                  [1., 1., 0., 0.3],
                  [1., 1., 0, 1.0],
                  [0., 0.25, 1., 0.3],
                  [0., 0.25, 1., 1.0]]
        colormap = {
            'colors': colors,
            'name': 'tiles_color',
            'interpolation': 'zero'
        }
        return colormap
