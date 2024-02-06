"""
napari spot finder code

cs15jan2024
"""

import os
import platform
import numpy as np

from superqt import QCollapsible
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout, QSizePolicy
from qtpy.QtWidgets import QScrollArea
from qtpy.QtWidgets import QWidget

from napari_spofi._data_handler import DataHandler
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

        # define widget layout
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # add widgets
        for w, w_name in zip((Annotation, Training, Prediction), ("annotation", "training", "prediction")):
            scroll = QScrollArea()
            w_native = w(self).widget.native
            w_native.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
            setattr(self.data, f"{w_name}_widget", w_native)
            scroll.setWidget(w_native)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
            collapsible = QCollapsible(w_name, self)
            collapsible.addWidget(scroll)
            self.layout().addWidget(collapsible)

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
            opacity=0.1,
            visible=False,
        )

    def add_tiles_layer(self):
        # add tiles layer (indicates tiles in annotation list)
        self.viewer.add_image(
            data=[np.zeros(self.data.img_shape[1:], dtype=np.uint8)],
            name="tiles",
            colormap="yellow",
            blending="additive",
            opacity=0.6,
            gamma=0.2,
            visible=True,
        )

    def viewer_double_click_cb(self, viewer, evnt):
        """
        viewer double click callback function (only active in 'tiles' layer)
        """
        layer = self.viewer.layers.selection.active
        if not layer.name == "tiles":
            return

        # convert mouse position to tile position, calculate rectangle edges to display tile
        self.data.check_tile(self.viewer.cursor.position)
