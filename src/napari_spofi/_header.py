from magicgui import magic_factory
from napari import Viewer
from qtpy.QtWidgets import QWidget, QSizePolicy
from pathlib import Path
import time

from napari.qt.threading import thread_worker


class Header(QWidget):
    """header widget"""

    def __init__(self, parent):
        super().__init__(parent)
        self.viewer = parent.viewer
        self.data = parent.data
        global _data
        _data = self.data
        self.widget = self._make_widget()

        @self.widget.the_button.clicked.connect
        def the_btn_fcn():
            self.data.losses = ":::"  # {"training": [], "validation": []}
            self.worker = self.test_fcn()
            self.worker.finished.connect(self.end_thread)
            self.worker.start()
            self.widget.the_button.text = "***"
            self.widget.native.setEnabled(False)

    def _on_init(widget):
        # initial settings
        widget.header_label.native.setObjectName("header_label")
        widget.header_label.native.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        path = Path(__file__).parent
        widget.header_label.native.setStyleSheet(open(path / "resources" / "styles.qss", 'r').read())

    @magic_factory(
        widget_init=_on_init,
        header_label=dict(widget_type="Label", value="special edition", label="***"),
        the_button=dict(widget_type="PushButton", text="???"),
        info=dict(widget_type="LineEdit", label="info", value="..."),
        layout="vertical",
        persist=False,
        call_button="show data",
    )
    def _make_widget(viewer: Viewer,
                     header_label,
                     the_button,
                     info: str,
                     ) -> None:
        print("\n\nshow data")
        # print(dir(_data))
        for arg in _data.new_args.keys():
            print(f"{arg}: {_data.new_args[arg]}, {getattr(_data, arg)}")
        print("show data\n\n")

    @thread_worker
    def test_fcn(self):
        time.sleep(0.3)

    def end_thread(self):
        print("test fcn finished")
        self.worker.quit()
        self.widget.the_button.text = "???"
        self.widget.native.setEnabled(True)
