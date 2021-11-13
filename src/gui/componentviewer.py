from __future__ import annotations

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QLabel
from PySide2.QtGui import QPixmap, QPainter

from gui.utils import component_to_qimage

from loaders.maindata import Component

class ComponentViewer(QLabel):
    def __init__(self, comp: Component, text: str=None, parent=None, f=Qt.Window) -> ComponentViewer:
        super().__init__(parent=parent, f=f)

        self.pixmap = QPixmap.fromImage(component_to_qimage(comp))
        if text is not None:
            painter = QPainter(self.pixmap)
            painter.setPen(Qt.red)
            painter.drawText(20, 20, text)
            
        self.setPixmap(self.pixmap)
        painter.end()
