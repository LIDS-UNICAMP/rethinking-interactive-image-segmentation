from __future__ import annotations
from PySide2.QtCore import Qt, Slot, Signal, QObject

from PySide2.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame,
                                QComboBox, QPushButton, )
from PySide2.QtGui import QCloseEvent, QColor

from gui.projectionview import ProjectionView


class SubProjectionDone(QObject):
    signal = Signal(list)


class PenColorSignal(QObject):
    signal = Signal(QColor)


class ProjSubsetFrame(QWidget):
    def __init__(self, labels_box: QComboBox, parent=None, f=Qt.Window):
        super().__init__(parent=parent, f=f)
        self.resize(800, 600)
        self.color_table = {'void': QColor(0, 0, 0, 0)}

        self.setupUI(labels_box)
        self.setupSignals()

        self.emitPenColor(self.label_box.currentText())

    def setupUI(self, labels_box: QComboBox) -> None:
        self.view = ProjectionView(self, scale=250)
        frame = QFrame(self)

        self.label_box = QComboBox(frame)
        self.label_box.setEditable(False)

        for i in range(labels_box.count()):
            icon = labels_box.itemIcon(i)
            text = labels_box.itemText(i)
            self.label_box.addItem(icon, text)
            color = icon.pixmap(1, 1).toImage().pixelColor(0, 0)
            if text != 'void':
                self.color_table[text] = color
        
        self.label_confirm = QPushButton(self.tr("Confirm"), frame)
        self.label_cancel = QPushButton(self.tr("Cancel"), frame)

        v_layout = QVBoxLayout(self)
        v_layout.addWidget(self.view)
        v_layout.addWidget(frame)

        h_layout = QHBoxLayout(frame)
        h_layout.addWidget(self.label_box)
        h_layout.addWidget(self.label_confirm)
        h_layout.addWidget(self.label_cancel)

    def setupSignals(self) -> None:
        # pen color
        self.pen_color_changed = PenColorSignal(self)
        self.pen_color_changed.signal.connect(self.view.setPenColor)
        self.label_box.currentTextChanged.connect(self.emitPenColor)
        self.label_box.currentTextChanged.connect(self.view.resetSelection)

        # proj view
        self.view.item_clicked_left.signal.connect(self.parent().setCompDisplay)
        self.view.item_clicked_right.signal.connect(self.parent().queryIndex)

        # labels
        self.label_cancel.clicked.connect(self.view.resetSelection)
        self.label_confirm.clicked.connect(self.view.confirmSelection)

        main_proj_view = self.parent().proj_frame.view
        self.sub_proj_done = SubProjectionDone(self)
        self.sub_proj_done.signal.connect(main_proj_view.updateColor)
    
    @Slot()
    def emitPenColor(self, text: str) -> None:
        color = self.color_table[text]
        self.pen_color_changed.signal.emit(color)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.view.resetSelection()
        components = [comp for comp in self.view.graphitem_to_comp.values()]
        self.sub_proj_done.signal.emit(components)
        self.parent().bbox_count += self.view.bbox_count

