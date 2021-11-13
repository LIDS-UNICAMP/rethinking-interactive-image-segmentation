from PySide2.QtCore import Qt, Slot, Signal, QPointF, QPoint
from PySide2.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsOpacityEffect)
from PySide2.QtGui import (QImage, QPixmap, QMouseEvent, QWheelEvent, QPen, QBrush, QColor, )


class ImageView(QGraphicsView):
    markers_updated = Signal(tuple)  # (positive, negative)
    position_clicked = Signal(tuple)  # (y, x)
    component_clicked = Signal(tuple)  # (y, x)
    label_position_clicked = Signal(tuple)  # (y, x)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTransformationAnchor(self.NoAnchor)
        self.setResizeAnchor(self.NoAnchor)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image = QImage()
        self.image_pixmap = self.scene.addPixmap(QPixmap())
        self.label_pixmap = self.scene.addPixmap(QPixmap())
        self.contour_pixmap = self.scene.addPixmap(QPixmap())
        self.component_pixmap = self.scene.addPixmap(QPixmap())
        self.proposal_pixmap = self.scene.addPixmap(QPixmap())

        self.label_opacity = QGraphicsOpacityEffect()
        self.label_pixmap.setGraphicsEffect(self.label_opacity)

        self.view_scale = 1.0

        self.radii = QPointF(3.0, 3.0)

        self.positives_markers = []
        self.negatives_markers = []
        self.graphic_markers = []

    def addImage(self, image: QImage) -> None:
        self.clearScene()
        self.image = image.convertToFormat(QImage.Format_ARGB32)
        self.image_pixmap.setPixmap(QPixmap.fromImage(self.image))
        self.fitInView(self.image.rect(), Qt.KeepAspectRatio)
        self.view_scale = 1.0

    def appendComponent(self, component: QImage) -> None:
        self.component_pixmap.setPixmap(QPixmap.fromImage(component))

    def appendContour(self, contour: QImage) -> None:
        self.contour_pixmap.setPixmap(QPixmap.fromImage(contour))

    def appendProposal(self, contour: QImage) -> None:
        self.proposal_pixmap.setPixmap(QPixmap.fromImage(contour))

    def appendLabel(self, label: QImage) -> None:
        self.label_pixmap.setPixmap(QPixmap.fromImage(label))

    def clearScene(self) -> None:
        self.clearMarkers()
        self.image_pixmap.setPixmap(QPixmap())
        self.label_pixmap.setPixmap(QPixmap())
        self.contour_pixmap.setPixmap(QPixmap())
        self.component_pixmap.setPixmap(QPixmap())
        self.proposal_pixmap.setPixmap(QPixmap())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        qpos = self.mapToScene(event.pos()).toPoint()
        if self.component_pixmap.contains(self.component_pixmap.mapFromScene(qpos)):
            pos = (qpos.y(), qpos.x())
            if event.button() is Qt.LeftButton:
                self.positives_markers.append(pos)
                self.drawCircle(qpos, Qt.cyan)
            elif event.button() is Qt.RightButton:
                self.negatives_markers.append(pos)
                self.drawCircle(qpos, Qt.yellow)
            #  emit update
            if event.button() is Qt.LeftButton or event.button() is Qt.RightButton:
                self.markers_updated.emit((self.positives_markers, self.negatives_markers))

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        qpos = self.mapToScene(event.pos()).toPoint()
        pos = (qpos.y(), qpos.x())
        if event.button() is Qt.LeftButton:
            # avoid resetting if component does not changeF
            if not self.component_pixmap.contains(self.component_pixmap.mapFromScene(qpos)):
                self.clearMarkers()
            self.component_clicked.emit(pos)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() == Qt.CTRL:
            old_pos = self.mapToScene(event.pos())
            in_factor = 1.05
            out_factor = 0.95
            factor = 1.0
            # y_delta = event.pixelDelta().y()  # macOS only
            y_delta = event.angleDelta().y()
            if y_delta > 0:
                factor = in_factor
            elif y_delta < 0:
                factor = out_factor
            self.view_scale *= factor
            self.scale(factor, factor)
            new_pos = self.mapToScene(event.pos())
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
        else:
            scroll_scale = 1
            # shift = event.pixelDelta()  # macOS only
            shift = event.angleDelta()
            self.translate(scroll_scale * shift.x(),
                           scroll_scale * shift.y())

    @Slot()
    def clearMarkers(self) -> None:
        self.proposal_pixmap.setPixmap(QPixmap())
        for item in self.graphic_markers:
            self.scene.removeItem(item)
        self.positives_markers.clear()
        self.negatives_markers.clear()
        self.graphic_markers.clear()

    def drawCircle(self, pos: QPoint, color: QColor) -> None:
        pen = QPen(color)
        brush = QBrush(color)
        item = self.scene.addEllipse(0, 0, 2 * self.radii.x(),
                                 2 * self.radii.y(), pen, brush)
        item.setPos(QPointF(pos) - self.radii)
        self.graphic_markers.append(item)
