from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QVBoxLayout, QTextEdit


MARKDOWN_TEXT = \
"""
# Projection

Projection view commands.

## UI Widgets
- `Current Label`: Dropdown box to select and add labels.
- `Labeled Invisible`: Checkbox to hide or not the annotated labels.
- `Select Subset`: Checkbox to zoom in selection mode.

## Mouse Commands
- `Left Double Click` on component: Selects in image view.
- `Left Drag` on canvas: Selects and label components.
- `Right Double Click` on component: Nearest neighbor query.
- `Right Drag` on component: Query and label propagation according to distance moved.
- `Wheel`: Scroll up/down.
- `CTRL + Wheel`: Zoom in/out.

## Keyboard Shortcuts
- `W`: Increases label opacity in image window.
- `S`: Decreases label opacity in image window.
- `A`: Move to previous image in the list.
- `D`: Move to next image in the list.
- `E`: Iterate over components with highest class similarity entropy in image view.

# Image 

Image domain view commands.

## UI Widgets
- `WS Contour Filter`: Watershed hierarchy contour filtering threshold.
- `WS Volume Threshold`: Watershed hierarchy volume attributes threshold.
- `Update`: Generates new watershed cut in **green**.
- `Confirm`: Accept new watershed segmentation.

## Mouse Commands
- `Left Click`: Inserts **left click label** marker on selected component.
- `Left Double Click`: Selects another component, focus in it projection view.
- `Right Click`: Inserts **right click label** marker on selected component.
- `Right Double Click`: Assigns clicked component with current projection view label.

## Keyboard Shortcuts
- `Space`: Confirm interactive segmentation.
"""


class CommandsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.layout = QVBoxLayout(self)

        self.setWindowTitle("UI Commands")
        self.setWindowModality(Qt.NonModal)
        self.setLayout(self.layout)

        self.textEditBox = QTextEdit(self)
        self.textEditBox.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.textEditBox.setMinimumSize(600, 800)
        try:
            # this might is not available in every python qt version
            self.textEditBox.setMarkdown(MARKDOWN_TEXT)
        except:
            self.textEditBox.setText(MARKDOWN_TEXT)

        self.layout.addWidget(self.textEditBox)
