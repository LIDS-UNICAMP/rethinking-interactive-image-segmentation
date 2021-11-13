# Rethinking Interactive Image Segmentation: Feature Space Annotation

## Installation

- Create your conda/virtual environment and activate it.
- Install PyTorch 1.7.1 with CUDA support if your GPU supports it.
- Install the remaining requirements with `pip install -r requirements`.


## Notes

- The file `config.py` contains app configuration (e.g., dataset path, batch size, cuda usage, etc.).
- When finished an experiment, we recommend pressing both the `Save Masks`, to map the annotation back to their respective image and `Save (Pickle)`, to store the app's current state.

## Demo

Here are the required steps to reproduce a simple demo with the subset from DAVIS as in the original article.

- The DAVIS dataset subset and its edge detection can be downloaded [here](https://drive.google.com/file/d/1-2kXx6KwMe4P_BoD_MeB8ZqUlt322Ig0/view?usp=sharing).
- Pretrained weights from ImageNet can be download [here](https://drive.google.com/file/d/1ObP_3jtTlfxUTtqODR1oVs7N3XW66S6a/view?usp=sharing).
- The `config.py` must be updated accordingly to the dataset paths.
- Run `python src/main.py`.

## Interface Usage Instructions

1. Click `Execute Projection` to start the process;
2. Edit the `Current label` combo box to add the annotation labels;
3. Interact with the canvas to annotate and press `Confirm` (`Cancel`) to confirm (cancel) the current annotation;
4. The `Labeled Invisible` checkbox forces annotated segments to be invisible;
5. The `Select Subset` checkbox enables the behavior of computing UMAP in a subset of data when confirming the selection of a few segments;
6. Ctrl (Command in Mac) + Scroll zoom in and out.
7. Double-clicking a segment in the projection space highlights it in the image space;
8. Clicking with the right and left mouse button on the highlighted segment in the image space add markers to it, and pressing `Split` splits the regions given these cues;
9. The remaining widgets are for recomputing (or previewing) the segmentation of the displayed image;
10. Double-clicking a segment in the image focus to it in the projection canvas;
11. The `Execute Metric Learn` button optimizes the network embedding given the annotated segments, this might take a few minutes without GPU;
12. The spinbox with an integer selects the number of samples to embed, this can be used to request more data;
13. The button `Save Masks` saves the current annotations into images, `background` labels, and unlabeled nodes are saved as 0.
14. The button `Save (Pickle)` stores the experiment's current state into the file indicated in `config.py` (the default is `app.pickle`) and displays some statistics about the recent run.

Additional interface instructions can be accessed through the `Help` file menu on the top left corner.

# Citing

```
TBD
```
