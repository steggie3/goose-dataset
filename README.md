# Goose Dataset

This is a dataset of 1,000 Canada goose images. They are all *mugshots*, with the head being the main feature of the image. Each image comes with an annotation XML in the PASCAL VOC format with a bounding box of the head and the object class *goose-head*. 

## About the Dataset

Canada geese have a distinct and high-contrast facial pattern that is highly recognizable. The images in this dataset are all somewhat visually similar. The head in the image is either the left side or the right side, not front or back. The background is usually blurry. Only one goose head is present in each image. These properties makes the dataset rather *simple* for various computer vision tasks, and can serve as a starting point for computer vision experiments such as object recognition, localization, detection, etc.

All images are photographed by me.

## Usage

The *images* directory contains 1,000 JPG images of the size 800 x 533. The *annotations* directory contains the XML annotations. These can be directly used in the [darkflow](https://github.com/thtrieu/darkflow) implementation of object detection by placing them under darkflow/test/training. 

If you are only using the images and not the annotations, I also implemented a *load_data()* API that has compatible usage as those in the [Keras built-in datasets](https://keras.io/datasets/). 

```python
(x_train, y_train), (x_test, y_test) = GooseDataset.load_data(test_ratio=0.2,
                                                              grayscale=False,
                                                              cropped=False,
                                                              resize_shape=None,
                                                              save=False)
```
- **Returns**:
  - 2 tuples:
    + **(x_train, x_test)**: Arrays of integers with shape (n_samples, n_rows, n_columns, 3) in the range [0, 255] for RGB, or arrays of floats with shape (n_samples, n_rows, n_columns, 1) in the range [0, 1] for grayscale.
    + **(y_train, y_test)**: Arrays of integers with shape (n_samples, 1). All values are 1, which stands for presence of goose in the picture.

- **Arguments**:
  - **test_ratio**: Float. Ratio of (number of testing data) / (number of all data).
  - **grayscale**: Boolean. Whether to load the images in grayscale.
  - **cropped**: Boolean. Whether to load only the cropped bounding boxes of the images. If *True*, a new directory *cropped_images* will be created in addition to loading the images.
  - **resize_shape**: [n_rows, n_columns], target shape of the loaded images. If *None*, the default shape [533, 800] will be used.
  - **save**: Boolean. Whether to save the images as how they are loaded into a new directory *processed_images* in addition to loading. This can be used to visualize the actual images being loaded after grayscaling, cropping, and/or resizing.

To see an actual example, please check [the Example Jupyter notebook](https://github.com/steggie3/goose-dataset/blob/master/Example.ipynb).

## Note

Please do not use this dataset to derive any work that harms animals.
