# NN-practice-recognizing-shapes

## Description

This project focuses on object recognition using deep learning techniques. The goal is to train a convolutional neural network (CNN) model to classify images into three categories: triangles, squares, and combinations of triangles and squares. The project includes data generation, model training, and visualization of filter patterns and activation maps.

## Folder Structure

The project repository contains the following folders:

- **data**: This folder contains the dataset used for training, validation, and testing. It includes numpy arrays of image data and corresponding labels.

- **model**: This folder contains the trained model architecture saved as a JSON file. It also includes the weights of the trained model stored in an HDF5 file.

- **weights**: This folder contains pre-trained weights for the model. These weights can be used to initialize the model for further training or for making predictions on new data.

## Dependencies

The following dependencies are required to run the project:

- Python (version 3.6 or higher)
- TensorFlow (version 2.0 or higher)
- Keras (version 2.2 or higher)
- NumPy (version 1.18 or higher)
- Matplotlib (version 3.1 or higher)
- OpenCV (version 4.2 or higher)

To install the dependencies, you can use the following command:

```
pip install -r requirements.txt
```

## Usage

To run the project, you can directly execute the main Python file (`main.py`). At the top of the code, there are boolean variables that define the task, which parts to perform, and the parameters of the model.

The main operations performed by the code include:

- **Data generation**: The script generates synthetic images of triangles, squares, and combinations of both. The generated data is stored in the `data` folder.

- **Model training**: The code trains a CNN model using the generated data. The trained model is saved in the `model` folder.

- **Visualization**: The script generates filter patterns and activation maps for the trained model. The generated visualizations are displayed using Matplotlib.

Please make sure to configure the boolean variables and parameters at the top of the `main.py` file according to your requirements.

## Results

The project provides insights into the performance of the trained model and visualizes the learned features through filter patterns and activation maps. The model's accuracy and loss during training and validation are displayed using Matplotlib plots.

<img width="468" alt="image" src="https://github.com/lironfarzam/NN-practice-recognizing-shapes/assets/87701576/f4f9edfc-efbd-4157-a214-5f2cbbc79994">


## Configuration

The following boolean variables can be adjusted at the top of the `main.py` file:

- **GENERATA_NEW_DATA**: Set to `True` to generate new data or `False` to use existing data.

- **GENERATA_NEW_MODEL**: Set to `True` to train a new model or `False` to use an existing model.

- **SHOW_ACTIVATIONS**: Set to `True` to show filter activations for a few images.

- **NUM_OF_IMAGES_FOR_ACTIVATION**: Number of images to show activations for (should be less than the total number of images).

- **SHOW_MAX_ACTIVATIONS**: Set to `True` to show the images that maximize each filter's activation.

- **SHOW_ACTIVATIONS_MAP**: Set to `True` to show the activation maps for a few images.

- **NUM_OF_IMAGES_FOR_MAP**: Number of images to show activation maps for (should be less than the total number of images).
  

## Contributing

Contributions to the project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull
