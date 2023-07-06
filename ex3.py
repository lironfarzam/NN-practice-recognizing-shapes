import math
import time
from random import randint, random
from AppKit import NSBeep
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray, load, savez_compressed
from PIL import Image, ImageDraw
from tensorflow.keras import layers, models, optimizers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image as keras_image
from keras import utils

import matplotlib.cm as cm
import cv2

import tensorflow as tf

# ------------------ CONSTANTS FOR MISSING PARTS ------------------
GENERATA_NEW_DATA = False
GENERATA_NEW_MODEL = False
SHOW_ACTIVATIONS = False
NUM_OF_IMAGES_FOR_ACTIVATION = 3
SHOW_MAX_ACTIVATIONS = False
SHOW_ACTIVATIONS_MAP = False
NUM_OF_IMAGES_FOR_MAP = 6
# ------------------ CONSTANTS FOR ACTIVATION MAP -------------------
RUN_ON_GPU = False
MACOS = True
# ------------------ CONSTANTS FOR DATA SAVE/LOAD ------------------
Q = '14'
PATH_TO_DATA = 'data/data_ex3_' + Q +'.npz'
PATH_TO_MODEL = 'model/model_ex3_' + Q +'.json'
PATH_TO_WEIGHTS = 'weights/model_ex3_' + Q +'.h5'
# ------------------ CONSTANTS FOR DATA GENERATION ------------------
# image size is w by w
w = 25
MIN_A = 6
MAX_A = 12
RANDOM_POS = True
RANDOM_DEG = True
DEG = 0

TRAINING_SET = 30000
VALIDATION_SET = 3000
TEST_SET = 3000
# ------------------ CONSTANTS FOR MODEL ----------------------------
EPOCHS = 155
BATCH_SIZE = 55
LEARNING_RATE = 0.00555

# ------------------ FUNCTIONS SECTION ------------------------------

def rotation_matrix(ad):
    # ad = angle in [deg]
    a = math.radians(ad)
    R = asarray([[math.cos(a), -math.sin(a)],
                 [math.sin(a), math.cos(a)]])
    return R


def calculate_overlap_area(poly1, poly2):
    """ 
    Calculate the overlap area of two polygons
    
    Args:
        poly1: List of tuples representing the vertices of the first polygon.
        poly2: List of tuples representing the vertices of the second polygon.

    Returns:
        overlap_area: The overlap area between the two polygons.
    """
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    intersection = polygon1.intersection(polygon2)
    overlap_area = intersection.area
    return overlap_area
    

def rotate(points, origin=(0, 0), degrees=0):
    # https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    R = rotation_matrix(degrees)
    o = np.atleast_2d(origin)
    p = np.atleast_2d(points)
    # CLARIFICATION: @ is the matrix multiplication operator in python
    # CLARIFICATION: .T is the transpose operator in numpy
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def random_equilateral_triangle(w, min_a, max_a,
                                random_pos=False, random_deg=False, deg=0):
    '''
    Calculates the coordinates of the 3 vertices of an equilateral triangle. 
    
    The triangle can have random scale, position and location,
    according to the parameters, with the constraint that the whole
    triangle will be inside a gris of size w x w.
    
    The coordinates can be used to draw a polygon inside a PIL image
    of size w x w.    
    
    Parameters
    ----------
    w : int
        Width of the image to draw into.
    min_a : int
        Minimal edge size of the triangle.
    max_a : int
        Maximal edge size of the triangle.
    random_pos : bool, optional
        Use random position for the triangle or not. The default is False.
    random_deg : bool, optional
        Use random rotation for the triangle or not. The default is False.
    deg : int, optional
        Rotation angle. The default is 0. 
        deg value is used only if random_deg is False. 

    Returns
    -------
    list of tuples of ints/floats
        Coordinates of the vertices of the triangle.


    Examples:
    1. A triangle of edge size 10, centered at the middle of the image,
       with its base parallel to the x-axis:
    
       poly = random_equilateral_triangle(w=w, min_a=10, max_a=10,
                            random_pos=False, random_deg=False, deg=0)
    
    2. A triangle of random edge size between 10 to 15, 
       centered at the middle of the image and rotated to 45 degs:
    
       poly = random_equilateral_triangle(w=w, min_a=10, max_a=15,
                            random_pos=False, random_deg=False, deg=45)
    '''

    # set edge size of the triangle in a
    # confine the values
    if min_a < w //4:
        min_a = w // 4

    if max_a > w //2:
        max_a = w // 2

    if max_a < min_a:
        max_a = min_a

    if min_a != max_a:
        a = randint(min_a,max_a)
    else:
        a = min_a

    # for safety
    if a < 5:
        a = 5

    # set the height of the triangle
    h = a * math.sqrt(3) / 2

    # set the center of the triangle
    if random_pos:
        # random position
        x = randint(a // 2, w - a // 2)
        y = randint(h // 2, w - h // 2)
    else:
        # center
        x = w // 2
        y = w // 2

    # calculate the coordinates of the vertices
    # the vertices are in the order of drawing
    # the first vertex is the left vertex
    # the second vertex is the right vertex
    # the third vertex is the top vertex

    # left vertex
    x1 = x - a // 2
    y1 = y + a // 2

    # right vertex
    x2 = x + a // 2
    y2 = y + a // 2

    # top vertex
    x3 = x
    y3 = y - h // 2

    points = [(x1, y1), (x2, y2), (x3, y3)]

    # rotate the triangle
    if random_deg:
        deg = randint(0, 359)
    
    if deg != 0.0:
        points = rotate(points, origin=(x, y), degrees=deg)

    # a list of tuples is needed by polygon
    return list(map(tuple, points))


# value used in calculating the margins for squares
sqrt2_2_05 = math.sqrt(2)/2 - 0.5

def random_square(w, min_a, max_a,
                  random_pos=False, random_deg=False, deg=0):
    '''
    Calculates the coordinates of the 4 vertices of a square. 
    
    The square can have random scale, position and location,
    according to the parameters, with the constraint that the whole
    square will be inside a gris of size w x w.
    
    The coordinates can be used to draw a polygon inside a PIL image
    of size w x w.    
    
    
    Parameters
    ----------
    w : int
        Width of the image to draw into.
    min_a : int
        Minimal edge size of the square.
    max_a : int
        Maximal edge size of the square.
    random_pos : bool, optional
        Use random position for the square or not. The default is False.
    random_deg : bool, optional
        Use random rotation for the square or not. The default is False.
    deg : int, optional
        Rotation angle. The default is 0. 
        deg value is used only if random_deg is False. 

    Returns
    -------
    list of tuples of ints/floats
        Coordinates of the vertices of the square.


    Examples:
    1. A square of edge size 10, centered at the middle of the image,
       and parallel to the axes:
    
       poly = random_square(w=w, min_a=10, max_a=10,
                            random_pos=False, random_deg=False, deg=0)
    
    2. A square of random edge size between 10 to 15, 
       centered at the middle of the image and rotated to 45 degs:
    
       poly = random_square(w=w, min_a=10, max_a=15,
                            random_pos=False, random_deg=False, deg=45)

    '''
    
    # set edge size of the square in a 
    # confine the values
    if min_a < w //4:
        min_a = w // 4
    
    if max_a > w //2:
        max_a = w // 2
    
    if max_a < min_a:
        max_a = min_a
        
    
    if min_a != max_a:
        a = randint(min_a,max_a) 
    else: 
        a = min_a
    
    # for safety
    if a < 5:
        a = 5
             
    # set position of the square
    # the first vertex is x1,y1
    if random_pos:
        # L is used in setting the margin: where not to draw
        L = int(sqrt2_2_05 * a)+1            
    
        x1 = randint(L,w-(L+a+1))
        y1 = randint(L,w-(L+a+1))
        
    else:
        x1 = (w-a)//2
        y1 = (w-a)//2
    # set the other vertices the order is:
    # x1,y1 - bottom left
    # x2,y2 - bottom right
    # x3,y3 - top right
    # x4,y4 - top left

    # set the other vertices
    x2 = x1 + a
    y2 = y1
    x3 = x2
    y3 = y1 + a
    x4 = x1
    y4 = y3
    
    points = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        
    # rotation of the square
    
    if random_deg:
        deg = randint(0,89)
    
    if deg != 0.0:
        
        # find the middle
        mx = x1 + a/2
        my = y1 + a/2
        
        # rotate the points by a random angle, about the middle
        points = rotate(points, origin = (mx,my), degrees = deg)
      
    # a list of tuples is needed by polygon
    return list(map(tuple, points))

    
def generate_data(w=25, min_a=0, max_a=10, random_pos=False, 
                  random_deg=False, deg=0, sample_count = 1000, DO_SHOW = False):
    
    '''
    Generates images and labels for data sets.
    
    Each image is a float32 numpy array of shape (w,w).
    It has black background (value == 0.0) and gray to white forground object 
    (0.0 < values <= 1.0 ). The object is either a solid (filled in) square or
    a solid equilateral triangle.
    The images set is a float32 numpy array of shape (sample_count, w, w, 1).
    Half of the images are squares and half are triangles.
    
    Each label is scalar (int) of values 0,1.
    0 is for equilateral triangle,
    1 is for square
    2 is for triangle and square together(under 50% overlap)
    The lables set is a uint8 numpy array of shape (sample_count,)
    
    
    Parameters
    ----------
    The first six parameters 
    w, min_a, max_a, random_pos, random_deg, deg
    define the object. See the documentation of random_square() for details.
    
    sample_count : int, optional
        The number of data elements in the data set to be generated.
        The default is 1000.
    DO_SHOW : bool, optional
        Display the images created or not. The default is False.
    
    
    Returns
    -------
    images : numpy array of shape=(sample_count, w, w, 1),dtype=np.float32
        The images set.
    labels : numpy array of shape=(sample_count), dtype=np.uint8
        The labels set.


    Examples:
    1. Create and show a set of 1000 images (~500 squares, ~500 triangels), with 
       edge size 10, centered at the middle of the image,
       and parallel to the axes:
    
       parameters_train = {'w': w, 'min_a': 10, 'max_a': 10, 'random_pos': False,
              'random_deg': False, 'deg': 0, 'sample_count': 1000, 'DO_SHOW': True}

       train_images, train_labels = generate_data(**parameters_train)

    to check and view the data you can use the function
    show_9_images (see below)
    '''
    
    
    wd = w*10 # generated image is wd x wd, and then scaled down to w x w
    min_d = min_a*10
    max_d = max_a*10
    
    
    images = np.zeros(shape=(sample_count, w, w, 1),dtype=np.float32)
    labels = np.zeros(shape=(sample_count), dtype=np.uint8)
    
    for i in range(0, sample_count):
        
        # if random() > 0.5:
        if i % 3 == 0:
            # insted of random > 0.5 use (i % 2) == 0 and even sample_count
            # to make sure the two classes have exactly the same number of images.
            poly = random_equilateral_triangle(
                w=wd, min_a=min_d, max_a=max_d,
                random_pos=random_pos, random_deg=random_deg,deg=deg) 
            label = 0 # triangle
            
        elif i % 3 == 1:
            poly = random_square(
                w=wd, min_a=min_d, max_a=max_d,
                random_pos=random_pos, random_deg=random_deg,deg=deg)
            label = 1 # square

                
        else:
            overlap = True
            while overlap:
                square = random_square(w=wd, min_a=min_d, max_a=max_d, random_pos=random_pos, random_deg=random_deg,
                                    deg=deg)
                triangle = random_equilateral_triangle(w=wd, min_a=min_d, max_a=max_d, random_pos=random_pos,
                                                    random_deg=random_deg, deg=deg)

                square_poly = Polygon(square)
                triangle_poly = Polygon(triangle)

                overlap_area = calculate_overlap_area(square_poly, triangle_poly)

                if overlap_area / min(square_poly.area, triangle_poly.area) < 0.5:
                    overlap = False

            poly = square
            poly2 = triangle
            label = 2  # square and triangle
            

        # see these links on drawing inside PIL images
        # https://www.geeksforgeeks.org/python-pil-imagedraw-draw-rectangle/
        # https://note.nkmk.me/en/python-pillow-imagedraw/    

        # creating new Image object 
        img = Image.new("L", (wd, wd)) 

        # create image with either square or triangle
        img1 = ImageDraw.Draw(img) 
        img1.polygon(poly, fill ="white", outline ="white") 
        # img.show() 

        if label == 2:
            img1 = ImageDraw.Draw(img)
            img1.polygon(poly2, fill="white", outline="white")
        
        if DO_SHOW:
            ## show in matplotlib figure
            fig, axs = plt.subplots(figsize=(6,6))

            axs.imshow(img, cmap='gray')
            axs.title.set_text('original')
            axs.axis('off')
        
            plt.show()
            plt.close()
        
            time.sleep(0.1)

            fig, axs = plt.subplots(figsize=(6,6))

        
        img = img.resize((w,w), Image.BICUBIC)

        # convert to numpay array and scale to 0..1
        np_im = np.array(img, dtype=np.float32)/255

        if DO_SHOW:
            axs.imshow(np_im, cmap='gray')
            axs.title.set_text('scaled down')
            axs.axis('off')
        
            plt.show()
            plt.close()
        
            time.sleep(0.1)
        
        del img,img1 # we don't need it, but use only the numpay array copy 
        
        
        images[i] = np_im.reshape(w,w,1)
        labels[i] = label
        
    return images, labels


labels_str = ('triangle','square','triangle & square')

# show the first 9 elements of a data set
def show_9_images(the_images, the_labels):
    fig, axs = plt.subplots(3,3,figsize=(8,8))

    L = len(the_labels)

    k = 0
    for i in range(0,3):
        for j in range (0,3):
            img = the_images[k,:].reshape(w,w)
            label = the_labels[k]
        
            axs[i,j].imshow(img, cmap='gray')
            axs[i,j].title.set_text(labels_str[label])
            axs[i,j].axis('off')
            k+=1
            if k >= L:
                return
    plt.show()
    plt.close()



def plot_images_with_squares(images, indexs, title:str):
    """
    Function to plot up to 9 images that have been incorrectly tagged.
    Show the images in a 3x3 grid starting from index 0 to 8.
    If there are fewer than 9 images, the rest of the grid will be empty.

    Args:
        images: All images.
        indexes: Indexes of images that have been incorrectly tagged, from 1 to 9.
        title: Title for the plot.
    """
    print('\033[92m' + title + "is show" + '\033[0m')
    fig, axs = plt.subplots(3,3,figsize=(8,8))
    fig.suptitle(title, fontsize=16)
    k = 0
    for i in range(0,3):
        for j in range (0,3):
            img = images[indexs[k],:].reshape(w,w)
            axs[i,j].imshow(img, cmap='gray')
            axs[i,j].title.set_text(indexs[k])
            axs[i,j].axis('off')
            k+=1
            if k >= len(indexs):
                return
    
    # show the plot and wait for a key press
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

                
def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    filter_activation = activation[0, :, :, filter_index]
    loss = tf.reduce_mean(filter_activation)
    return loss



# Loss maximization via stochastic gradient ascent
@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    """
    Perform a single step of gradient ascent to maximize the activation of a specific filter in a CNN.

    Args:
        image: Input image tensor.
        filter_index: Index of the filter to maximize.
        learning_rate: Learning rate for gradient ascent.

    Returns:
        Updated image tensor after the gradient ascent step.
    """
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)

    grads = tape.gradient(loss, image)
    grads /= tf.math.reduce_std(grads) + 1e-8

    image = tf.clip_by_value(image + learning_rate * grads, 0.0, 1.0)
    return image


def generate_filter_pattern(filter_index):
    """
    Generate a pattern that maximizes the activation of a specific filter in a CNN.

    Args:
        filter_index: Index of the filter to maximize.

    Returns:
        Generated pattern as a numpy array.
    """
    iterations = 20
    learning_rate = 1.0
    image = tf.random.uniform((1, img_width, img_height, 1))
    image = tf.Variable(image)

    for iteration in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)

    img = deprocess_image(image[0].numpy())
    return img


# Utility function to convert a tensor into a valid image        
def deprocess_image(image):
    """
    Convert a tensor image to a valid image format by applying deprocessing operations.

    Args:
        image: Input image as a numpy array.

    Returns:
        Deprocessed image as a numpy array.
    """
    image = (image - image.mean()).astype(np.float64)
    image /= (image.std() + 1e-8).astype(np.float64)
    image *= 0.1
    image += 0.5
    image = np.clip(image, 0, 1)
    image *= 255
    image = image.astype(np.float64)
    return image

def generate_activation_map(img):
    """
    Generate and display the activation map for an input image.

    Args:
        img: Input image as a numpy array.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img[0, :, :, 0], cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # Load the pre-trained VGG16 model
    model = VGG16(weights='imagenet', include_top=True)

    last_conv_layer_name = None
    classifier_layer_names = []

    for layer in model.layers[::-1]:
        if 'conv' in layer.name:
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name:
        found_last_conv = False
        for layer in model.layers:
            if found_last_conv:
                classifier_layer_names.append(layer.name)
            if layer.name == last_conv_layer_name:
                found_last_conv = True

    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)
    classifier_input = Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)

    classifier_model = Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap_resized = cv2.resize(jet_heatmap, (img.shape[2], img.shape[1]))
    overlay = jet_heatmap_resized * 0.4 + np.expand_dims(img[0, :, :, 0], axis=-1) * 0.6

    axs[1].imshow(overlay, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('Heatmap Overlay')

    axs[2].imshow(jet_heatmap)
    axs[2].axis('off')
    axs[2].set_title('Heatmap')

    cax = fig.add_axes([axs[0].get_position().x0, axs[0].get_position().y0 - 0.1, axs[0].get_position().width, 0.02])
    cbar = plt.colorbar(cm.ScalarMappable(cmap='gray'), cax=cax, orientation='horizontal')
    cbar.set_label('Original Image', fontsize=8)

    # cax = fig.add_axes([axs[1].get_position().x0, axs[1].get_position().y0 - 0.1, axs[1].get_position().width, 0.02])
    # cbar = plt.colorbar(cm.ScalarMappable(cmap='gray'), cax=cax, orientation='horizontal')
    # cbar.set_label('Heatmap Overlay', fontsize=8)

    cax = fig.add_axes([axs[2].get_position().x0, axs[2].get_position().y0 - 0.1, axs[2].get_position().width, 0.02])
    cbar = plt.colorbar(cm.ScalarMappable(cmap='jet'), cax=cax, orientation='horizontal')
    cbar.set_label('Heatmap', fontsize=8)

    # Add legend for the heatmap colors
    legend_img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        legend_img[:, i, :] = jet_colors[int(i * 255 / 100)]

    axs[1].figure.figimage(legend_img, cmap='jet', xo=axs[1].get_xlim()[1] + 20, yo=axs[1].get_ylim()[0] - legend_img.shape[0])
    axs[2].figure.figimage(legend_img, cmap='jet', xo=axs[2].get_xlim()[1] + 20, yo=axs[2].get_ylim()[0] - legend_img.shape[0])

    plt.show()



if __name__ == "__main__":

    print('\033[96m' + 'Start' + '\033[0m')
    if MACOS:
        NSBeep()
    else:
        print('\a')
        
    if(GENERATA_NEW_DATA):
        print('\033[96m' + 'Denerating data' + '\033[0m')

        parameters_train = {'w': w, 'min_a': MIN_A, 'max_a': MAX_A, 'random_pos': RANDOM_POS, 'random_deg': RANDOM_DEG, 'deg': DEG, 'sample_count': TRAINING_SET, 'DO_SHOW': False}
        parameters_test = {'w': w, 'min_a': MIN_A, 'max_a': MAX_A, 'random_pos': RANDOM_POS, 'random_deg': RANDOM_DEG, 'deg': DEG, 'sample_count': TEST_SET, 'DO_SHOW': False}

        # generate data 
        train_images, train_labels = generate_data(**parameters_train)

        validation_images, validation_labels = generate_data(**parameters_test)

        test_images, test_labels = generate_data(**parameters_test)

        print('\033[92m' + 'Data generated' + '\033[0m')

        #print the first 12 images
        show_9_images(train_images, train_labels)

        # # print counter for each label:
        # print('\033[92m' + 'Training set:' + '\033[0m')
        # print('triangle: ', np.count_nonzero(train_labels == 0))
        # print('square: ', np.count_nonzero(train_labels == 1))
        # print('triangle & square: ', np.count_nonzero(train_labels == 2))

        # print('\033[92m' + 'Validation set:' + '\033[0m')
        # print('triangle: ', np.count_nonzero(validation_labels == 0))
        # print('square: ', np.count_nonzero(validation_labels == 1))
        # print('triangle & square: ', np.count_nonzero(validation_labels == 2))

        # print('\033[92m' + 'Test set:' + '\033[0m')
        # print('triangle: ', np.count_nonzero(test_labels == 0))
        # print('square: ', np.count_nonzero(test_labels == 1))
        # print('triangle & square: ', np.count_nonzero(test_labels == 2))

        # Save the Data
        print('\033[96m' +'Saving Data to disk' + '\033[0m')

        savez_compressed(PATH_TO_DATA,
                        train_images, train_labels,
                        validation_images, validation_labels,
                        test_images, test_labels )

        print('\033[92m' + f'Data saved to disk in: {PATH_TO_DATA}' + '\033[0m')

    
    # load data
    print('\033[96m' +'Loading Data from disk' + '\033[0m')

    dict_data = load(PATH_TO_DATA)

    train_images      = dict_data['arr_0']
    train_labels      = dict_data['arr_1']
    validation_images = dict_data['arr_2']
    validation_labels = dict_data['arr_3']
    test_images       = dict_data['arr_4']
    test_labels       = dict_data['arr_5']

    print('\033[92m' +'Data loaded from disk' + '\033[0m')

    # print to see if everything is OK
    print('\033[96m' +'~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~`' + '\033[0m')
    print("The shapes of the data sets are:")
    print('train_images.shape: ' + str(train_images.shape))
    print('train_labels.shape: ' + str(train_labels.shape))
    print('validation_images.shape: ' + str(validation_images.shape))
    print('validation_labels.shape: ' + str(validation_labels.shape))
    print('test_images.shape: ' + str(test_images.shape))
    print('test_labels.shape: ' + str(test_labels.shape))
    print('\033[96m' +'~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~`' + '\033[0m')
    
    if train_images.shape[0] != train_labels.shape[0]:
        print('\033[91m' +'Error: train_images.shape[0] != train_labels.shape[0]' + '\033[0m')
    if validation_images.shape[0] != validation_labels.shape[0]:
        print('\033[91m' +'Error: validation_images.shape[0] != validation_labels.shape[0]' + '\033[0m')
    if test_images.shape[0] != test_labels.shape[0]:
        print('\033[91m' +'Error: test_images.shape[0] != test_labels.shape[0]' + '\033[0m')

    # Convert the labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=3)
    validation_labels = to_categorical(validation_labels, num_classes=3)
    test_labels = to_categorical(test_labels, num_classes=3)

    if 'model' in globals():
        del model

    model = models.Sequential()

    if GENERATA_NEW_MODEL:
        print('\033[96m' + 'Generating new model' + '\033[0m')

        if RUN_ON_GPU:
            # set up GPU to use as much memory as possible for training
            print('\033[93m' + 'Setting up GPU memory growth' + '\033[0m')
            physical_devices = tf.config.list_physical_devices('GPU')
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                from keras import backend as K
                K.set_image_data_format('channels_last')

                if len(physical_devices) > 1:
                    tf.config.experimental.set_memory_growth(physical_devices[1], True)
                print('\033[92m' + 'GPU memory growth set to True' + '\033[0m')
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                print('\033[91m' + 'Error: GPU memory growth could not be set' + '\033[0m')
                pass

        model.add(layers.Conv2D(8 , (3, 3), activation='relu', input_shape=(w, w, 1)))
        model.add(layers.MaxPooling2D((10, 10)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Flatten())
        model.add(layers.Dense(5, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))


        print('\033[92m' + 'Model summary:' + '\033[0m')
        model.summary()
        print('\033[92m' + "batch_size: " + '\033[0m' + str(BATCH_SIZE))
        print('\033[92m' + "epochs: " + '\033[0m' + str(EPOCHS))
        print('\033[92m' + "learning_rate: " + '\033[0m' + str(LEARNING_RATE))

        # Compile the model
        opt = optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt,
                    loss='categorical_crossentropy', 
                    # loss = 'binary_crossentropy',
                    metrics=['accuracy'])
        
        print('\033[96m' + 'Training model' + '\033[0m')

        history = model.fit(train_images, train_labels, epochs=EPOCHS,batch_size=BATCH_SIZE,
                            validation_data=(validation_images, validation_labels))

        print('\033[92m' + 'Training finished' + '\033[0m')
        
        # Show results
        validation_loss, validation_acc = model.evaluate(validation_images, validation_labels, verbose=0)
        print('\033[92m' + 'Validation accuracy: ' + '\033[0m',validation_acc)
        print('\033[92m' + 'Validation loss: ' + '\033[0m',validation_loss)

        # make a sound when the training is done by ascii bell
        print('\033[92m' + 'Training is done!' + '\033[0m')
        if MACOS:
            NSBeep()
            time.sleep(0.5)
            NSBeep()
            time.sleep(0.5)
            NSBeep()
        else:
            print('\a')
            time.sleep(0.5)
            print('\a')
            time.sleep(0.5)
            print('\a')
        

        # #show activations
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.close()


        # save the model

        # serialize model to JSON
        print('\033[96m' + 'Starts to save the model' + '\033[0m')
        model_json = model.to_json()
        with open(PATH_TO_MODEL, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(PATH_TO_WEIGHTS)
        print('\033[92m' + "Saved model to disk" + '\033[0m')


    # load json and create model
    print('\033[96m' + 'Starts to load the model' + '\033[0m')
    json_file = open(PATH_TO_MODEL, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(PATH_TO_WEIGHTS)
    print('\033[92m' + "Loaded model from disk" + '\033[0m')

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(test_images, test_labels, verbose=0)
    print("%s: %.3f%%" % (loaded_model.metrics_names[1], score[1]*100))
    model = loaded_model


    # Show some predictions of the data
    print("Test images shape:", test_images.shape)
    predictions = model.predict(test_images)
    print("Predictions shape:", predictions.shape)

    # Get the index of the highest class probability for each prediction
    predictions = np.argmax(predictions, axis=1)
    print("Reshaped predictions shape:", predictions.shape)

    # Convert test labels to corresponding classes (0, 1, or 2)
    test_labels = np.argmax(test_labels, axis=1)
    print("Reshaped test labels shape:", test_labels.shape)

    # convert to 0 or 1 or 2 by round
    predictions = np.round(predictions)
    print(predictions.shape)
    predictions = predictions.reshape(predictions.shape[0])
    print(predictions.shape)

    # print the percentage of correct predictions in the test set 
    print('\033[92m' + 'The percentage of correct predictions in the test set:' + '\033[0m',
    str(np.sum(predictions == test_labels)/test_labels.shape[0]*100) + '%')

    # find all mismatched predictions
    # print in red the sum of the mismatched predictions for each class
    mismatched_predictions = np.where(predictions != test_labels)[0]
    mismatched_triangles = []
    mismatched_squares = []
    mismatched_2Shapes = []

    for i in mismatched_predictions:
        if test_labels[i] == 0:
            mismatched_triangles.append(i)
        elif test_labels[i] == 1:
            mismatched_squares.append(i)
        elif test_labels[i] == 2:
            mismatched_2Shapes.append(i)

    print('\033[91m' + 'Number of mismatched triangles: ' + '\033[0m', len(mismatched_triangles))
    print('\033[91m' + 'Number of mismatched squares: ' + '\033[0m', len(mismatched_squares))
    print('\033[91m' + 'Number of mismatched 2 Shapes: ' + '\033[0m', len(mismatched_2Shapes))
    print("miss match index: ", mismatched_predictions)

    # show the mismatched predictions for each class
    if 0 < len(mismatched_triangles) < 10:
        print('\033[91m' + 'Mismatched triangles:' + '\033[0m')
        # show the images of the mismatched predictions for triangles in a grid of 3x3
        plot_images_with_squares(test_images, mismatched_triangles,title="miss match triangles")

    elif len(mismatched_triangles) >= 10:
        # show the first 9 images of the mismatched predictions for triangles in a grid of 3x3
        print('\033[91m' + 'Mismatched triangles:' + '\033[0m')
        plot_images_with_squares(test_images, mismatched_triangles[:9],title="miss match triangles semple")

    if 0 < len(mismatched_squares) < 10:
        print('\033[91m' + 'Mismatched squares:' + '\033[0m')
        plot_images_with_squares(test_images, mismatched_squares,title="miss match squares")

    elif len(mismatched_squares) >= 10:
        print('\033[91m' + 'Mismatched squares:' + '\033[0m')
        plot_images_with_squares(test_images, mismatched_squares[:9],title="miss match squares semple")

    if 0 < len(mismatched_2Shapes) < 10:
        print('\033[91m' + 'Mismatched 2Shapes:' + '\033[0m')
        plot_images_with_squares(test_images, mismatched_2Shapes,title="miss match 2 Shapes")

    elif len(mismatched_2Shapes) >= 10:
        print('\033[91m' + 'Mismatched 2Shapes:' + '\033[0m')
        plot_images_with_squares(test_images, mismatched_2Shapes[:9],title="miss match 2 Shapes semple")


    # show the activations of 2D layers
    # if SHOW_ACTIVATIONS and validation_acc > 0.98:
    if SHOW_ACTIVATIONS:

        for image_index in range(NUM_OF_IMAGES_FOR_ACTIVATION):
            activation_index = image_index

            # Show the image
            plt.figure()
            plt.imshow(test_images[image_index, :, :, 0], cmap='gray')
            plt.title(f'Image {image_index}', fontsize=16)
            plt.show()
            plt.close()

            for i, layer in enumerate(model.layers):
                layer_name = layer.name

                print(layer_name)

                if '2d' not in layer_name:
                    break

                if 'max_pooling' in layer_name:
                    # Define a new model with the desired layer as output
                    conv_output_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

                    # Get the activations for the input images
                    activations = conv_output_model.predict(test_images)
                    
                    # Display the activation map for the first filter
                    fig = plt.figure(figsize=(10, 10))
                    plt.suptitle(f'Activation Map for {layer_name} on image {image_index}', fontsize=16)
                    ax = plt.gca()
                    ax.imshow(activations[image_index, :, :, 0], cmap='jet')
                    ax.set_title(f'Filter 1')
                    ax.axis('off')
                    plt.show()
                    plt.close()

                if 'conv2d' in layer_name:
                    # Get the Conv2D layer
                    conv_layer = model.get_layer(layer_name)

                    # Define a new model with the Conv2D layer as output
                    conv_output_model = Model(inputs=model.input, outputs=conv_layer.output)

                    # Get the activations for the input images
                    activations = conv_output_model.predict(test_images)

                    # Check if there are multiple filters
                    num_filters = activations.shape[-1]  # Number of filters in the Conv2D layer

                    if num_filters > 1:
                        rows = int(np.ceil(np.sqrt(num_filters)))
                        cols = int(np.ceil(num_filters / rows)) if num_filters > 1 else 1


                        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
                        fig.suptitle(f'Activation Maps for {layer_name} on image {image_index}', fontsize=16)

                        for j in range(num_filters):
                            a_row = j // cols
                            a_col = j % cols
                            if cols > 1:
                                ax = axs[a_row,a_col]  # Update indexing here
                            else:
                                ax = axs[a_row]
                            ax.imshow(activations[image_index, :, :, j], cmap='jet')
                            ax.set_title(f'Filter {j+1}')
                            ax.axis('off')

                        plt.tight_layout()
                        plt.show()
                        plt.close()

                    else:
                        fig = plt.figure(figsize=(10, 10))
                        plt.suptitle(f'Activation Map for {layer_name} on image {image_index}', fontsize=16)
                        ax = plt.gca()
                        ax.imshow(activations[image_index, :, :, 0], cmap='jet')
                        ax.set_title(f'Filter 1')
                        ax.axis('off')
                        plt.show()
                        plt.close()

    # Displaying the images that cause the maximum response to the filters we found
    if SHOW_MAX_ACTIVATIONS:
        layer_name = 'conv2d'
        layer = model.get_layer(name=layer_name)
        
        # Creating a feature extractor model
        feature_extractor = Model(inputs=model.inputs, outputs=layer.output)
        
        # Resizing the input image
        resized_image = image.smart_resize(test_images[0], size=(25, 25))
            
        # Using the feature extractor
        activations = feature_extractor.predict(resized_image[np.newaxis, ...])

        # Function to generate filter visualizations
        img_width = w
        img_height = w


        # Generating a grid of all filter response patterns in a layer
        all_images = []
        for filter_index in range(8):
            print(f"Processing filter {filter_index}")
            image = deprocess_image(generate_filter_pattern(filter_index))
            all_images.append(image)

        # show the generated images in a grid
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f'Filter Response Patterns for {layer_name}', fontsize=16)

        for j in range(8):
            a_row = j // 3
            a_col = j % 3
            ax = axs[a_row,a_col]
            ax.imshow(all_images[j], cmap='gray')
            ax.set_title(f'Filter {j+1}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()
    

    # Visualizing heatmaps of class activation
    if SHOW_ACTIVATIONS_MAP:

        # Take the first 3 images from the test set and rescale them to 224x224
        images = [image.smart_resize(img, size=(224, 224)) for img in test_images[:NUM_OF_IMAGES_FOR_MAP]]
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images]

        for img in images_rgb:
            img = np.expand_dims(img, axis=0)
            generate_activation_map(img)