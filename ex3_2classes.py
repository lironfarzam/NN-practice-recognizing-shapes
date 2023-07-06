# -*- coding: utf-8 -*-
import math
import time
from random import randint, random
import sys
from AppKit import NSBeep
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray, load, savez_compressed
from PIL import Image, ImageDraw
from tensorflow.keras import layers, models, optimizers, Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# --- constants ---
GENERATA_NEW_DATA = False
GENERATA_NEW_MODEL = False
SHOW_ACTIVATIONS = True
NUM_OF_IMAGES_FOR_ACTIVATION = 4

Q = '13'
PATH_TO_DATA = 'data/data_ex3_' + Q +'.npz'
PATH_TO_MODEL = 'model/model_ex3_' + Q +'.json'
PATH_TO_WEIGHTS = 'weights/model_ex3_' + Q +'.h5'

# image size is w by w
w = 25
MIN_A = 6
MAX_A = 12
RANDOM_POS = True
RANDOM_DEG = True
DEG = 0

TRAINING_SET = 10000
VALIDATION_SET = 2000
TEST_SET = 2000

EPOCHS = 40
BATCH_SIZE = 60
LEARNING_RATE = 0.002

GPU = True
MACOS = True


# --- functions ---
def rotation_matrix(ad):
    # ad = angle in [deg]
    a = math.radians(ad)
    R = asarray([[math.cos(a), -math.sin(a)],
                 [math.sin(a), math.cos(a)]])
    return R
    

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
        if i % 2 == 0:
            # insted of random > 0.5 use (i % 2) == 0 and even sample_count
            # to make sure the two classes have exactly the same number of images.
            poly = random_equilateral_triangle(
                w=wd, min_a=min_d, max_a=max_d,
                random_pos=random_pos, random_deg=random_deg,deg=deg) 
            label = 0 # triangle
            
        else:
            poly = random_square(
                w=wd, min_a=min_d, max_a=max_d,
                random_pos=random_pos, random_deg=random_deg,deg=deg)
            label = 1 # square

        # see these links on drawing inside PIL images
        # https://www.geeksforgeeks.org/python-pil-imagedraw-draw-rectangle/
        # https://note.nkmk.me/en/python-pillow-imagedraw/    

        # creating new Image object 
        img = Image.new("L", (wd, wd)) 

        # create image with either square or triangle
        img1 = ImageDraw.Draw(img) 
        img1.polygon(poly, fill ="white", outline ="white") 
        # img.show() 
        
        if DO_SHOW:
            ## show in matplotlib figure
            fig, axs = plt.subplots(figsize=(6,6))

            axs.imshow(img, cmap='gray')
            axs.title.set_text('original')
            axs.axis('off')
        
            plt.show()
        
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
        
            time.sleep(0.1)
        
        del img,img1 # we don't need it, but use only the numpay array copy 
        
        
        images[i] = np_im.reshape(w,w,1)
        labels[i] = label
        
    return images, labels


labels_str = ('triangle','square')

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


def plot_images_with_squares(images, indexs, title:str):
    """
    function to plot up to 9 images that have been incorrectly tagged.
    show the images in 3x3 grid start from index 0 to 8.
    if there are less than 9 images the rest of the grid will be empty.

    Args:
        images: all images
        indexs: indexes of images that have been incorrectly tagged can be from 1 to 9.
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
    print("Press any key to continue")
    plt.waitforbuttonpress()
    plt.close()

    

if __name__ == "__main__":

    print('\033[96m' + 'Start' + '\033[0m')
    NSBeep()
        
    #%%
    # =======
    # generate data for training, validation and testing
    # add the parameters you need to the dictionaries below
    # the parameters are explained in the function generate_data()
    # =======

    if(GENERATA_NEW_DATA):
        print('\033[96m' + 'Denerating data' + '\033[0m')

        parameters_train = {'w': w, 'min_a': MIN_A, 'max_a': MAX_A, 'random_pos': RANDOM_POS, 'random_deg': RANDOM_DEG, 'deg': DEG, 'sample_count': TRAINING_SET, 'DO_SHOW': False}
        parameters_test = {'w': w, 'min_a': MIN_A, 'max_a': MAX_A, 'random_pos': RANDOM_POS, 'random_deg': RANDOM_DEG, 'deg': DEG, 'sample_count': TEST_SET, 'DO_SHOW': False}


        # generate data 
        train_images, train_labels = generate_data(**parameters_train)

        validation_images, validation_labels = generate_data(**parameters_test)

        test_images, test_labels = generate_data(**parameters_test)

        print('\033[92m' + 'Data generated' + '\033[0m')


        #%%

        # Save the Data
        print('\033[96m' +'Saving Data to disk' + '\033[0m')

        savez_compressed(PATH_TO_DATA,
                        train_images, train_labels,
                        validation_images, validation_labels,
                        test_images, test_labels )

        print('\033[92m' + f'Data saved to disk in: {PATH_TO_DATA}' + '\033[0m')

    #%%
    
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

    #%%

    if 'model' in globals():
        del model


    model = models.Sequential()

    if GENERATA_NEW_MODEL:

        # prepare the CNN for classifying triangles vs squares
        if GPU:
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


        model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(w, w, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(3, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))


        print('\033[92m' + 'Model summary:' + '\033[0m')
        model.summary()
        print('\033[92m' + "batch_size: " + '\033[0m' + str(BATCH_SIZE))
        print('\033[92m' + "epochs: " + '\033[0m' + str(EPOCHS))
        print('\033[92m' + "learning_rate: " + '\033[0m' + str(LEARNING_RATE))

        # Compile the model
        opt = optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=opt,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        # Train the model
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

        # #%%
        # #show activations
        # show 2 figer in same plot:
        # left - loss: The loss graph as a function of the number of epochs, for both the training and validation sets.
        # right - accuracy: The accuracy graph as a function of the number of epochs, for both the training and validation sets.
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

            
        # #%%

        # # save the model

        # serialize model to JSON
        print('\033[96m' + 'Starts to save the model' + '\033[0m')
        model_json = model.to_json()
        with open(PATH_TO_MODEL, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(PATH_TO_WEIGHTS)
        print('\033[92m' + "Saved model to disk" + '\033[0m')

    #%%

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
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(test_images, test_labels, verbose=0)
    print("%s: %.3f%%" % (loaded_model.metrics_names[1], score[1]*100))
    model = loaded_model


    #%%

    # show some predictions of the data
    print(test_images.shape)
    predictions = model.predict(test_images)
    print(predictions.shape)
    predictions = np.round(predictions)
    print(predictions.shape)
    predictions = predictions.reshape(predictions.shape[0])
    print(predictions.shape)

    # print('\033[92m' + 'Some predictions of the data:' + '\033[0m')
    # print('predictions: ' + str(predictions))
    # print('test_labels: ' + str(test_labels))
    # print the percentage of correct predictions in the test set 
    print('\033[92m' + 'The percentage of correct predictions in the test set:' + '\033[0m',
    str(np.sum(predictions == test_labels)/test_labels.shape[0]*100) + '%')

    #%%
    # find all mismatched predictions
    # save the mismatched predictions in a 2 lists - one for triangles and one for squares
    # print in red the sum of the mismatched predictions for each class
    # if the sum is < 10 for a class, print the images of the mismatched predictions for that class

    mismatched_predictions = np.where(predictions != test_labels)[0]
    mismatched_triangles = []
    mismatched_squares = []
    for i in mismatched_predictions:
        if test_labels[i] == 0:
            mismatched_triangles.append(i)
        else:
            mismatched_squares.append(i)
    print('\033[91m' + 'Number of mismatched triangles: ' + '\033[0m', len(mismatched_triangles))
    print('\033[91m' + 'Number of mismatched squares: ' + '\033[0m', len(mismatched_squares))
    print("miss match index: ", mismatched_predictions)

    #%%
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

    #%%
    # show the activations of 2D layers

    # image_index = 0  # Replace with the desired image index
    # activation_index = 0  # Replace with the desired activation index

if SHOW_ACTIVATIONS:
    for image_index in range(NUM_OF_IMAGES_FOR_ACTIVATION):
        activation_index = image_index

        # Show the image
        plt.figure()
        plt.imshow(test_images[image_index, :, :, 0], cmap='gray')
        plt.title(f'Image {image_index}', fontsize=16)
        plt.show()

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



#%%

# # Q8 model:
# 
#     model.add(layers.Conv2D(1, (5, 5), activation='relu', input_shape=(w, w, 1)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(1, activation='sigmoid'))
# 
#     EPOCHS = 50
#     BATCH_SIZE = 1
#     LEARNING_RATE = 0.002
#
#     Total params: 127
#
# ---------------------------------------------------------------
# # Q9 model:

#      model.add(layers.Conv2D(1, (5, 5), activation='relu', input_shape=(w, w, 1)))
#      model.add(layers.MaxPooling2D((2, 2)))
#      model.add(layers.Flatten())
#      model.add(layers.Dense(1, activation='sigmoid'))
#
#      EPOCHS = 7
#      BATCH_SIZE = 5
#      LEARNING_RATE = 0.0005
#
#      Total params: 127
#
# ---------------------------------------------------------------
# # Q10 model:
#
#       model.add(layers.Conv2D(1, (5, 5), activation='relu', input_shape=(w, w, 1)))
#       model.add(layers.MaxPooling2D((2, 2)))
#       model.add(layers.Flatten())
#       model.add(layers.Dense(1, activation='sigmoid'))
#
#       EPOCHS = 15
#       BATCH_SIZE = 5
#       LEARNING_RATE = 0.0005
#
#       Total params: 127
#
# ---------------------------------------------------------------
# # Q11 model:
#       
#       model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=(w, w, 1)))
#       model.add(layers.MaxPooling2D((2, 2)))
#       model.add(layers.Flatten())
#       model.add(layers.Dense(1, activation='sigmoid'))
# 
#       EPOCHS = 15
#       BATCH_SIZE = 5
#       LEARNING_RATE = 0.001
# 
#       Total params: 132
# ---------------------------------------------------------------
# # Q12 model:

    # model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(w, w, 1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(2, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(4, activation='relu'))
    # model.add(layers.Dense(1, activation='sigmoid'))

    # EPOCHS = 15
    # BATCH_SIZE = 5
    # LEARNING_RATE = 0.0002

    # Total params: 251

# ---------------------------------------------------------------
# # Q13 model:

    # model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(w, w, 1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(3, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(4, activation='relu'))
    # model.add(layers.Dense(1, activation='sigmoid'))

    # EPOCHS = 40
    # BATCH_SIZE = 60
    # LEARNING_RATE = 0.002
# ---------------------------------------------------------------

