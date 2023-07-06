
import math
import time
from shapely.geometry import Polygon
import math
import time
from random import randint, random
from AppKit import NSBeep
from shapely.geometry import Polygon
import random
import matplotlib.pyplot as plt


w = 25
MIN_A = 6
MAX_A = 12
RANDOM_POS = True
RANDOM_DEG = True
TT = 60
DEG = 0

# value used in calculating the margins for squares
sqrt2_2_05 = math.sqrt(2)/2 - 0.5
sqrt3_2 = math.sqrt(3) / 2



def rotation_matrix(ad):
    a = math.radians(ad)
    R = [[math.cos(a), -math.sin(a)],
         [math.sin(a), math.cos(a)]]
    return R

def rotate(points, origin=(0, 0), degrees=0):
    R = rotation_matrix(degrees)
    o = [origin[0], origin[1]]
    p = [[point[0], point[1]] for point in points]
    rotated_points = []
    for point in p:
        qx = R[0][0] * (point[0] - o[0]) + R[0][1] * (point[1] - o[1]) + o[0]
        qy = R[1][0] * (point[0] - o[0]) + R[1][1] * (point[1] - o[1]) + o[1]
        rotated_points.append((qx, qy))
    return rotated_points

def random_equilateral_triangle(w, min_a, max_a, random_pos=False, random_deg=False, deg=0):
    if min_a < w // 4:
        min_a = w // 4

    if max_a > w // 2:
        max_a = w // 2

    if max_a < min_a:
        max_a = min_a

    if min_a != max_a:
        a = random.randint(min_a, max_a)
    else:
        a = min_a

    if a < 5:
        a = 5

    h = a * math.sqrt(3) / 2

    if random_pos:
        x = random.randint(a // 2, w - a // 2)
        y = random.randint(h // 2, w - h // 2)
    else:
        x = w // 2
        y = w // 2

    x1 = x - a // 2
    y1 = y + a // 2

    x2 = x + a // 2
    y2 = y + a // 2

    x3 = x
    y3 = y - h // 2

    points = [(x1, y1), (x2, y2), (x3, y3)]

    if random_deg:
        deg = random.randint(0, 119)

    if deg != 0.0:
        points = rotate(points, origin=(x, y), degrees=deg)

    return points

def random_equilateral_triangle1(w, min_a, max_a, random_pos=False, random_deg=False, deg=0):
    '''
    Calculates the coordinates of the 3 vertices of an equilateral triangle.
    
    The triangle can have random scale, position, and rotation
    according to the parameters, with the constraint that the whole
    triangle will be inside a grid of size w x w.
    
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
        Coordinates of the vertices of the equilateral triangle.


    Examples:
    1. An equilateral triangle with edge size 10, centered at the middle of the image,
       and parallel to the axes:
    
       poly = random_equilateral_triangle(w=w, min_a=10, max_a=10,
                                          random_pos=False, random_deg=False, deg=0)
    
    2. An equilateral triangle with random edge size between 10 to 15, 
       centered at the middle of the image and rotated to 45 degrees:
    
       poly = random_equilateral_triangle(w=w, min_a=10, max_a=15,
                                          random_pos=False, random_deg=False, deg=45)

    '''

    # Set edge size of the triangle and confine the values
    if min_a < w // 4:
        min_a = w // 4

    if max_a > w // 2:
        max_a = w // 2

    if max_a < min_a:
        max_a = min_a

    if min_a != max_a:
        a = random.randint(min_a, max_a)
    else:
        a = min_a

    # For safety, ensure the minimum edge size is at least 5
    if a < 5:
        a = 5

    # Set position of the triangle
    # The first vertex is x1, y1
    if random_pos:
        # L is used in setting the margin: where not to draw
        L = int(sqrt3_2 * a) + 1

        x1 = random.randint(L, w - (L + a + 1))
        y1 = random.randint(L, w - (L + a + 1))

    else:
        x1 = (w - a) // 2
        y1 = (w - a) // 2

    # Calculate the coordinates of the other two vertices of the equilateral triangle
    x2 = x1 + a
    y2 = y1
    x3 = x1 + a // 2
    y3 = y1 + int(a * sqrt3_2)

    points = [(x1, y1), (x2, y2), (x3, y3)]

    # Rotation of the triangle
    if random_deg:
        deg = random.randint(0, 119)

    if deg != 0.0:
        # Find the middle
        mx = (x1 + x2 + x3) / 3
        my = (y1 + y2 + y3) / 3

        # Rotate the points by a random angle around the middle
        points = rotate(points, origin=(mx, my), degrees=deg)

    # Return the list of tuples representing the vertices of the equilateral triangle
    return list(map(tuple, points))

def random_square(w, min_a, max_a, random_pos=False, random_deg=False, deg=0):
    if min_a < w // 4:
        min_a = w // 4

    if max_a > w // 2:
        max_a = w // 2

    if max_a < min_a:
        max_a = min_a

    if min_a != max_a:
        a = random.randint(min_a, max_a)
    else:
        a = min_a

    if a < 5:
        a = 5

    if random_pos:
        x = random.randint(a // 2, w - a // 2)
        y = random.randint(a // 2, w - a // 2)
    else:
        x = w // 2
        y = w // 2

    x1 = x - a // 2
    y1 = y - a // 2

    x2 = x + a // 2
    y2 = y - a // 2

    x3 = x + a // 2
    y3 = y + a // 2

    x4 = x - a // 2
    y4 = y + a // 2

    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    if random_deg:
        deg = random.randint(0, 89)

    if deg != 0.0:
        points = rotate(points, origin=(x, y), degrees=deg)

    return points

def random_square1(w, min_a, max_a, random_pos=False, random_deg=False, deg=0):
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

def calculate_overlap_area(poly1, poly2):
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    intersection = polygon1.intersection(polygon2)
    overlap_area = intersection.area
    return overlap_area

def show_9_images(the_images):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    axs = axs.flatten()
    for img, ax in zip(the_images, axs):
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def generate_unique_triangles():
    triangles = []
    start_time = time.time()
    while time.time() - start_time < TT:
        triangle = random_equilateral_triangle(w=w, min_a=MIN_A, max_a=MAX_A, random_pos=RANDOM_POS, random_deg=RANDOM_DEG, deg=DEG)
        if triangle not in triangles:
            triangles.append(triangle)
            # print(triangle)
            print(len(triangles))
            start_time = time.time()

    return len(triangles)

def generate_unique_triangles1():
    triangles = []
    start_time = time.time()
    while time.time() - start_time < TT:
        triangle = random_equilateral_triangle(w=w, min_a=MIN_A, max_a=MAX_A, random_pos=RANDOM_POS, random_deg=RANDOM_DEG, deg=DEG)
        if triangle not in triangles:
            triangles.append(triangle)
            # print(triangle)
            start_time = time.time()

    return len(triangles)

def generate_unique_square():
    squares = []
    start_time = time.time()
    while time.time() - start_time < TT:
        square = random_square(w=w, min_a=MIN_A, max_a=MAX_A, random_pos=RANDOM_POS, random_deg=RANDOM_DEG, deg=DEG)
        if square not in squares:
            squares.append(square)
            # print(square)
            print(len(squares))
            start_time = time.time()

    return len(squares)

def generate_unique_square1():
    squares = []
    start_time = time.time()
    while time.time() - start_time < TT:
        square = random_square1(w=w, min_a=MIN_A, max_a=MAX_A, random_pos=RANDOM_POS, random_deg=RANDOM_DEG, deg=DEG)
        if square not in squares:
            squares.append(square)
            # print(square)
            start_time = time.time()

    return len(squares)

NSBeep()

triangle_sum = generate_unique_triangles()

NSBeep()
# triangle_sum1 = generate_unique_triangles1()


square_sum = generate_unique_square()
# square_sum1 = generate_unique_square1()
NSBeep()

print("Number of unique triangles: ", triangle_sum)
# print("Number of unique triangles1: ", triangle_sum1)
print("Number of unique squares: ", square_sum)
# print("Number of unique squares1: ", square_sum1)

NSBeep()
time.sleep(0.5)
NSBeep()
time.sleep(0.5)
NSBeep()

print("Total number of unique shapes: ", triangle_sum + square_sum)
# print("Total number of unique shapes1: ", triangle_sum1 + square_sum1)
