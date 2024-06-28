# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:53:59 2022

@author: Hande
"""

import math
import cv2
import numpy as np
import math
import skimage.io
import matplotlib.pyplot as plt
import skimage.filters
from scipy import ndimage

''' codes from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders '''

##------------------------------------------------------------------------------
## Definitions 
##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((col, row))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

##------------------------------------------------------------------------------
def rotate_image(image, angle_radians):
#def rotate_image(image, angle_degree):
    
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """
    # # convert angle in degrees to radians 
    # angle_radians = math.radians(angle_degree)
        
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle_radians, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

##------------------------------------------------------------------------------
def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

##------------------------------------------------------------------------------
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (  bb_w - 2 * x,   bb_h - 2 * y  )

##------------------------------------------------------------------------------
def rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """
    angle = math.radians(angle)
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)

##------------------------------------------------------------------------------
def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

##------------------------------------------------------------------------------
def crop_to_shape(img, w, h):
    x, y = int(img.shape[1] * .5), int(img.shape[0] * .5)

    return img[
        int(np.ceil(y - h * .5)) : int(np.floor(y + h * .5)),
        int(np.ceil(x - w * .5)) : int(np.floor(x + h * .5))
    ]
##------------------------------------------------------------------------------

# def demo():
#     """
#     Demos the largest_rotated_rect function
#     """

#     image = cv2.imread("lenna_rectangle.png")
#     image_height, image_width = image.shape[0:2]

#     cv2.imshow("Original Image", image)

#     print( "Press [enter] to begin the demo")
#     print ("Press [q] or Escape to quit")

#     key = cv2.waitKey(0)
#     if key == ord("q") or key == 27:
#         exit()

#     for i in np.arange(0, 360, 0.5):
#         image_orig = np.copy(image)
#         image_rotated = rotate_image(image, i)
#         image_rotated_cropped = crop_around_center(
#             image_rotated,
#             *largest_rotated_rect(
#                 image_width,
#                 image_height,
#                 math.radians(i)
#             )
#         )

#         key = cv2.waitKey(2)
#         if(key == ord("q") or key == 27):
#             exit()

#         cv2.imshow("Original Image", image_orig)
#         cv2.imshow("Rotated Image", image_rotated)
#         cv2.imshow("Cropped Image", image_rotated_cropped)

#     print ("Done")


# if __name__ == "__main__":
#     demo()
    
#%%

    # Find the direction of the roation 
def rotation_direction(filename, table, y_max, y_min, x_max, x_min):
    # Define direction of rotation 
    # Output: 
    #   rotation_angle: variable in degrees
    #   head_center (x,y)
    #   tail_center (x,y)
    #   rotation_direction: changes according to the direction the fish is facing
    
    fish_x_center = (x_min + x_max )/2
    fish_y_center = (y_min + y_max )/2 
     
        # Define coordinates where min and max values of table   
    min_x_values = table[(table['X']== x_min)]
    max_x_values = table[(table['X']== x_max)]
    
    min_y_values = table[(table['Y']== y_min)]
    max_y_values = table[(table['Y']== y_min)]
    
  #-----------
    # If fish is facing the left  
    if  filename == 'image_paths_left':
        
        # Set co-ordinates for middle of head and tail at the minimum and maximum X axis values respectively       
        head_center =  (x_min, ( min_x_values.Y.min() + (min_x_values.Y.min() ) /2 ))       
        tail_center =  (x_max, ( max_x_values.Y.min() + (max_x_values.Y.min() ) /2 ))  
        
        # If the tail is higher than the head       
        if tail_center[1] <= head_center[1] :
            # Calculate angle of roation in radians  
               rotation_angle = (0 - np.degrees(math.atan(abs(tail_center[1] - head_center[1])/abs(x_max - x_min))))
               rotation_direction = -1
               print ( 'angle = ' + str(math.degrees(rotation_angle)) + ' degrees' + '\n' + 
                       'Tail is on the right and higher than head of fish so rotate clockwise' )
               
        
        # If the tail is lower than the head
        else:
            #(max_x_values.Y.min() + (max_x_values.Y.min()) /2) > (min_x_values.Y.min() + (min_x_values.Y.min()) /2):
               # Calculate angle of roation in radians 
               rotation_angle = (0 + np.degrees(math.atan(abs(tail_center[1] - head_center[1])/abs(x_max- x_min))))
               rotation_direction = 1
               print ('angle = ' + str(math.degrees(rotation_angle)) + ' degrees' + '\n' + 
                        'Tail is on the right and lower than head of fish so rotate anticlockwise' )
                              
  #--------  
    # If fish is facing the right
    elif filename == 'image_paths_right':   
                   
       # Set co-ordinates for middle of head and tail at the maximum and minimum X axis values respectively 
        head_center =  (x_max, ( max_x_values.Y.min() + (max_x_values.Y.min() ) /2 ))     
        tail_center =  (x_min, ( min_x_values.Y.min() + (min_x_values.Y.min() ) /2 ))  
           
        if tail_center[1] <= head_center[1] :
            # Calculate angle of roation in radians  
               rotation_angle = (0 + np.degrees(math.atan(abs(tail_center[1] - head_center[1])/abs(x_max - x_min))))
               rotation_direction = 1
               print ( 'angle = ' + str(math.degrees(rotation_angle)) + ' degrees' + '\n' +
                       'Tail is on the left and higher than head of fish so rotate anticlockwise' )
        
        # If the tail is lower than the head
        else:
            #(max_x_values.Y.min() + (max_x_values.Y.min()) /2) > (min_x_values.Y.min() + (min_x_values.Y.min()) /2):
               # Calculate angle of roation in radians 
               rotation_angle = (0 - np.degrees(math.atan(abs(tail_center[1] - head_center[1])/abs(x_max- x_min))))
               rotation_direction = -1
               print ( 'angle = ' + str(math.degrees(rotation_angle)) + ' degrees' + '\n' + 
                       'Tail is on the left and lower than head of fish so rotate clockwise' )
                                 
    # If the direction the fish is facing is not defined     
    else:
            print('Error finding direction fish is facing and subsequent direction of rotation')
                 
    return ( rotation_angle, head_center, tail_center, rotation_direction )
    
#%%

# increase brightenss, contrast and sharpen tools 

''' 
https://www.geeksforgeeks.org/python-pil-imageenhance-brightness-and-imageenhance-sharpness-method/
https://holypython.com/python-pil-tutorial/how-to-adjust-brightness-contrast-sharpness-and-saturation-of-images-in-python-pil/

other useful links
https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_filter_coins.html
https://stackoverflow.com/questions/67075243/applying-multi-otsu-threshold-for-my-image
'''

#%%
# image = skimage.io.imread('C:/Users/Hande/Desktop/Code_testing/Experiment_2/Uninjected/Fish_1/Left_rotated.jpg')

# rotated_2 = rotate_image(image, -10)
# fig, ax = plt.subplots()
# plt.imshow(rotated_2, cmap='gray')
# plt.title('rotated_image')
# plt.show()

# #%%

# result_width , result_height = rotatedRectWithMaxArea(1733, 2311, -0.17453292519943295)

# image_cropped = crop_around_center(rotated_2, result_height, result_width)
# fig, ax = plt.subplots()
# plt.imshow(image_cropped, cmap='gray')
# plt.title('rotated_and_cropped_image')
# plt.show()













