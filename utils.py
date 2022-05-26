# All Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from random import shuffle

# Loading Functions
def load_mnist():
    """Loads mnist dataset.

    Args:
        None

    Returns:
        The mnist images and its labels formatted to 
        (num of instance, heightm width)
    """
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    return np.concatenate((X_train, X_test), axis=0), np.concatenate((Y_train, Y_test), axis=0)


# Plotting Tools
def plot_image(data, title = "Image"):
    """Plots image num in the list
    
    Args:
        data: image in format (image width, image height)
        img_num: The sample number in the dataset as int

    Returns:
        Plot of the image
    """
    fig = plt.figure
    plt.title(title)
    plt.imshow(data, cmap='gray')
    
    return plt.show() 

def shuffle_plot(img, height=2, width=2):
      """Plots an image with the same image with its
        patches shuffled

      Args:
          img: The image to be shuffled format (image width, image height)
          height: Height of the patch as int
          width: Width of the patch as int

      Returns:
          The plot figure
      """

      s = image_shuffle(img, height, width)

      plot_image(img, title="UnShuffled")
      plot_image(s, title="Shuffled")

def plot_all(data):
    """Plots all images in dataset

    Args:
        data: Dataset containing confidence maps (num maps, width, height)

    Returns:
        None
    """

    for i in range(np.shape(data)[0]):
        plot_image(data[i], title="Confidence Map " + str(i))


# Functions to create confidence maps
def make_sampling_vector(width, height, stride):
    """Create sampling vectors that define a grid.

    Args:
        width: Width of grid as int
        height: Height of grid as int
        stride: Steps between each sample
    
    Returns:
        A pair of vectors xv, yv that define the grid.
    """
    xv = np.arange(0, width, stride, dtype="float32")
    yv = np.arange(0, height, stride, dtype="float32")
    return xv, yv


def make_confidence_map(x, y, xv, yv, sigma=1):

    """Make confidence maps for a point.

    Args:
        x: X-coordinate of the center of the confidence map.
        y: Y-coordinate of the center of the confidence map.
        xv: X-sampling vector.
        yv: Y-sampling vector.
        sigma: Spread of the confidence map.
    
    Returns:
        A confidence map centered at the x, y coordinates specified as
        a 2D array of shape (len(yv), len(xv)).
    """

    cm = np.exp(-(
    (xv.reshape(1, -1) - x) ** 2 + (yv.reshape(-1, 1) - y) ** 2
    ) / (2* sigma ** 2))
    return cm



def generate_patch_cm(img, height, width, stride= 1, sigma=2):
    """Make confidence maps for a number of patches

    Args:
        img: Image to be processed format (width, height)
        height: Height of the patch as int
        width: Width of the patch as int
        sigma: Spread of the confidence map.
    
    Returns:
        A 3D-numpy array of confidence map centered at each patch
    """
    img_width = np.shape(img)[0]
    img_height = np.shape(img)[1]
  
    w = int(img_width/width)
    h = int(img_height/height)

    xv, yv = make_sampling_vector(img_width, img_height, stride)

    retval = []

    for i in range(height):
      for j in range(width):
          retval.append(make_confidence_map((w/2)+w*(j), (h/2)+h*(i) , xv, yv, sigma=sigma))
         
    return np.array(retval)

# Shuffling tools
def image_shuffle(img, height = 2, width = 2, return_order=False):
    """"Shuffles the image into with a height by width grid

    Args:
        data: Image to be shuffled format (image width, image height)
        height: Height of the patch as int
        width: Width of the patch as int
        return_order: if order of shuffle needs to be returned

    Returns:
        Shuffled image (with order as a 1-dim array if necessary) formatted to 
        (height, width) 
    """ 

    img_width = np.shape(img)[0]
    img_height = np.shape(img)[1]
    
    w = int(img_width/width)
    h = int(img_height/height)

    
    patches = {}
    order = []

    p = 0
    for i in range(height):
      for j in range(width):

        patches[str(p)] = img[w*(j):w*(j+1), h*(i):h*(i+1)]

        order.append(p)
        p += 1

    shuffle(order)

    iter = 0
    shuffled = np.array([])

    for i in range(height):
      rows = patches[str(order[iter])]
      iter += 1

      for j in range(1, width):
        rows = np.concatenate((rows, patches[str(order[iter])]), axis=1)
        iter += 1
      
      if(iter == width):
        shuffled = rows

      else:
        shuffled = np.concatenate((shuffled, rows), axis=0)

    if(return_order):
          return shuffled, order
    return shuffled

# Shuffling and Reassembling Functions
def generate_shuffled(data, patch_dim=2, return_label=False):
    """Generates Shuffled Image set for data

    Args:
        data: the image set to be shuffled (num images, img width, image height)
        patch_dim: the dimensions of the patch as int
        return_label: whether ordered labels are returned

    Returns:
        Shuffled image dataset (with labels) formatted to
        (num images, height, width)
    """
    shuffled = []
    cm = []
    for i in range(np.shape(data)[0]):
          order = {}
          s, o = image_shuffle(data[i], height=patch_dim, width=patch_dim, return_order=True)
          shuffled.append(s)
          c = generate_patch_cm(data[i], patch_dim, patch_dim)

          place = 0
          for j in o:
              order[str(j)] = c[place]
              place += 1

          cms = []
          for cnt in range(patch_dim*patch_dim):
              cms.append(order[str(cnt)])
          
          cm.append(cms)

    
    if(return_label):
        return shuffled, cm
    return cm

def reassembler(img, cms, height=2, width=2):
    """Reconstructs image according to confidence maps

    Args:
        img: Shuffled image formatted to (image width, image height)
        cms: confidence maps formatted to (num maps, width map, height map)
        height: height of each patch as int
        width: witdh of each patch as int

    Returns:
        Reconstructed Image as a 2dim array
    """
    reconstructed = []
    order = []

    cords = create_grid(cms[0], height, width)
    patches = split_patches(img, height, width)

    for sz in range(np.shape(cms)[0]):
        x,y = np.unravel_index(cms[sz].argmax(), cms[sz].shape)

        for pos in range(np.shape(cords)[0]):
            if(x <=  cords[pos][0] and y <=  cords[pos][1]):
                  order.append(pos)
                  break
        
    for o in order:
        reconstructed.append(patches[o])
    
    reconstructed = np.array(reconstructed)
    rec = np.array([])

    for i in range(height):
        row = np.array([])

        for j in range(width):
            if(j == 0):
                row = reconstructed[i+width*j]
            else:
                row = np.concatenate((row, reconstructed[i+width*j]), axis=1)
            
        if(i == 0):
          rec = row
        else:
          rec = np.concatenate((rec, row), axis=0)

    return rec 

def create_grid(img,height, width):
    """Creates grid cordinates for each patch

    Args:
        img: image data formatted to (width, height)
        height: height of each patch
        width: witdh of each patch

    Returns:
        array of patch indecies (num patches, upper x, upper y)
    """

    cords = []

    img_width = np.shape(img)[0]
    img_height = np.shape(img)[1]
    
    w = int(img_width/width)
    h = int(img_height/height)

    for i in range(height):
      for j in range(width):
        cords.append([w*(j+1),h*(i+1)])

    return np.array(cords)


def split_patches(img,height, width):
    """splits shuffled image into patches

    Args:
        img: shuffled image data formatted to (width, height)
        height: height of each patch
        width: witdh of each patch

    Returns:
        array of patch as (num patches, width, height)
    """
    
    img_width = img.shape[0]
    img_height = img.shape[1]
    
    w = int(img_width/width)
    h = int(img_height/height)

    
    patches = []

    for i in range(height):
      for j in range(width):

        patches.append(img[w*(j):w*(j+1), h*(i):h*(i+1)])

    return patches

# helper functions
def find_max(cm):
    """Finds the coordinates of confidence map peak

    Args:
        cm: the confidence map formatted to (width, height)

    Returns:
        coordinates of peak as (x, y)
    """
    cords = np.unravel_index(cm.argmax(), cm.shape)
    return cords
