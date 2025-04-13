import numpy as np
import cv2

def load_image(path):
    """
    This function loads an image from a given path and converts it to RGB format.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image at path {path} could not be loaded.")
    return img

def make_black_and_white(img, r_difference, g_difference, b_difference):
    """
    This function takes an image and converts it to black and white based on the differences
    between the red, green, and blue channels. A negative value for the difference
    selects the channel that the other two channels are compared to. Example:
    r_difference = -1, g_difference = 10, b_difference = 20 means that pixels that
    have a red value greater than the green value + 10 and the blue value + 20
    will be set to white, and the rest will be set to black.
    """
    img_blue = img[:,:,0]
    img_green = img[:,:,1]
    img_red = img[:,:,2]
    assert (r_difference <0) ^ (g_difference < 0) ^ (b_difference < 0), "At least one difference must be negative"
    if(r_difference < 0):
        indexer = ((img_red > img_green + g_difference ) & 
                                (img_red > img_blue + b_difference))
    elif(g_difference < 0):
        indexer = ((img_green > img_red + r_difference ) & 
                                (img_green > img_blue + b_difference))
    elif(b_difference < 0):
        indexer = ((img_blue > img_red + r_difference ) & 
                                (img_blue > img_green + g_difference))
        
    black_and_white = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    black_and_white[indexer] = 255
    return black_and_white

def find_biggest_obj(img):
    """
    This function takes a black and white image and finds the biggest object in it.
    Returns a black and white image with the biggest object in white
    and the rest in black.
    """
    assert len(img.shape) == 2, "Image must be black and white"
    assert img.dtype == np.uint8, "Image must be of type uint8"
    assert img.max() == 255, "Image must be black and white"
    assert img.min() == 0, "Image must be black and white"
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # canvas = np.zeros((h, w), dtype=np.uint8)
    # cv2.drawContours(canvas, [biggest], -1, (255), 1)
    biggest_index = max(enumerate(contours), key=lambda x: cv2.contourArea(x[1]))[0]
    biggest = contours[biggest_index]
    x, y, w, h = cv2.boundingRect(biggest)
    biggest = biggest - [x, y]

    canvas = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(canvas, [biggest], -1, (255), -1)

    hole = hierarchy[0, biggest_index,2]
    while hole > 0:
        cv2.drawContours(canvas, [contours[hole]- [x, y]], -1, (0), -1)
        hole = hierarchy[0,hole,0]

    return canvas

def resize_and_padd(img, h, w):
    """
    Takes an arbitrary size black and white image and resizes
    it to the given height and width.
    """
    o_height, o_width = img.shape
    
    # first pad to ratio
    new_ratio = h / w
    if(o_height > o_width):
        # height is bigger that width: pad along width
        new_width = int(o_height / new_ratio)
        padding = (new_width - o_width) // 2
        padded = cv2.copyMakeBorder(img,
                           top=0,
                           bottom=0,
                           left=padding,
                           right=padding,
                           borderType=cv2.BORDER_CONSTANT,
                           value=(0))
    else:
        new_height = int(o_width * new_ratio)
        padding = (new_height - o_height) // 2
        padded = cv2.copyMakeBorder(img,
                           top=padding,
                           bottom=padding,
                           left=0,
                           right=0,
                           borderType=cv2.BORDER_CONSTANT,
                           value=(0))
    # then resize to final size
    resized = cv2.resize(padded, (w, h), interpolation=cv2.INTER_AREA)
    return resized

def padd_to_size(img, h, w):
    
    og_h, og_w = img.shape
    top, bottom, left, right = 0, 0, 0, 0
    if(h > og_h ):
        top = (h - og_h) // 2
        bottom = h - og_h - top
    if(w > og_w):
        left = (w - og_w) // 2
        right = w - og_w - left
        
    padded = cv2.copyMakeBorder(img,
                           top=top,
                           bottom=bottom,
                           left=left,
                           right=right,
                           borderType=cv2.BORDER_CONSTANT,
                           value=(0))
    return padded