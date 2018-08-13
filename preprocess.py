from PIL import Image
import numpy as np
import cv2


def rgb_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 

def resize(orig_img, new_width, new_height):
    return cv2.resize(orig_img, (new_width, new_height))

def get_initial_state(state, num_of_states, 
        output_width, 
        output_height):
    processed_image = resize(rgb_to_gray(state), output_width, output_height)
    return [processed_image for _ in range(num_of_states)]
    
def preprocess_img(state, new_size):
    I = Image.fromarray(state, 'RGB')
    I = I.convert('L')
    I = I.resize((new_size, new_size), Image.ANTIALIAS)
    I = np.array(I).astype('uint8')
    return I