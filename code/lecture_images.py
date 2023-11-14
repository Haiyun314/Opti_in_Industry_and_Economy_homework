import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

path = '../../lecture_images/'
items = os.listdir(path)[1:]

def clear_image(image, image_shape):
    assert isinstance(image, np.ndarray), 'image type should be a numpy array'
    assert len(image_shape) == 2, 'image_shape is not correct'
    image_shape.reverse()
    modified_image = np.zeros(image_shape)
    for i in range(2, image_shape[0] -2):
        for j in range(image_shape[1]):
            if image[i + 1, j] - image[i -2, j] > 100:
                modified_image[i, j] = 255
            else:
                modified_image[i, j] = 0
    return modified_image

def show_image(number_of_images, *args):
    assert number_of_images > 0, 'number of images is invalid'

    fig, ax = plt.subplots(1, number_of_images)
    if number_of_images == 1:
        ax.imshow(args[0])
        ax.set_title(str(0))
        plt.show()
        plt.close()
    else:
        for i in range(number_of_images):
            ax[i].imshow(args[i])
            ax[i].set_title(str(i))
        plt.show()
        plt.close()



for item in items:
    try:
        image = cv2.imread(path + item)
    except Exception:
        raise 'not able to read the image'
    image = np.array(image)
    image_shape = [int(i/3) for i in list(np.shape(image)[:2])][::-1] #change the shape to the third of original length and width
    resized_image = cv2.resize(image, image_shape)
    # src = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    # ret, resized_image = cv2.threshold(src, 125, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((5, 5), np.uint8)
    # morphology = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=2)
    # cv2.imwrite(item, resized_image)
    



