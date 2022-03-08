import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

import copy
import torch
from typing import Iterable
from Configure import preprocess_config

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE

    ### END CODE HERE

    #image = preprocess_image(image, training) # If any.
    pass
    #return image


def preprocess_image(preprocess_config):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    preprocess_train = transforms.Compose([])

    if preprocess_config['crop']:
        preprocess_train.transforms.append(transforms.RandomCrop(32, padding=preprocess_config['crop_padding']))

    if preprocess_config['flip']:
        preprocess_train.transforms.append(transforms.RandomHorizontalFlip())

    preprocess_train.transforms.append(AutoAugmentCIFAR())
    preprocess_train.transforms.append(transforms.ToTensor())
    preprocess_train.transforms.append(normalize)

    if preprocess_config['cutout']:
        c_holes = preprocess_config["cutout_holes"]
        c_length = preprocess_config["cutout_length"]
        preprocess_train.transforms.append(Cutout(n_holes=c_holes, length=c_length))

    preprocess_test = transforms.Compose([transforms.ToTensor(),normalize]) 

    return preprocess_train, preprocess_test
    ### END CODE HERE


# Other functions
### YOUR CODE HERE

class Cutout(object):
    """
    Create cutout in images by masking out random squares of given length.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, image):
        """
        Image with n_holes of dimension length x length cut out of it.
        """
        mask = np.ones((image.size(1), image.size(2)), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(image.size(1))
            x = np.random.randint(image.size(2))

            y1 = np.clip(y - self.length // 2, 0, image.size(1))
            y2 = np.clip(y + self.length // 2, 0, image.size(1))
            x1 = np.clip(x - self.length // 2, 0, image.size(2))
            x2 = np.clip(x + self.length // 2, 0, image.size(2))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask

        return image

class AutoAugmentCIFAR(object):
   
    def __init__(self):
        self.policies = [
            sub_policy("invert", 0.1, 7, "contrast", 0.2, 6),
            sub_policy("sharpness", 0.8, 1, "sharpness", 0.9, 3),
            sub_policy("autocontrast", 0.5, 8, "equalize", 0.9, 2),

            sub_policy("color", 0.4, 3, "brightness", 0.6, 7),
            sub_policy("sharpness", 0.3, 9, "brightness", 0.7, 9),
            sub_policy("equalize", 0.6, 5, "equalize", 0.5, 1),
            sub_policy("contrast", 0.6, 7, "sharpness", 0.6, 5),

            sub_policy("equalize", 0.3, 7, "autocontrast", 0.4, 8),
            sub_policy("brightness", 0.9, 6, "color", 0.2, 8),
            sub_policy("solarize", 0.5, 2, "invert", 0.0, 3),

            sub_policy("equalize", 0.2, 0, "autocontrast", 0.6, 0),
            sub_policy("equalize", 0.2, 8, "equalize", 0.6, 4),
            sub_policy("color", 0.9, 9, "equalize", 0.6, 6),
            sub_policy("autocontrast", 0.8, 4, "solarize", 0.2, 8),
            sub_policy("brightness", 0.1, 3, "color", 0.7, 0),

            sub_policy("solarize", 0.4, 5, "autocontrast", 0.9, 3),
            sub_policy("autocontrast", 0.9, 2, "solarize", 0.8, 3),
            sub_policy("equalize", 0.8, 8, "invert", 0.1, 3),
            sub_policy("sharpness", 0.2, 6, "autocontrast", 0.9, 1)
        ]
        

    def __call__(self, image):
        return random.choice(self.policies)(image)


class sub_policy(object):

    def __init__(self, tranform1, prob1, id1_magnitude, tranform2, prob2, id2_magnitude):

        operations = {
            "autocontrast":
                lambda image, val: ImageOps.autocontrast(image),

            "brightness":
                lambda image, val: ImageEnhance.Brightness(image).enhance(1 + val * random.choice([-1, 1])),

            "color":
                lambda image, val: ImageEnhance.Color(image).enhance(1 + val * random.choice([-1, 1])),

            "contrast":
                lambda image, val: ImageEnhance.Contrast(image).enhance(1 + val * random.choice([-1, 1])),

            "equalize":
                lambda image, val: ImageOps.equalize(image),

            "invert":
                lambda image, val: ImageOps.invert(image),
            "posterize":
                lambda image, val: ImageOps.posterize(image, val),

            "sharpness":
                lambda image, val: ImageEnhance.Sharpness(image).enhance(1 + val * random.choice([-1, 1])),

            "solarize":
                lambda image, val: ImageOps.solarize(image, val)
        }

        ranges = {
            "autocontrast": [0] * 10,
            "brightness": np.linspace(0.0, 0.9, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "equalize": [0] * 10,
            "invert": [0] * 10,
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "solarize": np.linspace(256, 0, 10)
        }

        self.prob1 = prob1
        self.tranform1 = operations[tranform1]
        self.magnitude1 = ranges[tranform1][id1_magnitude]

        self.prob2 = prob2
        self.tranform2 = operations[tranform2]
        self.magnitude2 = ranges[tranform2][id2_magnitude]


    def __call__(self, image):
        
        if random.random() < self.prob1:
            #pass
            image = self.tranform1(image, self.magnitude1)
        
        if random.random() < self.prob2:
            #pass
            image = self.tranform2(image, self.magnitude2)
        
        return image


### END CODE HERE