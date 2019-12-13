import torchvision.transforms.functional as TF
import cv2
import numpy as np
class Resize():

    def __init__(self, scale, resize_label=True):
        self.scale = scale
        self.resize_label = resize_label

    def __call__(self, args):
        input, label = args

        input = cv2.resize(input, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.resize_label:
            label = cv2.resize(label, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        return input, label

class PIL2Torch():
    def __call__(self, args):
        input, label = args
        # BGR to RGB
        input = input[..., [2,1,0]]
        input_t = TF.to_tensor(input)
        return input_t, label

class UniformCrop():
    '''
    Performs uniform cropping on numpy images (cv2 images)
    '''
    def __init__(self, crop_size):

        self.crop_size = crop_size
    def random_crop(self, input, label):
        image_size = input.shape[-2]
        crop_limit = image_size - self.crop_size
        x, y = np.random.randint(0, crop_limit, size=2)

        input = input[y:y+self.crop_size, x:x+self.crop_size, :]
        label = label[y:y+self.crop_size, x:x+self.crop_size]

        return input, label

    def __call__(self, args):
        input, label = args
        return self.random_crop(input, label)


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, args):
        input, label = args

        SAMPLE_SIZE = 5 # an arbitrary number that I came up with
        BALANCING_FACTOR = 200

        random_crops = [self.random_crop(input, label) for i in range(SAMPLE_SIZE)]
        crop_weights = np.array([input.sum() for input, label in random_crops]) + BALANCING_FACTOR
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(SAMPLE_SIZE, p=crop_weights)
        input, label = random_crops[sample_idx]

        return input, label