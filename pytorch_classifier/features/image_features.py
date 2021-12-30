from numpy import empty
import torch
import cv2 as cv

detector = cv.ORB_create()


class ToDescriptor:
    '''
    Convert the image to a descriptor vector
    '''

    def __init__(self, num_of_desc):
        self.num_of_desc = num_of_desc

    def __call__(self, image_tensor):
        descriptors = get_tensor_img_desc(image_tensor)
        if descriptors.shape[0] < self.num_of_desc:
            dif = self.num_of_desc - descriptors.shape[0]
            descriptors = torch.cat((descriptors, torch.zeros(
                dif, descriptors.shape[1])), 0)
        return descriptors[:self.num_of_desc]

def get_numpy_image(img_tensor):
    # numpy image: H x W x C
    # torch image: C x H x W
    img = torch.transpose(img_tensor, 0, 1)
    return torch.transpose(img, 1, 2)


def get_tensor_img_desc(img_tensor):
    img_tensor = get_numpy_image(img_tensor)
    return get_image_descriptor_vec(img_tensor.cpu().detach().numpy())


def get_image_descriptor_vec(img):
    '''
    Returns a descriptor tensor for the image.
    Each keypoint in the image got its descriptor vector, 
    thus the output feature tensor of the image is a 2d tensor.
    '''
    kp = detector.detect(img, None)
    _, des = detector.compute(img, kp)
    return torch.zeros((32, 32)) if des is None else torch.tensor(des)


# def show_image_tensor(img_tensor):
#     img_tensor = get_numpy_image(img_tensor)
#     cv.imshow('name', img_tensor.cpu().detach().numpy())
#     cv.waitKey(0)
