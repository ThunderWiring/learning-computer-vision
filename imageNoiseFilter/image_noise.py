import cv2
import re
from imageUtils.image_scrapper import get_image_srcs, slice_img
import random
import numpy as np
import torch
import torch.nn.functional as F

def apply_sap_noise(img, snr = 0.9):
  '''
  apply SaP noise to the passed image (np array)
  '''
  h=img.shape[0]
  w=img.shape[1]
  img1=img.copy()
  sp=h*w # Calculate the number of image pixels
  NP=int(sp*(1-snr))
  for i in range (NP):
    randx=np.random.randint(1,h-1) # Generate a random integer between 1 and h-1
    randy=np.random.randint(1,w-1) # Generate a random integer between 1 and w-1
    if np.random.random()<=0.5: # np.random.random() generates a floating point number between 0 and 1.
      img1[randx,randy]=0
    else:
      img1[randx,randy]=255
  return img1


def prepare_image_noise_dataset():
  return []

def get_dataset_image():
  '''
  prepares a dataset containing 28x28px images with SaP noise as the input, and the
  image without the noise as the label.
  The set contains both training and test sets
  '''

  keywords = ['car'
  , 'cat', 'horse', 'computer', 'ship', 'racoon', 'worm'
  ,'batterfly', 'film', 'camera', 'laptop', 'fish', 'helmet', 'bottle', 'watch',
  'clothes', 'phone', 'bear', 'dog', 'bag', 'guitar', 'microphone', 'book', 'rose',
  'bus', 'pen', 'goat', 'tractor', 'toddler', 'elephant'
  ,'Chinese'
  ,'French'
  ,'German'
  ,'Hebrew'
  ,'Hindi'
  ,'Italian'
  ,'Japanese'
  ,'Korean'
  ,'Portuguese'
  ,'Spanish'
  ]
     
  print('preparing the dataset')
  dataset_images = []
  for q in keywords:
    dataset_images += get_image_srcs(q)
  return dataset_images


def scale_image(img, factor):
  return img

def get_rotated_image(img):
  rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
  rot180 = cv2.rotate(rot90, cv2.ROTATE_90_CLOCKWISE)
  rot270 = cv2.rotate(rot180, cv2.ROTATE_90_CLOCKWISE)
  return [rot90, rot180, rot270]

def get_augmented_image(img, shape=None):
  '''
  returns the augmentations of an image.
  Each image will be rotated few times and resized.
  '''
  if img.ndim != 2:
    l = np.sqrt(np.size(img))
    new_shape = (l,l) if shape is None else shape
    img = np.reshape(img, new_shape)
  # return [scale_image(r_img) for r_img in get_rotated_image(img)]
  return get_rotated_image(img)

class SaP_dataset:
  '''
  Prepares a dataset for the salt and pepper (SaP) model
  '''
  def __init__(self):
    self.training_images = []
    self.training_labels = []
    self.test_images = []
    self.test_labels = []
    self._images = get_dataset_image()

  def prepare(self):
    slices = []
    for img in self._images:
      slices += slice_img(img, (28,28) )

    noisey_slices = []
    for img in slices:
      noisey_slices += [apply_sap_noise(img)]

    # a random 10% of the slices will be reserved for testing and the rest for training.
    test_set_size = int(0.1 * len(slices))
    test_imgaes_indices = [random.randint(0, len(slices)) for _ in range(test_set_size)]
    train_imgaes_indices = [i for i in range(len(slices)) if i not in test_imgaes_indices]

    self.training_images = [apply_sap_noise(slices[i]) for i in train_imgaes_indices]
    self.training_labels = [slices[i] for i in train_imgaes_indices]
    self.test_images = [apply_sap_noise(slices[i]) for i in test_imgaes_indices]
    self.test_labels = [slices[i] for i in test_imgaes_indices]
    return self

  def augment_image_dataset(self):
    '''
    Expands the dataset by augmenting the current images
    '''
    train_img_label = []
    for img in self.training_labels:
      train_img_label += get_augmented_image(img)
    
    train_img_in = [apply_sap_noise(aug_img) for aug_img in train_img_label]
    self.training_images += train_img_in
    self.training_labels += train_img_label
  
  def get_test_data(self):
    test_images_t = torch.zeros(len(self.test_images),784)
    test_labels_t = torch.zeros(len(self.test_labels),784)

    for idx, img in enumerate(self.test_images):
      in_tensor = torch.flatten(torch.tensor(img))
      lbl_tensor = torch.flatten(torch.tensor(self.test_labels[idx]))
      test_images_t[idx] = in_tensor #F.normalize(in_tensor, dim=0)
      test_labels_t[idx] = lbl_tensor #F.normalize(lbl_tensor, dim=0)
    
    return test_images_t, test_labels_t
  
  def get_training_baches(self, num_of_batches=100):
    '''
    Returns the training data segmented into batches
    '''
    total = len(self.training_images)
    batch_size = int(total/num_of_batches)
    remain = total - batch_size  * num_of_batches
    print('calculating training batches: tota', total, 'batch_size', batch_size)

    
    # each training image is of size 28x28=784
    # each batch will contain batch_size such images
    # in total, there are `total` batches
    # the output of this method should be a rank-3 tensor
    train_batches = torch.zeros(num_of_batches, batch_size, 784)
    label_batches = torch.zeros(num_of_batches, batch_size, 784)
    
    shuffled_idx = [i for i in range(total)]
    random.shuffle(shuffled_idx)

    for i in range(num_of_batches):
      train_batch = train_batches[i]
      training_label_batch = label_batches[i]
      start = i * batch_size
      idxs = shuffled_idx[start: start + batch_size]
      for batch_ele_idx in range(batch_size):
        image_idx = idxs[batch_ele_idx]
        in_tensor = torch.tensor(self.training_images[image_idx]) 
        lbl_tensor = torch.tensor(self.training_labels[image_idx])
        train_input_tensor = in_tensor #F.normalize(in_tensor , dim=0)
        train_label_tensor = lbl_tensor#F.normalize(lbl_tensor, dim=0)
        train_batch[batch_ele_idx] = torch.flatten(train_input_tensor)
        training_label_batch[batch_ele_idx] = torch.flatten(train_label_tensor)
      
    return train_batches, label_batches







    
  
  












