import requests
import re
import cv2 as cv 
import numpy as np
from PIL import Image

def _get_url(query):
  return 'https://www.google.com/search?tbm=isch&q='+query


def _get_gray_image(url):
  '''
  Returns numpy ndarray for the image from the url
  '''
  image = Image.open(requests.get(url, stream=True).raw)
  try:
    res =  cv.cvtColor(np.array(image), cv.COLOR_BGR2GRAY)
    return res
  except:
    return np.array(image)


def get_image_srcs(query):
  res = requests.get(url=_get_url(query))
  srcs = re.findall('src="https:\/\/[a-zA-Z0-9\-\.\/\?=\:_&;]+"\/>', res.text)
  img_urls = [url.split('"')[1] for url in srcs]
  return [_get_gray_image(url) for url in img_urls]


def slice_img(img_array, new_size):
  '''
  Slices the im_array representing the image to multiple images of the given size
  '''
  img_shape = img_array.shape
  if new_size[0] > img_shape[0] or new_size[1] > img_shape[1]:
    return [cv.resize(image_array, new_size)]
  
  hori_slices = int(img_array.shape[0]/new_size[0])
  vert_slices = int(img_array.shape[1]/new_size[1])
  
  slices = []
  for ver in range(vert_slices):
    for hor in range(hori_slices):
      hor_start = 0 if hor is 0 else hor - 1
      ver_start = 0 if ver is 0 else ver - 1
      slices  += [
        img_array[
          hor_start*new_size[0]:(hor_start+1)*new_size[0], 
          ver_start * new_size[1]: (ver_start+1)*new_size[1]
        ]
      ]
  
  return slices
