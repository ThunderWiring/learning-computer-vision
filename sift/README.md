Resources i used to learn about SIFT:

SIFT original paper: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
High level overview: http://weitz.de/sift/index.html?size=large
Low level analysis: http://www.ipol.im/pub/art/2014/82/article.pdf

This is an educational project aims at implementing SIFT.

SIFT algorithm extracts features from image (keypoints + descriptors) in order to be able to identify this image from different viewpoints.

## Why SIFT?
As a beginner in computer vision, this algorithm covers a lot of basic concepts, like keypoints, gradients, histograms, ... etc. i found it extremly helpful to better understand those concepts and built an intuition regarding how the computer descripe an image.

## Results
Visualizing the results was a bit harder, the drawing functionality didn't provide accurate results, this is something that i'm still working on.

For example, here are the keypoints for this image:
```[34.8005, 51.1822]
[155.767, 102.81]
[92.9017, 247.396]
[192.355, 110.826]
[172.631, 160.276]
[23.4936, 182.052]
[10.7941, 216.988]
[14.3789, 218.514]
[206.365, 219.442]
[139.654, 54.7965]
[169.579, 115.089]
[165.44, 160.867]
[121.738, 52.737]
[135.911, 69.0487]
[178.107, 86.0004]
[197.191, 50.2803]
[109.824, 58.7346]```
And this is how opencv drew them:
![alt text](https://github.com/ThunderWiring/learning-computer-vision/blob/main/images/keypoints.JPG?raw=true)