# face-recognition thesis
Masters thesis on face recognition using machine and deep learning techniques

The complete thesis text can be found [here](http://hdl.handle.net/10889/12597).

Thesis language: Greek

## Brief explanation in English
This thesis is focused on 2 objectives
- A complete breakdown of the mathematical backbone behind most cutting edge object detection and object classification
techniques as well as some older ones
- Development of a python application which utilizes these concepts and is capable of recognising faces in pictures, videos and real-time webcam video streams. Some performance tweaks have also been made in order to utilize GPU resources more efficiently and to provide faster results for recognising faces in real time.

### Face detection methods available
- Histogram of oriented gradients (HOG) feature descriptors and pre trained SVM classifier with linear sliding windows implementation on the CPU
- Pre trained residual CNN of ResNet34 type with convolutional sliding windows implementation on the GPU

On both of these methods Non Maximum Suppression algorithm is used in order to predict face bounding boxes in images.

### Face classification method

- The CNN architecture used is of type ResNet32 with the number of filters per layer halved for performance. The output layer produces 128-vector encodings making it a siamese network for one shot learning purposes. Transfer learning was used with initialized weights that produced 99.38% accuracy score on the Standard LFW face recognition benchmark.

### Setup

#### Hardware/OS used

- CPU: AMD Ryzen 5 with 6 physical cores, 12 threads, clock @ 3.8 GHz.
- GPU: NVIDIA GeForce GTX 970 with 1664 CUDA cores, clock @ 1174 MHz, 3.5 GB VRAM, 256-bit memory bus.
- RAM: 16 GB DDR4 @ 3200 MHz.
- DISK: SSD.
- OS: Ubuntu 18.04.
- Web Camera: Logitech C270 HD WEBCAM, 720p max resolution, 30 fps max capture rate @ 640x480 resolution.

#### Software used

- CUDA Toolkit 10.1
- cuDNN 7.6
- [dlib](https://github.com/davisking/dlib) with python bindings
- [face_recognition](https://pypi.org/project/face_recognition/)
- OpenCV

You have to make sure DLIB is built with -DLIB_USE_CUDA = 1 (e.g. $ cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1).

### Dataset used info

- 236 images from 7 labels
- Images were collected using google image search and belong mostly to actors playing in Jurrasic Park (1993) movie for testing purposes, as well as 18 images of me.

### Codes

Comments are used inside the python scripts that briefly describe how to use them and what they do

### Demo results
- [lunch scene hog](https://www.youtube.com/watch?v=zwEndBDHaaA&feature=youtu.be)
- [lunch scene cnn](https://www.youtube.com/watch?v=G1qOlcf1Vws&feature=youtu.be)
- [RT serial face recognition hog](https://youtu.be/CKpEE-dsw_8)
- [RT serial face recognition cnn](https://youtu.be/9EsVTHkRb0M)
- [RT parallel face recognition cnn](https://youtu.be/MXyvp5FSaQ4)
