# Face-Mask-Detection-Project
Using Machine Learning concepts to design several models that determine whether an individual is wearing a face mask, not wearing a face mask, or wearing a face mask incorrectly. We will use the performance metrics gathered on these models (including accuracy, precision, recall, and f1 scores) to determine which model performs the best. The models explored include KNN, Decision Trees, CNN, Naive Bayes, and SVM. 

<h1 align="center">Face Mask Detection</h1>
<div align= "center"><img src="https://github.com/Vrushti24/Face-Mask-Detection/blob/logo/Logo/facemaskdetection.ai%20%40%2051.06%25%20(CMYK_GPU%20Preview)%20%2018-02-2021%2018_33_18%20(2).png" width="200" height="200"/>

# Overview

<p align="center"><img src=https://2ocjot45j55j36bcug1rfm7z-wpengine.netdna-ssl.com/wp-content/uploads/2020/05/blog-002.jpg" width="700" height="400"></p>

## Dataset 
We combined 2 different datasets together to form a three class classification dataset with *face mask*, *no face mask*, or *incorrect face mask wear* categories. This new dataset containes over 17,000 images. As for the dataset we used to test bias, we created a W (white) and POC (people of color) dataset. Since there was not an existing dataset available, we obtained the images for W and POC by creating our own dataset collections, each containing around 100-200 image samples. 

- https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
- https://github.com/cabani/MaskedFace-Net



## Frame Works Used
[OpenCV](https://opencv.org/)

[Keras](https://keras.io/)

[TensorFlow](https://www.tensorflow.org/)


## Features
Our face mask detector doesn't use any morphed masked images dataset and the model is accurate. Owing to the use of MobileNetV2 architecture, it is computationally efficient, thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.).


## All the dependencies and required libraries listed below:
Installation; pip install the following:
- Tensorflow
- Keras
- imutils
- numpy
- opencv-python>=4.22
- matplotlib
- argparse
- scipy
- scikit-learn
- pillow
- streamlit
- onnx
- tf2onnx

## Create a Python virtual environment named 'test' and activate it

## Results

`KNN`
The optimal parameters resulted in 86.7% accuracy, and performed lower when dimensionality reduction techniques were included.

`Decision Tree`
The optimal parameters resulted in 84% accuracy, and performed lower when dimensionality reduction techniques were included.

`SVM`
Reached an accuracy of ~93% with our testing dataset.

`Naive Bayes`
Because of naive bayes assumption and how much pixels correlate with each other, it performed with ~68% for multinomial and ~77% for Gaussian.

`CNN`
Best performing: Reached an accuracy of ~98% with our testing data. 


## Conclusion
In conclusion our best performing model was CNN, achieving 98% testing accuracy. Models that were comparable to our baseline paper underperformed by 11-16%. In regards to our POC and W datasets, all models also underperformed dramatically, with an emphasis on poorer results for the W dataset. We suspect differences in image perspectives and dataset quality led to this disparity.


## Credits
* [https://www.pyimagesearch.com/](https://www.pyimagesearch.com/)
* [https://www.tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
