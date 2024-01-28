# Enhancing Catalyst Analysis through Advanced Image Processing: Semantic Segmentation of Metallic Nanoparticles in TEM Imaging

## Introduction

Catalysts, enhancing efficiency and selectivity in chemical reactions, often consist of a metal that acts as the active site for the reaction and a support material that increases the surface area and stability of the metal particles. 
This combination is crucial in various industrial and environmental processes, enabling optimized reaction conditions and outcomes.
Metallic nanoparticles are deposited on inert supports via Supercritical Fluid Reaction Deposition synthesizing catalysts. 
The loaded supports are analyzed in terms of the nanoparticles with a Scanning Transmission Electron Microscope (STEM) at the Laboratory for Electron Microscopy (LEM) at KIT. 
Bright-field (BF) and High-Angle Annular Dark Field (HAADF) images are sampled with the microscope FEI Osiris ChemiStem. 
For a resolution in the range of 0.1 nm, a sample thickness below 50 nm is required. Therefore, only supports with a diameter lower than 50 nm are sufficient for TEM analysis. 
The solid-loaded supports are suspended in demineralized water. Then, the suspension is sprayed on a carbon-coated copper net and dried before the measurement.

The particle size distribution (PSD) of the metallic particles is important for the performance of a catalyst [[1](#references)]. 
The PSD is determined in 4 steps: Transforming the image into a binary image, detecting the Region of Interest (RoI), determining characteristics of the RoI, and calculating the PSD. 
First, a binary image is gained by performing semantic segmentation on HAADF images. In the binary image, the pixels containing metallic particles, the RoI, have the highest value and the rest (background, support, or the carbon-coated copper net) the lowest value (e.g., 255 and 0 for unsigned 8-bit integer, respectively). 
A state-of-the-art algorithm is designed to transform the TEM images into binary images. 
The metallic particles are detected in the binary image, and the pixel-wise area and perimeter information are measured and sampled. 
The pixel-wise length of the scale is determined. 
Reading in the scale with Google’s Tesseract-Optical-Character-Recognition [[2](#references)] using python-tesseract [[3](#references)], the pixel-to-nanometer ratio is obtained. Assuming spherical-shaped particles, the equivalent spherical diameter is calculated. Finally, the PSD of the equivalent spherical diameter is determined.

## Model’s architecture

Semantic segmentation is a pixel-wise classification. In the case of loaded supports, the metallic particles embody the RoI. The class label (e.g., 0 and 1) is assigned to each pixel in the image. 

Since the breakthrough with the AlexNet in 2012 [[4](#references)], deep learning models have been widely used in computer vision for detection, segmentation, and recognition [[5](#references)]. In 2015, a convolution network for biomedical image segmentation was developed [[6](#references)]. This model, called U-Net by its authors, is developed to detect fine details in the image while requiring a few annotated images as ground truth [[7](#references)]. The U-Net outperformed fully convolution networks with the sliding window approach and won the International Symposium on Biomedical Imaging (ISBI) in 2015 [[6](#references)]. The main improvement of the U-Net is the combining of the features gained in the downsampling and upsampling path. The reduction of the image size using max-pooling or convolution reduces the spatial resolution while emphasizing the important structures. Additional information on pooling and convolution arithmetic can be found in [[8](#references)]. In the downsampling path, also called the encoder, the features are extracted using convolution layers and max-pooling operations, reducing the input image's size. In contrast to autoencoders, the feature maps of the encoder and decoder are combined in the upsampling path, also called the decoder. Therefore, the feature maps are upsampled using 2x2 convolution, doubling the spatial size. The resulting feature maps, containing detailed information, are concatenated with feature maps gained from the encoder containing more spatial information. Then, a convolution filter is applied to combine the feature maps. Thus, the spatial information and the emphasized structures are combined. As a regularization method, batch normalization is applied before each activation layer. In the activation layer, a rectified linear unit (ReLu) is used. The ReLu was introduced in 2010 [[9](#references)], possibly named after Relu Patrascu at the University of Toronto. In contrast to the sigmoid function or the tangents hyperbolic, ReLu emphasizes sparsity in the neural network, leading to higher generalization and avoiding the gradient vanishing effect [[10](#references)]. For the next upsampling operation, convolution filter, batch normalization, and ReLu for activation are used. In the original description, no padding is used before applying convolution filters, resulting in a slight reduction of the output [[6](#references)]. For more details, the author refers to the original paper [[6](#references)]. 

For further improvement of the original U-Net [[6](#references)], the encoder part is substituted with more advanced convolution networks. In particular, using deep Residual Networks (ResNet) [[11](#references)] in the encoder part increases the performance of the U-Net. The encoder part benefits from the deeper architecture, increasing the depth and number of the features. ResNets implements residual learning to train large networks, resulting in higher accuracy, less trainable, and faster converging compared to non-residual learning networks such as the VGG16 or VGG19 networks [[11](#references),[12](#references)]. The key idea in the ResNets consists of residual skip connections. Being the smallest size, capturing diagonal aspects, 3x3 convolution filters enable feature generation. Technically, posterior to each convolution filter, batch normalization is applied before the activation function [[11](#references)]. Batch normalization speeds up the time till convergence and has a regularization effect [[13](#references)]. As a general rule in the ResNets, the features are doubled when halving the feature map. More details can be found in the original paper [[11](#references)].
Squeeze and excitation blocks are introduced further to improve the Convolution Neural Networks (CNN) [[14](#references)].
In particular, a SE-Resnet34 is deployed as an encoder.

##Data characterization
The total amount of data consists of 72 annotated images. After the convergence of the network, the performance is tested on 5 reserved images. The residual 67 images are used for training and validation. The data is imagewise independent. In addition, an identical distribution of the information overall images was assumed. Therefore, a split is performed for training and validation, categorizing 70% and 30% of the 67 images for training and validation, respectively. 

In addition to the limited amount of data, the data is negatively imbalanced. Particularly, about 4.3% of the training and validation data pixels consist of metallic particles; the remaining 95.7% contain the background, support, or carbon-coated copper net.




