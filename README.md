# Enhancing Catalyst Analysis through Advanced Image Processing: Semantic Segmentation of Metallic Nanoparticles in TEM Imaging

## Introduction

Catalysts, enhancing efficiency and selectivity in chemical reactions, often consist of a metal that acts as the active site for the reaction and a support material that increases the surface area and stability of the metal particles. 
This combination is crucial in various industrial and environmental processes, enabling optimized reaction conditions and outcomes.
Especially interesting for the catalyst's performance are the loading and the Particle Size Distribution (PSD). 
Metallic nanoparticles are deposited on inert supports via Supercritical Fluid Reaction Deposition synthesizing catalysts. 
The loaded supports are analyzed in terms of the nanoparticles with a Scanning Transmission Electron Microscope (STEM) at the Laboratory for Electron Microscopy (LEM) at KIT. 
Bright-Field (BF) and High-Angle Annular Dark Field (HAADF) images are sampled with the microscope FEI Osiris ChemiStem. 
For a resolution in the range of 0.1 nm, a sample thickness below 50 nm is required. Therefore, only supports with a diameter lower than 50 nm are sufficient for TEM analysis. 
The solid-loaded supports are suspended in demineralized water. Then, the suspension is sprayed on a carbon-coated copper net and dried before the measurement.

The PSD of the metallic particles is important for the performance of a catalyst [[1](#references)]. 
The PSD is determined in 4 steps: Transforming the image into a binary image, detecting the Region of Interest (RoI), determining characteristics of the RoI, and calculating the PSD. 
First, a binary image is gained by performing semantic segmentation on HAADF images. In the binary image, the pixels containing metallic particles, the RoI, have the highest value and the rest (background, support, or the carbon-coated copper net) the lowest value (e.g., 255 and 0 for unsigned 8-bit integer, respectively). 
A state-of-the-art algorithm is designed to transform the TEM images into binary images. 
The metallic particles are detected in the binary image, and the pixel-wise area and perimeter information are measured and sampled. 
The pixel-wise length of the scale is determined. 
Reading in the scale with Google’s Tesseract-Optical-Character-Recognition [[2](#references)] using python-tesseract [[3](#references)], the pixel-to-nanometer ratio is obtained. Assuming spherical-shaped particles, the equivalent spherical diameter is calculated. Finally, the PSD of the equivalent spherical diameter is determined.

## Model’s architecture

Semantic segmentation is a pixel-wise classification. In the case of loaded supports, the metallic particles embody the RoI. The class label (e.g., 0 and 1) is assigned to each pixel in the image. 

Since the breakthrough with the AlexNet in 2012 [[4](#references)], deep learning models have been widely used in computer vision for detection, segmentation, and recognition [[5](#references)]. In 2015, a convolution network for biomedical image segmentation was developed [[6](#references)]. This model, called U-Net by its authors, is developed to detect fine details in the image while requiring a few annotated images as ground truth [[7](#references)]. The U-Net outperformed fully convolution networks with the sliding window approach and won the International Symposium on Biomedical Imaging (ISBI) in 2015 [[6](#references)]. The main improvement of the U-Net is the combining of the features gained in the downsampling and upsampling path. The reduction of the image size using max-pooling or convolution reduces the spatial resolution while emphasizing the important structures. Additional information on pooling and convolution arithmetic can be found in [[8](#references)]. In the downsampling path, also called the encoder, the features are extracted using convolution layers and max-pooling operations, reducing the input image's size. In contrast to autoencoders, the feature maps of the encoder and decoder are combined in the upsampling path, also called the decoder. Therefore, the feature maps are upsampled using 2x2 convolution, doubling the spatial size. The resulting feature maps, containing detailed information, are concatenated with feature maps gained from the encoder containing more spatial information. Then, a convolution filter is applied to combine the feature maps. Thus, the spatial information and the emphasized structures are combined. As a regularization method, batch normalization is applied before each activation layer. In the activation layer, a Rectified Linear unit (ReLu) is used. The ReLu was introduced in 2010 [[9](#references)], possibly named after Relu Patrascu at the University of Toronto. In contrast to the sigmoid function or the tangents hyperbolic, ReLu emphasizes sparsity in the neural network, leading to higher generalization and avoiding the gradient vanishing effect [[10](#references)]. For the next upsampling operation, convolution filter, batch normalization, and ReLu for activation are used. In the original description, no padding is used before applying convolution filters, resulting in a slight reduction of the output [[6](#references)]. For more details, the author refers to the original paper [[6](#references)]. 

For further improvement of the original U-Net [[6](#references)], the encoder part is substituted with more advanced convolution networks. In particular, using deep Residual Networks (ResNet) [[11](#references)] in the encoder part increases the performance of the U-Net. The encoder part benefits from the deeper architecture, increasing the depth and number of the features. ResNets implements residual learning to train large networks, resulting in higher accuracy, less trainable, and faster converging compared to non-residual learning networks such as the VGG16 or VGG19 networks [[11](#references),[12](#references)]. The key idea in the ResNets consists of residual skip connections. Being the smallest size, capturing diagonal aspects, 3x3 convolution filters enable feature generation. Technically, before each convolution filter, batch normalization is applied before the activation function [[11](#references)]. Batch normalization speeds up the time till convergence and has a regularization effect [[13](#references)]. As a general rule in the ResNets, the features are doubled when halving the feature map. More details can be found in the original paper [[11](#references)].
Squeeze and excitation blocks are introduced further to improve the Convolution Neural Networks (CNN) [[14](#references)].
In particular, a SE-Resnet34 is deployed as an encoder.

## Data characterization
The total amount of data consists of 72 annotated images. After the convergence of the network, the performance is tested on 5 reserved images. The residual 67 images are used for training and validation. The data is imagewise independent. In addition, an identical distribution of the information overall images was assumed. Therefore, a split is performed for training and validation, categorizing 70% and 30% of the 67 images for training and validation, respectively. 

In addition to the limited amount of data, the data is negatively imbalanced. About 4.3% of the training and validation data pixels consist of metallic particles; the remaining 95.7% contain the background, support, or carbon-coated copper net.

## Unveiling the Foundations of Model Training: Strategies and Theories in Transfer Learning

In transfer learning, the network is initially trained on a large dataset and then applied to a similar problem with less data [[15](#references)]. This approach is advantageous when the features learned from the large dataset can effectively represent the data in the new, smaller dataset [[15](#references)]. Particularly in cases with limited data, where training a network from scratch could lead to overfitting, transfer learning leverages generalized features from the larger dataset to solve similar problems [[16](#references)].

In this study, various encoder weights, previously trained on the ImageNet 2012 dataset [[19](#references)], are employed. The ImageNet 2012 dataset is extensive, containing 1.2 million images across 1000 different classes [[19](#references)]. These pretrained weights are sourced from [[17](#references),[18](#references)].

Subsequently, the decoder path of the network is trained using the ADAM optimizer. This training process adopts the learning decay strategy used by the authors of ResNet [[11](#references)]. The learning rate begins at 0.01 and is reduced by a factor of 10 whenever there is a plateau in error metrics.

Considering the negative imbalance of the data, using standard metrics like accuracy might lead to incorrect assessments of the network's performance [[20](#references),[21](#references)]. The Intersection-over-Union (IoU), also known as the Jaccard coefficient, is identified as a more suitable metric for semantic segmentation tasks, particularly with negatively imbalanced data, as reported by Rahman and Wang [[21](#references)]. Although Chico and Jurman recommend the Mathews correlation for all binary classification tasks, the IoU is deemed appropriate for datasets with negative imbalance [[20](#references)]. Rahman and Wang also developed an IoU approximation suitable for backpropagation in neural networks [[21](#references)]. Optimizing the network using IoU loss instead of the commonly used softmax loss has been shown to improve prediction accuracy [[21](#references)]. An early stopping mechanism based on the IoU score of validation data is implemented to prevent overfitting. Specifically, the training is halted if the IoU score does not improve over a set number of epochs.
Upon that, the old training loss is saved. 
Finally, the model is trained on the training and validation dataset.
Thereby, the training is stopped when reaching the old training loss, avoiding overfitting and leveraging the whole dataset.


As highlighted by the creators of U-Net [[6](#references)], image augmentation is essential, especially in the context of biomedical image classification tasks, where data is often scarce. Given that experiments and electron microscopy are both time-consuming and costly, they typically yield limited data. Image augmentation, while not increasing the data quantity, enhances its diversity, leading to a more generalized and robust model [[22](#references)]. Augmentation techniques for biomedical and microscope images, such as grid distortion and elastic transformation, are particularly beneficial [[7](#references),[22](#references)]. The Albumentations module is used for implementing image augmentation [[22](#references)], which applies a combination of methods, including both nondestructive transformations (like vertical and horizontal flips, 90° rotations) and destructive methods (like random scaling and rotating). The final steps in augmentation involve random contrast changes and image sharpening or blurring.

In contrast to non-adaptive optimizers, adaptive learning algorithms, such as ADAM, do not necessitate extensive hyperparameter tuning, albeit at a slight performance cost [[13](#references)]. To conserve the time required for tuning parameters in non-adaptive optimizers, the network in this study is optimized using Adaptive Moment Estimation (Adam), following the guidelines presented in [[23](#references)].

## Exploration the training
For the training and testing of the neural network, the modules Keras [[18](#references)] and Segmentation Model [[17](#references)] are used acting as an interface for the TensorFlow [[24](#references)] library. 
The U-Net is trained with different backbones. 
First, the weights from the ImageNet 2012 are frozen and only the encoder weights are fitted to the images. 
After the error on training and validation plateaus, the decoder and encoder path are trained with a low learning rate for fine-tuning. 
The models are evaluated with IoU and the confusion matrix.
The training and testing of the neural network were performed on the bwUniCluster (2.0). 
The author acknowledges support from the state of Baden-Württemberg through bwHPC.

## Performance comparing for different backbones

## References

[1] M. Casapu, A. Fischer, A.M. Gänzler, R. Popescu, M. Crone, D. Gerthsen, M. Türk, J.-D. Grunwaldt, Origin of the Normal and Inverse Hysteresis Behavior during CO Oxidation over Pt/Al2O3, ACS Catal. 7 (2017) 343–355

[2] R. Smith, An Overview of the Tesseract OCR Engine 629–633

[3] Samuel Hoffstaetter, Juarez Bochi, Matthias Lee, Ryan Mitchell, Emilio Cecchini, John Hagen, Darius Morawiec, Eddie Bedada, Uğurcan Akyüz, Pytesseract

[4] A. Krizhevsky, I. Sutskever, G.E. Hinton, ImageNet classification with deep convolutional neural networks, Commun. ACM 60 (2017) 84–90

[5] Y. LeCun, Y. Bengio, G. Hinton, Deep learning, Nature 521 (2015) 436–444

[6] O. Ronneberger, P. Fischer, T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, arXiv, 2015

[7] O. Ronneberger, P. Fischer, T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, arXiv, 2015

[8] V. Dumoulin, F. Visin, A guide to convolution arithmetic for deep learning, arXiv, 2016

[9] Vinod Nair, Geoffrey E. Hinton, Rectified linear units improve restricted boltzmann machines, in: Proceedings of the 27th International Conference on International Conference on Machine Learning, pp. 807–814

[10] X. Glorot, A. Bordes, Y. Bengio, Deep Sparse Rectifier Neural Networks, in: G. Gordon, D. Dunson, M. Dudík (Eds.), Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, 
PMLR, Fort Lauderdale, FL, USA, 2011, pp. 315–323

[11] K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, arXiv, 2015

[12] K. Simonyan, A. Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, arXiv, 2014

[13] C. Garbin, X. Zhu, O. Marques, Dropout vs. batch normalization: an empirical study of their impact to deep learning, Multimed Tools Appl 79 (2020) 12777–12815

[14] J. Hu, L. Shen, S. Albanie, G. Sun, E. Wu, Squeeze-and-Excitation Networks, IEEE transactions on pattern analysis and machine intelligence 42 (2020) 2011–2023

[15] I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, 2016

[16] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, T. Darrell, DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition

[17] P. Yakubovskiy, Segmentation Models, GitHub repository (2019)

[18] F. Chollet, others, Keras, 2015, https://github.com/fchollet/keras

[19] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A.C. Berg, L. Fei-Fei, ImageNet Large Scale Visual Recognition Challenge, Int J Comput Vis 115 (2015) 211–252

[20] D. Chicco, G. Jurman, The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation, BMC genomics 21 (2020) 6

[21] G. Bebis, R. Boyle, B. Parvin, D. Koracin, F. Porikli, S. Skaff, A. Entezari, J. Min, D. Iwai, A. Sadagic, C. Scheidegger, T. Isenberg, Advances in Visual Computing 10072 (2016) 234–244

[22] A. Buslaev, V.I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, A.A. Kalinin, Albumentations: Fast and Flexible Image Augmentations, Information 11 (2020) 125

[23] S. Ruder, An overview of gradient descent optimization algorithms, 2016

[24] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G.S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mane, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Viegas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, X. Zheng, TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems

