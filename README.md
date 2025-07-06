# **Optimization of ResNet-50 Architecture for Imbalanced Dataset using Fuzzy Cost-Sensitive Learning**
---
## **Introduction**
Imbalanced datasets are a significant challenge in real-world machine learning applications, especially in image classification tasks where certain classes are severely underrepresented. Traditional deep learning models often become biased toward majority classes, leading to poor recognition of rare or minority categories.

This project proposes an optimized ResNet-50 architecture integrated with Fuzzy Cost-Sensitive Learning (CSL) to effectively address class imbalance. By introducing fuzzy logic into the cost-sensitive learning framework, the system dynamically adjusts misclassification penalties during training. This allows the model to better focus on underrepresented classes while maintaining overall classification performance.

The model is evaluated on a garbage classification dataset with ten waste categories, demonstrating a high accuracy of 97.06%, outperforming conventional methods such as standard CNNs and MobileNetV2. The proposed approach is highly practical for real-world applications such as automated waste sorting, environmental monitoring, and other scenarios where fair and accurate classification is crucial.

## **Objectives**
**1.Improve the classification performance on imbalanced datasets.**  
**2.Minimize misclassification of minority classes.**  
**3.Demonstrate the effectiveness of integrating fuzzy logic into cost-sensitive learning.**  
**4.Provide a practical solution for real-world waste classification systems.**  

## **Dataset Description**
The study focuses on addressing the unique challenges presented by an imbalanced garbage classification dataset. The dataset includes ten distinct categories: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, and trash.

The distribution of the dataset is highly uneven. Categories such as clothes contain a significantly larger number of images, while others such as metal and battery are underrepresented. This imbalance poses a challenge for the model to perform consistently well across all classes.

 <p align="center">
  <img src="Figure1.png" alt="Dataset Size" width="50%"/>
</p>

<h4 align="center">Dataset Size</h4>

<p align="center">
  <img src="Figure2.png" alt="Dataset Example" width="50%"/>
</p>

<h4 align="center">Dataset Example</h4>


## **Proposed Methodology**
The proposed approach integrates the ResNet-50 architecture with fuzzy cost-sensitive learning to address class imbalance. The pipeline begins with preprocessing the garbage dataset, including resizing, normalization, and augmentation.

All images were resized to 256×256 pixels to maintain uniformity. Augmentation techniques such as random rotations, flips, and cropping were applied to increase the variety of images for underrepresented classes without altering their key features. Pixel values were normalized to a scale of 0 to 1 to accelerate the training process.

The ResNet-50 model was used as the core feature extractor. Pretrained on ImageNet, the ResNet-50 architecture was adapted by replacing the final output layer to support ten waste categories. The initial layers were kept unchanged to retain general feature learning abilities, while the later layers were fine-tuned for dataset-specific patterns.

The fuzzy penalty loss function introduced a dynamic way to address the uneven distribution of classes. Instead of treating all misclassifications equally, this function adjusted penalties based on class distribution. Higher penalties were assigned for errors in rare categories such as battery and metal, and lower penalties for common categories like clothes. Penalties were updated dynamically during training, encouraging balanced and fair learning.

The fuzzy penalty loss is defined as:  
L<sub>Fuzzy</sub> = - (1 / N) ∑<sub>i=1</sub><sup>N</sup> w<sub>y<sub>i</sub></sub> · log(p<sub>y<sub>i</sub></sub>)  
Where:

- *w*<sub>*yᵢ*</sub>: weight assigned to the true class *yᵢ*, inversely related to class frequency.
- *p*<sub>*yᵢ*</sub>: predicted probability for the true class.
- *N*: total number of samples.

The model was trained using the Adam optimizer with an initial learning rate of 0.001, gradually reduced during training. Batch sizes of 32 or 64 were used, and dropout layers were added to avoid overfitting.




