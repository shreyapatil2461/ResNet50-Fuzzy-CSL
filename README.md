# **Optimization of ResNet-50 Architecture for Imbalanced Dataset using Fuzzy Cost-Sensitive Learning**
---
## **Introduction**
Imbalanced datasets are a significant challenge in real-world machine learning applications, especially in image classification tasks where certain classes are severely underrepresented. Traditional deep learning models often become biased toward majority classes, leading to poor recognition of rare or minority categories.

This project proposes an optimized ResNet-50 architecture integrated with Fuzzy Cost-Sensitive Learning (CSL) to effectively address class imbalance. By introducing fuzzy logic into the cost-sensitive learning framework, the system dynamically adjusts misclassification penalties during training. This allows the model to better focus on underrepresented classes while maintaining overall classification performance.

The model is evaluated on a garbage classification dataset with ten waste categories, demonstrating a high accuracy of 97.06%, outperforming conventional methods such as standard CNNs and MobileNetV2. The proposed approach is highly practical for real-world applications such as automated waste sorting, environmental monitoring, and other scenarios where fair and accurate classification is crucial.

## **Objectives**
1.Improve the classification performance on imbalanced datasets.
2.Minimize misclassification of minority classes.
3.Demonstrate the effectiveness of integrating fuzzy logic into cost-sensitive learning.
4.Provide a practical solution for real-world waste classification systems.

## **Dataset Description**
The study focuses on addressing the unique challenges presented by an imbalanced garbage classification dataset. The dataset includes ten distinct categories: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, and trash.

The distribution of the dataset is highly uneven. Categories such as clothes contain a significantly larger number of images, while others such as metal and battery are underrepresented. This imbalance poses a challenge for the model to perform consistently well across all classes.
