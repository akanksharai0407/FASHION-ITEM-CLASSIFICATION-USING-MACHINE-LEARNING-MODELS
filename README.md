# Fashion Item Classifier 

This project focuses on building a fashion item classifier using traditional machine learning techniques. It aims to automate the categorization of clothing items in grayscale images, helping improve product search and tagging in e-commerce platforms.

---

## Problem Statement

With the rapid rise of e-commerce platforms and fashion content, it's becoming increasingly important to automate the classification of clothing items for better search and recommendation. Manual tagging is not scalable and often inconsistent.

This project solves a **multiclass image classification** problem using ML techniques (KNN, SVM, Random Forest) on the **FashionMNIST** dataset, based on HOG features.

---

## Algorithms Used

- K-Nearest Neighbors (KNN)
- Support Vector Machine (Linear & RBF kernel)
- Random Forest Classifier

All models were trained and evaluated on features extracted using the **Histogram of Oriented Gradients (HOG)** method.

---

##  Dataset

- **FashionMNIST** (by Zalando Research)  
- 70,000 grayscale images (28x28 pixels)  
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot  
- Downloaded via `torchvision.datasets`

Dataset link: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

---

## Evaluation

Each model was evaluated using:

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Visual comparison of misclassified images

| Model         | Accuracy |
|---------------|----------|
| SVM (RBF)     | 85.46%   |
| Random Forest | 82.98%   |
| SVM (Linear)  | 82.88%   |
| KNN           | 81.72%   |

 We also visualized:
- Sample input images per class
- Misclassified outputs from the best model (SVM RBF)

---

## Libraries Used

- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `torch`, `torchvision`
- `skimage` (for HOG feature extraction)
- `tqdm` (for progress tracking)

---

## Future Improvements

- Try dimensionality reduction (PCA) for faster training
- Compare with CNNs or transfer learning
- Extend to more complex datasets like DeepFashion

---

## Authors

- Akanksha Rai  
- Zhanel Ashirbek

