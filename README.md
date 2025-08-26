# 🧠 Brain Tumor Detection with CNN

Welcome to the **Brain Tumor Detection** project! This repository
implements a Convolutional Neural Network (CNN) to classify MRI brain
images as tumorous or non-tumorous. Built with TensorFlow, OpenCV, and
Python, it includes preprocessing, data augmentation, model training,
and evaluation. 🚀

## 📖 Project Overview

This project aims to detect brain tumors in MRI scans using a deep
learning approach. It leverages a CNN model trained on an augmented
dataset of \~2065 images, achieving robust performance for binary
classification (tumor vs. no tumor). The pipeline includes image
preprocessing, model training, and evaluation with metrics like
accuracy, F1 score, and ROC curves. 🩻

## ✨ Features

-   **Preprocessing**: Crops brain contours and normalizes images to
    240x240 pixels. 🖼️
-   **Data Augmentation**: Balances dataset using ImageDataGenerator for
    rotation, flipping, and zooming. 🔄
-   **CNN Model**: Simple architecture with Conv2D, BatchNormalization,
    and MaxPooling layers. 🧬
-   **Evaluation**: Computes accuracy, F1 score, and ROC-AUC; visualizes
    results. 📊
-   **Single Image Prediction**: Upload and classify new MRI images with
    confidence scores. 🩺

## 🛠️ Installation

1.  Clone the repository:

    ``` bash
    git clone https://github.com/shervinnd/brain-tumor-detection.git
    ```

2.  Install dependencies:

    ``` bash
    pip install tensorflow opencv-python numpy matplotlib scikit-learn
    ```

3.  Ensure you have a GPU-enabled environment (e.g., Google Colab with
    T4 GPU) for faster training. ⚡

## 📂 Dataset

The model uses an augmented dataset (\~2065 images) stored in the
`augmented data` folder, with subfolders `yes` (tumor) and `no` (no
tumor). Images are preprocessed to focus on brain regions and
normalized. 🖥️

## 🚀 Usage

1.  **Train the Model**:
    -   Run the Jupyter notebook (`Brain_Tumor_Detection.ipynb`) to
        preprocess data, train the CNN, and save the best model
        (`best_model.h5`).
2.  **Evaluate**:
    -   Check accuracy, F1 score, and ROC curve on the test set (15% of
        data).
3.  **Predict on New Images**:
    -   Use the provided script to upload an MRI image and get
        predictions with confidence scores.

## 📈 Model Architecture

The CNN consists of:

-   ZeroPadding2D (padding=2)
-   Conv2D (32 filters, 7x7 kernel)
-   BatchNormalization
-   ReLU Activation
-   Two MaxPooling2D (4x4, stride=4)
-   Flatten
-   Dense (1 unit, sigmoid activation)

Compiled with Adam optimizer and binary crossentropy loss. 🧠

## 📊 Results

-   **Test Accuracy**: \~90% (varies by dataset quality)
-   **F1 Score**: \~0.88
-   **ROC-AUC**: Visualized for model performance
-   Example prediction: Displays true/predicted labels with confidence
    for test images. 📉

## 📸 Example

Upload an MRI image, and the model will predict if it contains a tumor,
displaying the result with a confidence score. Example output:

    Prediction: Tumor (Confidence: 0.92)

## 🤝 Contributing

Contributions are welcome! 🙌 Please:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/Feature`).
3.  Commit changes (`git commit -m 'Add Feature'`).
4.  Push to the branch (`git push origin feature/Feature`).
5.  Open a pull request.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for
details. 📄

## 🙏 Acknowledgments

-   Inspired by medical imaging and deep learning research. 🩺
-   Thanks to TensorFlow, OpenCV, and the open-source community! 💻
-   Dataset sourced from publicly available MRI scans (augmented for
    training).

## 📬 Contact

For questions or feedback, open an issue or reach out via GitHub. Let's
advance healthcare AI together! 🚀
