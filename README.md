# Image Classification Using CNN

A deep learning project that classifies images into 10 different categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The project includes an interactive web interface developed using Streamlit.

## Project Overview

This project uses the CIFAR-10 dataset to train a CNN model that predicts the category of an uploaded image.

The model supports the following classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Streamlit
- Pillow
- Matplotlib
- Seaborn
- Scikit-learn

## Model Architecture

The CNN consists of:

1. Convolutional layers for feature extraction
2. Max-pooling layers for dimensionality reduction
3. A flattening layer
4. A dense layer for classification features
5. A dropout layer to help reduce overfitting
6. A softmax output layer for 10-class prediction

The input images are resized to 32 × 32 pixels and normalized before prediction.

##  Model Performance

- **Test Accuracy:** 74.88%
- **Test Loss:** 0.7716
- **Dataset:** CIFAR-10
- **Training Epochs:** 10

These results were obtained on the CIFAR-10 test dataset. Performance on real-world photographs may differ.

##  Project Structure

```text
Image-Classification/
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
│
├── models/
│   └── image_classifier.keras
│
└── results/
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── training_accuracy.png
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/RishikaYadav960/Image-Classification.git
```

### 2. Navigate to the project folder

```bash
cd Image-Classification
```

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the web application.

## Features

- Upload images in JPG, JPEG, and PNG formats
- Predict image categories using a trained CNN
- Display prediction confidence scores
- View prediction scores for all 10 classes
- Explore training accuracy and confusion matrix visualizations

## Limitations

The model was trained on CIFAR-10, which contains only 10 object categories. It is not designed to recognize arbitrary objects or reliably classify photographs outside its training distribution.

Prediction confidence does not guarantee that a prediction is correct.

## Future Improvements

- Experiment with transfer learning using pretrained models such as MobileNetV2
- Improve generalization to real-world photographs
- Evaluate performance across different image conditions
- Improve the user interface and prediction explanations
- Deploy the application for public access

## Author

**Rishika Yadav**

B.Tech Computer Science and Engineering (AI & ML)

## License

This project is intended for educational and learning purposes.
