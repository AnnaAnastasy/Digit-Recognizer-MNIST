# Power of Neural Networks in Digit Recognition

## Overview
This project explores the power of neural networks in image classification, specifically for recognizing handwritten digits from the MNIST dataset. Starting with a simple dense neural network, we then move to Convolutional Neural Networks (CNNs) to leverage their advanced capability in processing spatial information, significantly improving accuracy.

## Project Structure
1. **Introduction**: Background on neural networks and the motivation for using CNNs for image data.
2. **Data Preprocessing**: Loading, normalizing, and reshaping the data to prepare for neural network training.
3. **Model Development**: 
   - Baseline model using a dense neural network.
   - Advanced model using a CNN for improved feature extraction and classification.
4. **Training and Evaluation**: Training models and comparing results, with the CNN achieving an impressive 0.99514 accuracy score on Kaggle.

## Results
- **Best Model**: Convolutional Neural Network (CNN)
- **Score**: The final model achieved a score of 0.99514 on the Kaggle "Digit Recognizer" competition leaderboard.

## Requirements
- Python 3.x
- Libraries: TensorFlow, Keras, NumPy, Matplotlib, etc.

## Usage
1. Clone the repository.
2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/).
3. Save the downloaded file in the same directory as the Jupyter Notebook.
4. Install the necessary libraries:
```pip install tensorflow keras numpy pandas matplotlib scikit-learn```
5. Run the notebook to view model training and evaluation.

## Acknowledgments
This project was created as an exercise in deep learning with a focus on image recognition tasks. Special thanks to the Kaggle community and MNIST dataset creators.
