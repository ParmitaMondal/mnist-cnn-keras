# mnist-cnn-keras
Convolutional Neural Network (CNN) implementation in Keras/TensorFlow for classifying MNIST handwritten digits. Includes training, evaluation, visualization of loss/accuracy, and prediction examples.

## 📂 Project Structure
- `ECE565_project3_1.ipynb` — Main Jupyter Notebook with code (training, evaluation, visualization).
- `requirements.txt` — Python dependencies.
- `README.md` — Project overview.

## 🚀 Features
- Loads and preprocesses MNIST dataset (60,000 training, 10,000 test images).
- Normalizes images and applies one-hot encoding to labels.
- Builds a CNN with Conv2D, MaxPooling2D, Dropout, and Dense layers.
- Compiles and trains the model using Adam optimizer.
- Evaluates model on test dataset and reports accuracy/loss.
- Visualizes **training vs validation loss/accuracy** curves.
- Makes predictions on sample test images and displays results.

## 📦 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/mnist-cnn-keras.git
cd mnist-cnn-keras
pip install -r requirements.txt
