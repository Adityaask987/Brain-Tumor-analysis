# MRI Brain Tumor Analysis

This repository contains a Jupyter Notebook implementation of an MRI Brain Tumor analysis model. The model is designed to classify brain MRI scans into tumor or non-tumor categories, leveraging advanced deep learning techniques to achieve high accuracy.

## Dataset 

https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

## Project Overview

Detecting brain tumors early is crucial for effective treatment. This project uses MRI scans to automatically identify the presence of brain tumors, providing a tool for quick and reliable diagnosis.

### Key Features:
- **Image Preprocessing**: MRI images are preprocessed and resized to 224x224 pixels for uniform input into the model.
- **Deep Learning Architecture**: The model is built using a convolutional neural network (CNN) with PyTorch to extract features from MRI scans.
- **Training on GPU**: The project utilizes a Kaggle P100 GPU to process the input images efficiently.
- **Epochs**: The model is trained over 15 epochs, processing approximately 2.6 lakh input images and 1.3 lakh output images.
- **Data Augmentation**: Data augmentation techniques are used to prevent overfitting and improve the model's robustness.
  
## File Structure
- `"brain-tumor-detection.ipynb"`: The main Jupyter Notebook that contains the code for preprocessing, training, and evaluating the model.
  
## Dependencies

To run this project, you'll need the following libraries:

- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV (for image processing)
  
You can install these dependencies using the following command:

```bash
pip install torch torchvision numpy matplotlib scikit-learn opencv-python
```

## Results
The model demonstrates promising results, achieving high accuracy in detecting tumors from MRI scans. Detailed results, including loss and accuracy curves, are included in the notebook.

