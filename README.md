# Hierarchical Crack Classification

This repository contains a deep learning project for hierarchical classification of concrete cracks in civil infrastructure. The system uses a two-level hierarchical approach with state-of-the-art convolutional neural networks to first detect if a crack is present, and then classify the type of crack if detected.

## Project Overview

The project implements a hierarchical image classification system that:

1. **First Level**: Determines whether an image contains a crack or not (binary classification)
2. **Second Level**: For images containing cracks, further classifies the type of crack into three categories:
   - Delamination
   - Spalling
   - Surface Crack

This hierarchical approach allows for more efficient classification and improved accuracy for this specialized domain.

## Repository Structure

```
HIERARCHICAL-CRACK-CLASSIFICATION/
├── models/                    # Saved model directory
│   ├── densenet_model/        # Trained DenseNet model (Level 1)
│   └── inception_model/       # Trained InceptionV3 model (Level 2)
├── crack-images.zip           # Compressed dataset
├── hierarchical_classification.ipynb  # Main Jupyter notebook
└── README.md                  # This file
```
## Dataset structure
```
CRACK-IMAGES
├── crack_images/              # Dataset directory
    ├── crack/                 # Images with cracks
    │   ├── delamination/      # Delamination type cracks
    │   ├── spalling/          # Spalling type cracks
    │   └── surface_crack/     # Surface crack images
    └── no_crack/              # Images without cracks
```

## Features

- **Hierarchical Classification**: Two-level classification approach that mimics expert decision-making
- **Transfer Learning**: Utilizes pre-trained models (DenseNet201 and InceptionV3) adapted for crack classification
- **Visualization**: Performance tracking with accuracy and loss plots
- **Optimized Architecture**: Fine-tuned CNN architectures for different levels of classification

## Model Architecture

### Level 1: Binary Classification (Crack vs. No Crack)
- **Base Model**: DenseNet201 (pre-trained on ImageNet)
- **Customization**: 
  - Frozen first 10 layers to preserve general feature extraction
  - Global Average Pooling
  - Dense layer with 512 neurons
  - Dropout (0.3) for regularization
  - Output layer with 2 neurons (sigmoid activation)

### Level 2: Crack Type Classification
- **Base Model**: InceptionV3 (pre-trained on ImageNet)
- **Customization**:
  - Frozen first 10 layers
  - Global Average Pooling
  - Dense layer with 512 neurons
  - Dropout (0.3) for regularization
  - Output layer with 3 neurons (sigmoid activation)

## Hyperparameters

### Level 1 Model (DenseNet201)
- **Optimizer**: Adam (default learning rate)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: Default from tf.keras.utils.image_dataset_from_directory
- **Epochs**: 100
- **Input Size**: 256×256×3
- **Data Split**: 70% training, 20% validation, 10% testing

### Level 2 Model (InceptionV3)
- **Optimizer**: Adam (default learning rate)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: Default from tf.keras.utils.image_dataset_from_directory
- **Epochs**: 150
- **Input Size**: 256×256×3
- **Data Split**: 70% training, 20% validation, 10% testing

## Requirements

- TensorFlow 2.x
- Keras
- NumPy
- OpenCV
- Matplotlib
- Pandas
- VisualKeras
- Pathlib

## Usage

### Dataset Preparation

1. The dataset is in the `crack-images` folder
2. Ensure the dataset is organized as follows:
   - `data/crack/delamination/` - Delamination crack images
   - `data/crack/spalling/` - Spalling crack images
   - `data/crack/surface_crack/` - Surface crack images
   - `data/no_crack/` - Images without cracks

### Running the Notebook

1. Open the Jupyter notebook:
```
jupyter notebook hierarchical_classification.ipynb
```

2. Run all cells in the notebook to:
   - Load and preprocess the dataset
   - Train the Level 1 model (DenseNet201)
   - Train the Level 2 model (InceptionV3)
   - Evaluate the hierarchical classification system
   - Test with new images

### Using the Trained Models

The notebook contains code to load the pre-trained models and perform predictions:

```python
from keras.models import load_model

# Load the trained models
model1 = load_model('path/to/densenet_model')
model2 = load_model('path/to/inception_model')

# Preprocess a test image
img = cv2.imread('path/to/test-image')
resize = tf.image.resize(img, (256, 256))
normalized = resize/255

# Level 1 prediction
yhat1 = model1.predict(np.expand_dims(normalized, 0))
yhat1 = np.argmax(yhat1)

# Hierarchical decision
if yhat1 == 1:
    print('No crack')
elif yhat1 == 0:
    print('Crack detected')
    # Level 2 prediction
    yhat2 = model2.predict(np.expand_dims(normalized, 0))
    yhat2 = np.argmax(yhat2)
    if yhat2 < 1:
        print('Delamination')
    elif yhat2 == 1:
        print('Spalling')
    else:
        print('Surface crack')
```

## Performance Metrics

The hierarchical classification system is evaluated using:

- **Precision**: Measures the accuracy of positive predictions
- **Recall**: Measures the ability to find all positive samples
- **Accuracy**: Overall accuracy of predictions

Detailed performance metrics are reported separately for both levels of classification.

## Future Work

- Expand the dataset with more diverse crack examples
- Implement real-time crack detection for video streams
- Develop a web or mobile application for field use
- Explore additional crack types and structural defects
- Optimize models for edge devices for on-site deployment

## License

MIT License

Copyright (c) 2025 Rebeka Rachel Lukacs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

