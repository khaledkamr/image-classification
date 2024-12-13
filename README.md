# Fruits and Vegetables Image Classification
This project focuses on classifying images of fruits and vegetables using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras. The dataset is organized into three parts: training, validation, and test sets, and images are classified into distinct categories such as different fruits and vegetables.

## Model Architecture
The model is a Sequential CNN with the following layers:
1. **Rescaling Layer**: Normalizes pixel values to the range [0, 1].
2. **Convolutional Layers**: Three convolutional layers with ReLU activation and max-pooling:
   - Conv2D (16 filters, kernel size 3x3)
   - Conv2D (32 filters, kernel size 3x3)
   - Conv2D (64 filters, kernel size 3x3)
3. **Flatten Layer**: Converts the feature map into a 1D vector.
4. **Dropout Layer**: Prevents overfitting by randomly setting 20% of the neurons to zero during training.
5. **Dense Layers**: Fully connected layers:
   - Dense (128 units)
   - Dense (number of categories)

## Training
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 25

The model is trained using the training dataset, and performance is validated using the validation dataset.

## Results
During training, the model's accuracy and loss are plotted for both training and validation sets. Key metrics:
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

## Testing
The model is tested on the unseen test dataset and can make predictions on individual images.