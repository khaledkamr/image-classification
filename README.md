# Fruits and Vegetables Image Classification
This project focuses on classifying images of fruits and vegetables using a **Convolutional Neural Network (CNN)** implemented in TensorFlow and Keras. The dataset is organized into three parts: training, validation, and test sets, and images are classified into distinct categories such as different fruits and vegetables.

The model is trained using the **training dataset**, and performance is validated using the **validation dataset**. Results are visualized using accuracy and loss plots for both training and validation phases.

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

## **Deploying the Web App**

The trained model has been deployed as a **web application** using the **Streamlit** library. Users can upload an image of a fruit or vegetable, and the app will classify the image and display the predicted category along with the prediction Accuracy.

### **Features of the Web App**:
- Upload an image directly (JPG, JPEG, PNG).  
- View the predicted fruit or vegetable category.  
- See the Accuracy score of the prediction. 


## **How to Run the Web App Locally**

### 1. **Clone the Repository**
Download or clone the repository from GitHub:

```bash
git clone https://github.com/your-username/Fruits_Veggies_Classifier.git
cd Fruits_Veggies_Classifier
```

### 2. **Install Dependencies**
Ensure you have Python 3.8+ installed. Install the required libraries using `pip`:

```bash
pip install tensorflow streamlit numpy matplotlib
```

### 3. Run the Web App
Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

## **Sample Output**

Here’s an example output after running the web app:

![sample-1](https://github.com/user-attachments/assets/29adbf61-5362-43d0-9955-98fdb4f1a05a)

![sample-2](https://github.com/user-attachments/assets/2e883191-af45-4596-992e-5c8c48007ec7)
