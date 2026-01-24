# NO-CODE MACHINE LEARNING PLATFORM

A **Streamlit-based Machine Learning platform** that allows users to view data, clean data, visualize it, train models, compare two models, and evaluate performance. Users can also perform predictions on new datasets, save their models, and reuse them for future predictions. This project is designed to simulate a real-world ML workflow.
## Live Demo
🔗 https://train-your-own-model.streamlit.app/
## ALGORITHMS USED:

### **From Scratch**
* Linear Regression
* Binary Classification
* Multiclass Classification (Shallow NN)
* K-Means
* Anomaly Detection (Gaussian)
* Decision Tree

### **Using Libraries**
* Linear Regression
* Binary Classification
* Multiclass Classification (Shallow NN)
* K-Means
* Decision Tree
* Random Forest
* XGBoost

---

## Key Features:

### **1. Train New Model**
* **Data Preview**: View your dataset and a statistical summary.
* **Cleaning**: Clean your dataset by removing duplicates, encoding categorical values, filling missing values, etc. Includes a feature for feature transformation.
* **Visualization**: Visualize your dataset in four forms: scatterplot, histogram, boxplot, and heatmap.
* **Standardization**: Standardize your dataset for better model performance.
* **Training**: Select the Algorithm Type, Model Type, and Implementation Type. Set parameters to train models and compare two different models simultaneously.
* **Testing**: Evaluate your model performance on the `X_test` set.
* **Prediction**: Predict values for a new dataset based on the trained model and download the results.

### **2. Use Existing Model**
* **Model Selection**: Choose from your list of saved models.
* **Activation**: Activate the selected model for use.
* **Prediction**: Predict results based on the activated model.
* **Export**: Download the predicted data.

---

## Workflow

### **For Training a New Model**
1. Upload the dataset you want to use.
2. Clean the dataset.
3. Apply transformations (Log, Square, Square Root, or Absolute) to make data distributions more symmetric.
4. Select the features for **X** and **y**.
5. Split the dataset into Train/Test sets or Train/CV/Test sets.
6. Standardize the data splits.
7. Select the model you wish to train.
8. (Optional) Compare two models by clicking **Compare Models**.
9. View results based on the CV or Train dataset.
10. Evaluate final performance on the Test dataset.
11. Run predictions on new data and download the output.

### **For Using an Existing Model**
1. Select the saved model you want to use.
2. Activate it.
3. Navigate to the **Prediction** tab and upload the data you wish to predict.

---

> **NOTE:** In **Use Existing Model**, there is a specific model format requirement. Currently, you can only use existing models that were created and saved within this app.

---
##  Installation & Setup

Ensure you have Python 3.x installed on your machine.

### **1. Clone the Repository**
```bash
git clone [https://github.com/BEGINNERUSER-git/Train-your-own-Model.git](https://github.com/BEGINNERUSER-git/Train-your-own-Model.git)
cd Train-your-own-Model
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **2. Run the App**
```bash
streamlit run app.py
```



---


https://github.com/user-attachments/assets/5295ea3e-aaa1-47ba-ba5c-3a82f1c35ce0

---
## License

Copyright © 2026 BEGINNERUSER-git

This project is licensed under the MIT License – see the LICENSE file for details.

