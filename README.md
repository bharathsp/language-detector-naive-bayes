# 🌍🔤 Language Detection Model  
*A Machine Learning project for identifying languages from text using Multinomial Naive Bayes.*

---

## 📌 Project Overview  
This project implements a **Language Detection Model** using the **Multinomial Naive Bayes algorithm**.  
It classifies text into its respective language by applying **text preprocessing, feature engineering, hyperparameter tuning (GridSearchCV), and model validation**.  

The final model is packaged into a **pipeline** and saved with **Pickle** for deployment.
The pipeline is then serialized into a joblib file. create api endpoints using fast api. Develop interactive webpage using html and css. Deploy locally into docker and test the application. Post local testing deploy the application into Azure app services.

---

## 📂 Dataset  
📄 **Language Detection.csv**  
- Contains text samples and their corresponding language labels.  
- Preprocessing steps:  
  - Handle missing values  
  - Clean text using **Regex** (remove special characters, numbers)  
  - Convert text to **lowercase**  
  - Encode labels using **Label Encoder**  

---

## 🛠️ Project Workflow  

### 1️⃣ Data Preprocessing  
✔️ Handle missing values  
✔️ Clean text (regex: remove numbers & special characters)  
✔️ Convert to lowercase  
✔️ Encode labels  

### 2️⃣ Feature Engineering  
- Split into **Train/Test sets**  
- Apply **TfidfVectorizer** with `ngram_range=(1,2)`  

### 3️⃣ Model Training & Hyperparameter Tuning  
- Algorithm: **Multinomial Naive Bayes**  
- Hyperparameters tuned with **GridSearchCV**  
```python
params = {'alpha': [0.01, 0.1, 0.5, 1]}
grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

### 4️⃣ Model Validation

* Predictions on test set
* Evaluation metrics:

  * ✅ Accuracy Score
  * ✅ Confusion Matrix
  * ✅ Classification Report

📊 **Confusion Matrix Visualization** for Actual vs Predicted

### 5️⃣ Model Deployment

* Build a **Pipeline** with `TfidfVectorizer` + `MultinomialNB`
* Save model using **Pickle**

```python
pipe = Pipeline([('vectorizer', tfidf), ('multinomialNB', best_model)])
pipe.fit(X_train, y_train)
```

---

## 🚀 Making Predictions

```python
text = "Hello, how are you?"
y = pipe.predict([text])
print(le.classes_[y[0]], y)
```

**Output:**

```
English [0]
```

## Serialize the final language detector model and label encoder into joblib file
*languageDetector.py*
Save trained pipeline using joblib - *trained_pipeline-0.1.0.joblib*
Save label encoder separately - *label_encoder.joblib*

<img width="403" height="348" alt="image" src="https://github.com/user-attachments/assets/80e94220-af8e-4428-a058-f2d76ed3866f" />

## API endpoints created using FastAPI
*server.py*
Root endpoint - static/language_detection_frontend.html
Prediction endpoint - Returns "predicted_language"

<img width="699" height="886" alt="image" src="https://github.com/user-attachments/assets/d80d07c3-8fe7-4650-909d-4b971356a435" />

## HTML file with CSS
language_detection_frontend.html is created inside static folder

## Create a Dockerfile
Use official lightweight Python image
Set working directory
Install system dependencies (optional: for pandas/numpy performance)
Copy requirements first
Install dependencies
Copy application files
Copy static files
Expose FastAPI default port
Run the app with uvicorn

<img width="586" height="545" alt="image" src="https://github.com/user-attachments/assets/f734b53b-5564-4edc-9611-0a33357bf3a2" />

## Deploy to docker on local

docker build -t language-detection-api .

<img width="888" height="217" alt="image" src="https://github.com/user-attachments/assets/99aec8da-5d18-4704-9af9-9c69b7d6b327" />

docker run -d -p 8000:8000 language-detection-api

<img width="774" height="39" alt="image" src="https://github.com/user-attachments/assets/72df21eb-4915-4766-bb98-c29defc33c0d" />

## Test the app on local

outputs - http://localhost:8000/

<img width="1079" height="600" alt="image" src="https://github.com/user-attachments/assets/96e6d919-f7f4-469d-875e-000af93e75d0" />

<img width="1057" height="562" alt="image" src="https://github.com/user-attachments/assets/661caf4b-860c-4b55-947e-c2cb39f4a7a8" />

## Deploy the application into Azure app services

## Test the app on cloud
Check if the app is accessible anywhere using the endpoints

---

## 📊 Tech Stack

* 🐍 Python
* 📚 Scikit-learn
* 🔢 Numpy, Pandas
* 📊 Matplotlib, Seaborn
* 🛠️ Regex
* 🗄️ Pickle
* joblib
* docker
* Azure
* fast api
* uvicorn
* html
* css

---

## 📸 Visualizations

* Language Distribution Plot
* Confusion Matrix Heatmap

---

## ✅ Results

* Optimized model with **GridSearchCV**
* Achieved high accuracy across multiple languages
* End-to-end pipeline ready for deployment

---

## 📌 Future Improvements

* Add support for more languages 🌐
* Experiment with **Deep Learning models (LSTMs, Transformers)**
* Deploy as a **REST API / Streamlit app**
