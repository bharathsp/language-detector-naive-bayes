# 🌍🔤 Language Detection Model  
*A Machine Learning project for identifying languages from text using Multinomial Naive Bayes.*

---

## 📌 Project Overview  
This project implements a **Language Detection Model** using the **Multinomial Naive Bayes algorithm**.  
It classifies text into its respective language by applying **text preprocessing, feature engineering, hyperparameter tuning (GridSearchCV), and model validation**.  

The final model is packaged into a **pipeline** and saved with **Pickle** for deployment.

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

---

## 📊 Tech Stack

* 🐍 Python
* 📚 Scikit-learn
* 🔢 Numpy, Pandas
* 📊 Matplotlib, Seaborn
* 🛠️ Regex
* 🗄️ Pickle

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

docker build -t language-detection-api .
<img width="888" height="217" alt="image" src="https://github.com/user-attachments/assets/99aec8da-5d18-4704-9af9-9c69b7d6b327" />

<img width="565" height="488" alt="image" src="https://github.com/user-attachments/assets/1de25c25-d125-47f5-a1a6-e45807e43572" />

<img width="119" height="63" alt="image" src="https://github.com/user-attachments/assets/c232f9b4-86fd-435d-b5cb-867438013d61" />

<img width="655" height="752" alt="image" src="https://github.com/user-attachments/assets/a1810fe9-650c-427b-9685-cb39e7d3540b" />

docker run -d -p 8000:8000 language-detection-api
