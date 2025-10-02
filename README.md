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

<img width="1638" height="301" alt="image" src="https://github.com/user-attachments/assets/93e8152d-c063-4bf4-ab95-b8199211fea1" />

docker run -d -p 8000:8000 language-detection-api

<img width="774" height="39" alt="image" src="https://github.com/user-attachments/assets/72df21eb-4915-4766-bb98-c29defc33c0d" />

<img width="1627" height="312" alt="image" src="https://github.com/user-attachments/assets/e99583bb-89ed-4f24-9d8c-2fd7f3adc4a6" />

## Test the app on local

outputs - http://localhost:8000/

<img width="1079" height="600" alt="image" src="https://github.com/user-attachments/assets/96e6d919-f7f4-469d-875e-000af93e75d0" />

<img width="1057" height="562" alt="image" src="https://github.com/user-attachments/assets/661caf4b-860c-4b55-947e-c2cb39f4a7a8" />

## Push image to docker hub
Azure App Service needs to pull your image from a container registry. You can use:
Azure Container Registry (ACR) (recommended)
Or Docker Hub

Tag the local image
docker tag language-detection-app:latest bharath0805/language-detection-app:latest

Push it to Docker Hub
docker push bharath0805/language-detection-app:latest

<img width="1055" height="241" alt="image" src="https://github.com/user-attachments/assets/1a0a4556-d294-4a29-a1e6-e74b3c4ccc41" />

Verify on Docker Hub

Go to hub.docker.com
 → login → check under your account bharath0805.

You should see the repository language-detection-app.

<img width="1893" height="443" alt="image" src="https://github.com/user-attachments/assets/dfc1e848-9bdb-4e17-9ee5-ede8cba61845" />

On Docker Desktop

<img width="1914" height="406" alt="image" src="https://github.com/user-attachments/assets/ba8e26d1-1c99-463b-97f2-536ef1cec0dd" />

## Deploy the application into Azure app services
Step 2: Create a Web App for Containers

Go to Azure Portal
.

Search for App Services → Create → Web App.

Fill the basics:

Subscription → your subscription

Resource Group → create or select existing

Name → e.g., language-detect-api (this becomes part of the URL: https://language-detect-api.azurewebsites.net)

Publish → Docker Container

Operating System → Linux

Region → pick nearest (India → Central India, South India, etc.)

<img width="718" height="894" alt="image" src="https://github.com/user-attachments/assets/23dc64d7-9578-47f3-9724-8139e79f5f92" />

🐳 Configure Docker Hub Container Settings

Image Source → ✅ Other container registries
(this is used for Docker Hub when you aren’t using ACR)

Name → this is just the App Service name you gave earlier (e.g. main).

Docker Hub Options →

Access Type: Public (since your repo is public)

If private → enter Docker Hub username + access token.

Registry server URL → leave as:

https://index.docker.io


Image and Tag → put your Docker Hub repo + tag:

bharath0805/language-detection-app:latest


Port → set to 8000 🚨 (important — your uvicorn app runs on port 8000 inside the container, not 80).
Azure will route external port 80 → internal port 8000.

Startup Command → leave blank (FastAPI is already started via CMD ["uvicorn", ...] in your Dockerfile).

<img width="690" height="643" alt="image" src="https://github.com/user-attachments/assets/e2e0a54b-5373-4e25-8e73-800b1b9e0199" />

Review + Create

<img width="630" height="804" alt="image" src="https://github.com/user-attachments/assets/3e97631b-8e60-4c17-822c-9b9b073d0c94" />

<img width="1201" height="435" alt="image" src="https://github.com/user-attachments/assets/3ec50caa-627d-4c75-bff6-0848beba3efe" />

## Test the app on cloud

<img width="850" height="249" alt="image" src="https://github.com/user-attachments/assets/10641f6e-659e-4040-98e5-5db5d3cb74f6" />

Test Your App
Go to your App Service → Overview → Copy the Default Domain (e.g. [https://language-detect-api.azurewebsites.net/](https://language-detect-api-gde0d4a3a0cyeyae.centralus-01.azurewebsites.net/)).
Test endpoints:
https://<yourapp>.azurewebsites.net/ → should load your index.html.
https://<yourapp>.azurewebsites.net/predict → test POST with JSON:
Check if the app is accessible anywhere using the endpoints

<img width="1609" height="792" alt="image" src="https://github.com/user-attachments/assets/8cbb158f-8d09-408f-a356-639283ad9cbd" />

![WhatsApp Image 2025-10-02 at 07 24 42_68575ed3](https://github.com/user-attachments/assets/bf867e02-9fc5-4a36-a64e-d26076a73de5)

## Continuous Deployment

(Optional) Enable Continuous Deployment
If you want auto-deploy when you push to Docker Hub:
Go to your App Service in Azure.
In the left menu → Deployment Center.
Choose Source: Docker Hub.
Authenticate with your Docker Hub account.
Select your repo (bharath0805/language-detection-app) and branch/tag.
👉 Now, every time you docker push a new image, Azure will auto-update the running container. 🚀

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
