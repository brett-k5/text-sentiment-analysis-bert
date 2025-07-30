🚀 Text Sentiment Analysis with BERT + Gradient Boosting
📋 Project Overview
This project tackles sentiment analysis by combining powerful BERT embeddings with classic machine learning models. The pipeline turns raw text into deep contextual features using BERT, then feeds those features into gradient boosting and logistic regression models to classify sentiment.

🔥 What I Did
Used pretrained BERT models (via Hugging Face Transformers) to extract high-quality text embeddings ✨

Trained LightGBM and CatBoost gradient boosting classifiers on the BERT features 🌲

Also trained a simple but effective Logistic Regression model ⚡

Evaluated everything using the ROC AUC metric — scored a strong 0.94! 🎯

🛠️ Tech Stack & Libraries
Python 🐍

Transformers (Hugging Face) for BERT embeddings 🤗

LightGBM & CatBoost for gradient boosting 💨

scikit-learn for logistic regression and evaluation 📊

PyTorch for model backend 🧠

Pandas, NumPy, Matplotlib, Seaborn for data wrangling & visualization 📈

🚀 How to Run
Clone this repo and install requirements:

pip install -r requirements.txt
Prepare your dataset (text + labels).

Run the notebooks/scripts to generate embeddings and train models.

Check out model performance and visualizations!

🎯 Results
Achieved a ROC AUC of 0.94 — which means the model is excellent at distinguishing sentiment classes! 🥳

📦 Required File Uploads for Running Scripts in Google Colab
If you run any of the .py scripts in Google Colab, there will be multiple uploads using files.upload() for each script. The files that must be uploaded for each upload are listed with the corresponding scripts they must be uploaded for below.

First Import: Python scripts and utility modules.

Second Import: Data files and model artifacts needed at runtime.

🔹 log_reg_tuning.py
First Import:

log_reg_tuning.py

Second Import:

embedded_features.npz

train_target.csv

🔹 lightgbm_tuning.py
First Import:

lightgbm_tuning.py

Second Import:

embedded_features.npz

train_target.csv

🔹 catboost_tuning.py
First Import:

catboost_tuning.py

Second Import:

embedded_features.npz

train_target.csv

🔹 dummy_model_test.py
First Import:

dummy_model_test.py

model_utils.py

Second Import:

embedded_features.npz

train_target.csv

test_target.csv

model_log.pkl

🔹 model_log_test.py
First Import:

model_log_test.py

model_utils.py

Second Import:

embedded_features.npz

train_target.csv

test_target.csv

🔹 model_log_cust_reviews_test.py
First Import:

model_log_cust_reviews_test.py

Second Import:

custom_reviews.csv

embedded_custom_reviews.npz

model_log.pkl

🤝 Contact
Feel free to reach out if you want to chat about the project or need help getting started!

🗂️ Project Structure
All files live in the main directory for easy access:
```
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── notebook.ipynb           # Main Jupyter notebook with code & analysis
├── sprint_14_project.ipynb  # Additional notebook(s)
├── data/                    # (Optional) Place to add your datasets if needed
└── scripts/                 # (Optional) Python scripts for training & evaluation
