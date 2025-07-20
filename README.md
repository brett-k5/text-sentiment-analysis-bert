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
