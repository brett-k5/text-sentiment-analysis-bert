Text Sentiment Analysis with BERT
  
📋 Project Overview  
The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. The goal is to train a model to automatically detect negative reviews. We have a dataset of IMBD movie reviews with polarity labelling to build a model for classifying positive and negative reviews.

🔥 Summary  
Used pretrained BERT models (via Hugging Face Transformers) to extract high-quality text embeddings

Trained gradient boosting classifiers (LightGBM and CatBoost), and a Logistic Regression model on the BERT features 

Best model during cross validation was the Logistic Regression model:  
F1 score - 0.86  
Accuracy score - 0.86  
ROC AUC score = 0.94   

I also tested the Logistic Regression model on a small set of novel samples, and it peformed with accuracy score of 0.875

🛠️ Tech Stack & Libraries
- Python 

- Transformers (Hugging Face) for BERT     
embeddings 

- LightGBM & CatBoost for gradient boosting 

- scikit-learn for logistic regression and evaluation

- PyTorch for model backend 

- Pandas, NumPy, Matplotlib, Seaborn for data wrangling & visualization


🚀 How to Run  
Clone this repo and install requirements:
pip install -r requirements.txt

You will have to run the data_pre_processing.py and script on a platform with GPU capacity first, as this creates the BERT text embeddings each of the models requie to train on. After that you can run the *_tuning_cv.py scripts for each model. Order does not matter for this part. The resulting models will be refit on the full training set and saved to the models subdirectory with the selected hyperparameter values set (assuming you are running them from the project directory. If you are running the *_cv_tuning.py scripts on google colab you will have to manually move them from your downloads to the models subdirectory.)

After training and hyperparameter tuning each of the models you can test the best model (model_log) by running model_log_test.py and compare it to a dummy model by running dummy_model_test.py. 

Next you will have to run cust_reviews_pre_processing.py. It would be preferable to run that script on a platform with GPU availability. After that, you can run cust_reviews_test.py from the project directory. 

All .py scripts run from the project directory should be run by entering python -m src.*.py into your IDE terminal from the project directory.

Each script that requires GPU to run in a timely fashion was written to be compatible with running on google colab. I provided a notebook script with comment instructions for each.py script to make this process smoother. You can simply upload the notebook script to google colab, and upload the files as instructed in that notebook's comments. I ran these scripts on L4 GPU, but T4 would likely suffice. The scripts that need to be (or would benefit from) running with GPU are:

data_pre_processing.py
model_cat_cv_tuning.py
model_light_cv_tuning.py
cust_reviews_pre_processing.py

📊 Results  
Results are documented in the results_and_analysis.ipynb notebook which should be run from the project directory. 

🤝 Datasets   
The only data you need to start is imdb_reviews.tsv. However, you will create an embedded_features.npz file and an embedded_custom_reviews.npz with the pre_processing .py scripts. If you run the pre_processing .py scripts from colab they will be downloaded to your computer and you must manually move them to the project directory. If you run those scripts from the project directory, they will simply be saved there. 

🤝 Contact  
Feel free to reach out if you want to chat about the project or need help getting started!

🗂️ Project Structure  
All files live in the main directory for easy access:
```
text_sentiment_analysis_bert/
│
├── cv_tuning_results/ # Grid search result objects
│ ├── grid_search_cat.pkl
│ ├── grid_search_light.pkl
│ └── grid_search_log.pkl
│
├── models/ # Serialized trained models
│ ├── model_cat.json
│ ├── model_light.pkl
│ └── model_log.pkl
│
├── notebooks/ # Jupyter notebooks for exploration & training
│ ├── cust_reviews_pre_processing_colab.ipynb
│ ├── data_pre_processing_colab.ipynb
│ ├── model_cat_cv_tuning_colab.ipynb
│ ├── model_light_cv_tuning_colab.ipynb
│ └── restults_and_analysis.ipynb
│
├── param_grids/ # Hyperparameter search grids
│ ├── param_grid_cat.pkl
│ ├── param_grid_light.pkl
│ └── param_grid_log.pkl
│
├── src/ # Source code modules
│ ├── cust_reviews_pre_processing.py
│ ├── cust_reviews_test.py
│ ├── data_pre_processing.py
│ ├── dummy_model_test.py
│ ├── model_cat_cv_tuning.py
│ ├── model_light_cv_tuning.py
│ ├── model_log_cv_tuning.py
│ ├── model_log_test.py
│ └── model_utils.py
│
├── imdb_reviews.tsv # Raw dataset
├── README.md # Project documentation
└── requirements.txt # Python package dependencies
