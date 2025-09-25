# Text Sentiment Analysis with BERT

## 📋 Project Overview  
The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. The goal is to train a model to automatically detect negative reviews. We have a dataset of IMBD movie reviews with polarity labelling to build a model for classifying positive and negative reviews.

## 🔥 Summary  
Used pretrained BERT models (via Hugging Face Transformers) to extract high-quality text embeddings

Trained gradient boosting classifiers (LightGBM and CatBoost), and a Logistic Regression model on the BERT features 

Best model during cross validation was the Logistic Regression model:  
F1 score - 0.86  
Accuracy score - 0.86  
ROC AUC score = 0.94   

I also tested the Logistic Regression model on a small set of novel samples, and it peformed with accuracy score of 0.875

## 🛠️ Tech Stack & Libraries

**Python 3.10+**

Machine Learning & Modeling:  
- `scikit-learn` – Logistic regression, evaluation metrics, grid search, and baseline models  
- `LightGBM` – Efficient gradient boosting for large datasets  
- `CatBoost` – Gradient boosting with native support for categorical features  
- `joblib` – Model persistence and parallel computation

Natural Language Processing:  
- `transformers` (by Hugging Face) – Loading and using BERT models, tokenizers, and configs  
- `torch` (PyTorch) – Backend tensor computations and model support

Data Manipulation & Analysis:  
- `pandas` – DataFrames, querying, and string operations  
- `numpy` – Efficient numerical computation and file I/O (e.g., `np.savez_compressed`, `np.load`)  
- `math` – Standard library math utilities

Visualization:  
- `matplotlib` – Plotting model results and exploratory visuals

Utilities:  
- `tqdm` – Progress bars for training and data processing loops  
- `json` – Configuration and results serialization (standard library)


## 🚀 How to Run  
Clone this repo and install requirements:
pip install -r requirements.txt

⚙️ Running Notebooks  
To run the Jupyter notebooks in this project, ensure you have the required packages installed. Conda is recommended, especially on Windows, but any Python virtual environment will work.

### Create and activate your environment:

**Using Conda (recommended on Windows):**

```
conda create --name project_name_env python=3.11
conda activate project_name_env
```



You will have to run the data_pre_processing.py and script on a platform with GPU capacity first, as this creates the BERT text embeddings each of the models requie to train on. After that you can run the *_tuning_cv.py scripts for each model. Order does not matter for this part. The resulting models will be refit on the full training set and saved to the models subdirectory with the selected hyperparameter values set (assuming you are running them from the project directory. If you are running the *_cv_tuning.py scripts on google colab you will have to manually move them from your downloads to the models subdirectory.)

After training and hyperparameter tuning each of the models you can test the best model (model_log) by running model_log_test.py and compare it to a dummy model by running dummy_model_test.py. 

Next you will have to run cust_reviews_pre_processing.py. It would be preferable to run that script on a platform with GPU availability. After that, you can run cust_reviews_test.py from the project directory. 

All .py scripts which are not in the src folder are the scripts that execute the code (with the exception of the pre_processing scripts that have to be run with GPU support) and should be run from the project directory where they are located. 

Each script that requires GPU to run in a timely fashion was written to be compatible with running on google colab. I provided a notebook script with comment instructions for each.py script to make this process smoother. You can simply upload the notebook script to google colab, and upload the files as instructed in that notebook's comments. I ran these scripts on L4 GPU, but T4 would likely suffice. The scripts that need to be (or would benefit from) running with GPU are:

data_pre_processing.py
model_cat_cv_tuning.py
model_light_cv_tuning.py
cust_reviews_pre_processing.py

## 📊 Results  
Results are documented in the results_and_analysis.ipynb notebook which should be run from the project directory. 

## 🤝 Datasets   
The only data you need to start is imdb_reviews.tsv. However, you will create an embedded_features.npz file and an embedded_custom_reviews.npz with the pre_processing .py scripts. If you run the pre_processing .py scripts from colab they will be downloaded to your computer and you must manually move them to the project directory. If you run those scripts from the project directory, they will simply be saved there. 


## 🗂️ Project Structure  
All files live in the main directory for easy access:
```
text_sentiment_analysis_bert/
│
├── cv_tuning_results/              # Grid search result objects
│   ├── grid_search_cat.pkl
│   ├── grid_search_light.pkl
│   └── grid_search_log.pkl
│
├── models/                         # Serialized trained models
│   ├── model_cat.json
│   ├── model_light.pkl
│   └── model_log.pkl
│
├── notebooks/                      # Jupyter notebooks for exploration & training
│   ├── cust_reviews_pre_processing_colab.ipynb
│   ├── data_pre_processing_colab.ipynb
│   ├── model_cat_cv_tuning_colab.ipynb
│   ├── model_light_cv_tuning_colab.ipynb
│   └── results_and_analysis.ipynb
│
├── param_grids/                    # Hyperparameter search grids
│   ├── param_grid_cat.pkl
│   ├── param_grid_light.pkl
│   └── param_grid_log.pkl
│
├── src/                            # Reusable code modules only
│   ├── cust_reviews_pre_processing.py
│   ├── data_pre_processing.py
│   └── model_utils.py
│
├── model_cat_cv_tuning.py         # CV tuning scripts (moved from src/)
├── model_light_cv_tuning.py
├── model_log_cv_tuning.py
│
├── cust_reviews_test.py           # Test scripts (moved from src/)
├── dummy_model_test.py
├── model_log_test.py
│
├── imdb_reviews.tsv               # Raw dataset
├── README.md                      # Project documentation
└── requirements.txt               # Python package dependencies
```

## 🧠 Authors

- Developed by Brett Kunkel | [www.linkedin.com/in/brett-kunkel](www.linkedin.com/in/brett-kunkel) | brttkunkel@gmail.com

---

## 📜 License

This project is licensed under the MIT License.