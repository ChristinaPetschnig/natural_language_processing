# Natural Language Processing

# Social Media Content Analysis Project

## Research Question

How effective is natural language processing (NLP) in classifying social media content promoting eating disorders?

## Overview
This repository is dedicated to the analysis and identification of social media content related to eating disorders. We've developed models using machine learning techniques to classify posts based on whether they glorify eating disorders. The modules in this project carry out data pre-processing, feature extraction, training, evaluation, and persistence of machine learning models.

### Repository Structure
Below is a high-level overview of the structure of this project, detailing all the folders and files contained within it:

- `requirements.txt`: This file lists all the dependencies required to run the project's code. You can install the necessary libraries and packages with specific versions using `pip install -r requirements.txt` to set up your environment.

- Jupyter Notebooks:
    - `naive_bayes_random_forest.ipynb`: A comprehensive notebook that guides you through the preprocessing steps, feature extraction, training, and evaluating both the Naive Bayes and Random Forest models.
    - `bert_notebook.ipynb`: This notebook focuses on leveraging BERT, a state-of-the-art transformer model, for text classification tasks associated with our dataset.

- Data Folder: 
    - `internet_data.csv`: A dataset of manually collected social media posts.
    - `data_promoting_eating_disorders.csv`: Contains synthetic social media posts that endorse eating disorders.
    - `neutral_data.csv`: Includes synthetic social media posts that are neutral toward eating disorders.
    - `all_social_media_posts.csv`: A compiled dataset of all the unique social media posts collected and generated, encompassing all categories mentioned above.

- Models Folder:
    - `best_naive_bayes_model.joblib`: The saved trained Naive Bayes model after hyperparameter tuning and evaluation.
    - `best_random_forest_model.joblib`: The saved trained Random Forest model post hyperparameter optimization and performance assessment.

#### BERT Model
The BERT model, due to its size, is not included directly in this repository. However, it can be accessed and downloaded from Google Drive via this link:
[Access the BERT model here](https://drive.google.com/drive/folders/1q7Q7oZbrY3p5Mu4pBGssp4ogMvTEciwD?usp=sharing)

### Getting Started

To run the notebooks and use the models, please follow these steps:

1. Install the requirements by running `pip install -r requirements.txt` in your command line interface.
2. Launch and execute the `naive_bayes_random_forest.ipynb` Jupyter notebook. Ensure the `data` folder is present in the same directory as the notebook or modify the path in the code as needed.
3. Proceed to run the `bert_notebook.ipynb` with the same considerations for data pathing. Adjust the Adam optimizer based on your system's architecture if necessary.

### Key Contents of the Notebooks

#### Naive Bayes and Random Forest Notebook:
- Data de-duplication and initial exploratory analysis.
- Text preprocessing methods including tokenization, stopword removal, and stemming.
- Feature extraction using TF-IDF vectorization.
- Visualizations of data distribution and most frequent terms.
- Hyperparameter tuning, model evaluation, and cross-validation for both Naive Bayes and Random Forest classifiers.
- Persistence of the best-performing models using joblib.

#### BERT Notebook:
- Loading and processing the dataset for BERT.
- Tokenization utilizing the BERT tokenizer.
- Application of KFold cross-validation to assess BERT's performance.
- Calculation of accuracy, precision, recall, and F1 scores, along with confusion matrices for evaluation.
- Training the final BERT model on the complete dataset and evaluating its performance on a test set.

## Extras:

### Ethical Considerations:

In the process of manually gathering social media content, we exercised adherence to ethical standards, particularly concerning privacy. We conscientiously avoided the collection of personal data, thereby safeguarding the privacy of individuals and maintaining the integrity of our data collection methodology.
