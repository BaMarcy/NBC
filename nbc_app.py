# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 00:59:51 2022

@author: Deepworker
"""

import streamlit as st
import pandas as pd
import numpy as np
from urllib.error import URLError


st.title('Breast Cancer Classification with Naïve Bayes Classifier')

st.markdown("""
    This is a native implementation of Naïve Bayes Classifier from scratch in Python.

    Check the code on [GitHub](https://www.linkedin.com/in/marcellbalogh/) and connect on [LinkedIn.](https://www.linkedin.com/in/marcellbalogh/)
    
    
""")

dataset = '''

The tagged data set is from the "Breast Cancer Wisconsin (Diagnostic) Database" freely available in python's sklearn library, for details see:  
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

* Number of Samples: 569  
* Number of Features: 30 
* Number of Classes: 2 

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. Ten real-valued features are computed for each cell nucleus. The mean, standard error and 'worst' or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. All feature values are recoded with four significant digits.

The two target classes correspond to negative outcomes (Benign) and positive outcomes (Malignant). '''

class_code = '''class NaiveBayes():
    def __init__(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6

    def fit(self, X):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        probs = np.zeros((self.num_examples, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density_function(X, self.classes_mean[str(c)], self.classes_variance[str(c)])
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.eps))
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs'''
    
run_code = '''
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
import numpy as np

data = load_breast_cancer()
X = data["data"]
Y = data["target"]

NBC = NaiveBayes(X, Y)
NBC.fit(X)
preds = NBC.predict(X)

NBC_sklearn = GaussianNB()
NBC_sklearn.fit(X, Y)
preds_sklearn = NBC_sklearn.predict(X)


print(f"Accuracy: { sum( preds==Y ) / X.shape[0] }")
print(f"Accuracy (sklearn): { sum( preds_sklearn==Y ) / X.shape[0] }")
'''


with st.expander("The dataset:", expanded=True):
    
     
    col1, col2 = st.columns(2)

    with col1:
        st.header("Visualization")
        st.image("breast_cancer_dataset_PCA.png")
        st.image("breast_cancer_dataset_heatmap.png")
    
    with col2:
        st.header("Description")
        st.write(dataset)
    
    
     
with st.expander("See Naïve Bayes Classifier implementation from scratch in Python:"):
     st.code(class_code, language='python')
     
with st.expander("Compare the implementation with sklearn:"):
     st.code(run_code, language='python')
     st.text('>>> Accuracy: 0.9402460456942003')
     st.text('>>> Accuracy (sklearn): 0.9420035149384886')
    
     
