'''
The output of the file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'
'''

from collections import defaultdict
import glob
import re
import scipy.io

import nltk
nltk.download('punkt')
nltk.download("stopwords")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************
# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    texts = []
    for filename in filenames:
        with open(filename, encoding="latin-1") as f:
            text = f.read() # Read in text from file
            text = text.replace("Subject: ","")
            texts.append(text)
    df = pd.DataFrame(data=texts,columns=["texts"])
    x = df["texts"].values
    stops = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stops, lowercase=True, max_features=100)
    design_matrix = vectorizer.fit_transform(x).toarray()
    df2 = pd.DataFrame(design_matrix, columns = vectorizer.get_feature_names())
    df2.to_csv(os.path.join(os.getcwd(),r'tfidf.csv'))
    print("matrix shape",design_matrix.shape)
    return design_matrix

# ************** Script starts here **************
spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(i) + '.txt' for i in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

x = np.vstack((spam_design_matrix,ham_design_matrix))
y = [1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)

file_dict = {}
file_dict['training_data'] = x
file_dict['training_labels'] = y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict)



