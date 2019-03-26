"""
This is the starter code and some suggested architecture we provide you with. 
But feel free to do any modifications as you wish or just completely ignore 
all of them and have your own implementations.
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats

import random
import inspect

from savecsv import results_to_csv

class Node:
    def __init__(self, feature=None, thresh=None, label=None, left=None, right=None):
        self.feature = feature
        self.thresh = thresh
        self.label = label
        self.left = left
        self.right = right

class DecisionTree:

    def __init__(self):
        """
        TODO: initialization of a decision tree
        """
        self.trained_tree = None

    @staticmethod
    def entropy(y):
        _, counts = np.unique(y, return_counts=True)
        return stats.entropy(counts, base=2)

    @staticmethod
    def info_gain(y, y_left, y_right):
        entropy_before = DecisionTree.entropy(y)
        lyl, lyr = len(y_left), len(y_right)
        entropy_after = (lyl*DecisionTree.entropy(y_left) + lyr*DecisionTree.entropy(y_right))/(lyl + lyr)
        return entropy_before - entropy_after

    @staticmethod
    def split(x, y, feature, thresh):
        left_filter = np.where(x > thresh)[0]
        right_filter = np.where(x <= thresh)[0]
        # print("split feature",x.shape,y.shape,left_filter.shape,right_filter.shape)   
        return x[left_filter], y[left_filter], x[right_filter], y[right_filter]
        
    @staticmethod
    def best_split(x, y):
        best_gain, best_thresh, best_feature = 0, 0, None
        num_features = x.shape[1]
        for feature in range(num_features):
            # x_feature = x[:,feature]
            # argsrt = np.argsort(x_feature)
            # sorted_x = x_feature[argsrt]
            # sorted_y = y[argsrt]
            for value in range(x.shape[0]):
                _, y_left, _, y_right = DecisionTree.split(x, y, feature, value)
                if len(y_left) > 0 and len(y_right) > 0:
                    ig = DecisionTree.info_gain(y, y_left, y_right)
                    if ig > best_gain:
                        best_gain = ig
                        best_thresh, best_feature = value, feature
        return best_feature, best_thresh

    def grow_tree(self, x, y, depth):
        # print("currently on depth ",depth)
        #in case leaf isn't pure, majority value is safe choice
        majority_value = Counter(y).most_common(1)[0][0]
        # print("maj val",majority_value)
        #250 ~ 5% of training set   

        #DecisionTree.entropy(y) == 0 or
        if DecisionTree.entropy(y) == 0 or len(y) < 250 or depth == 15:
            # print("here?")
            return Node(label=majority_value)

        # print("or here")
        feature, thresh = DecisionTree.best_split(x, y)
        x_left, y_left, x_right, y_right = DecisionTree.split(x, y, feature, thresh)

        if len(y_left) > 0 and len(y_right) > 0:
            left = self.grow_tree(x_left, y_left, depth+1)
            right = self.grow_tree(x_right, y_right, depth+1)
            return Node(feature, thresh, "feature # "+str(feature)+" > "+str(thresh), left, right)
        else:
            return Node(label=majority_value)     

    def fit(self, x, y):
        """
        TODO: fit the model to a training set. Think about what would be 
        your stopping criteria
        """
        self.trained_tree = self.grow_tree(x, y, 0)

    def predict_single(self, node, sample):
        # print(node.label,node.feature,node.left,node.right)
        if node.label == 0 or node.label == 1:
            return node.label
        else:
            node = node.left if sample[node.feature] > node.thresh else node.right
            return self.predict_single(node, sample)

    def predict(self, x, mode="test"):
        """
        TODO: predict the labels for input data 
        """
        num_samples = x.shape[0]
        y = np.zeros((num_samples))
        for row in range(num_samples):
            pred = self.predict_single(self.trained_tree, x[row,:])
            # print("pred",pred)
            y[row] = pred
        if mode == "test":
            results_to_csv(y)
        else:
            return y

    # def __repr__(self):
    #     """
    #     TODO: one way to visualize the decision tree is to write out a __repr__ method
    #     that returns the string representation of a tree. Think about how to visualize 
    #     a tree structure. You might have seen this before in CS61A.
    #     """
    #     return 0


class RandomForest():
    
    def __init__(self):
        """
        TODO: initialization of a random forest
        """

    def fit(self, X, y):
        """
        TODO: fit the model to a training set.
        """
        return 0
    
    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        return 0