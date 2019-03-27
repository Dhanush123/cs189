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

from savecsv import preds_to_csv

import ray
import itertools
import numbers

class Node:
    def __init__(self, feature=None, thresh=None, label=None, left=None, right=None):
        self.feature = feature
        self.thresh = thresh
        self.label = label
        self.left = left
        self.right = right

class DecisionTree(object):

    def __init__(self, max_depth=15, header=""):
        """
        TODO: initialization of a decision tree
        """
        self.trained_tree = None
        self.max_depth = max_depth
        self.header = header

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
        left_filter = np.where(x[:,feature] > thresh)[0]
        right_filter = np.where(x[:,feature] <= thresh)[0]
        # print("split feature",x.shape,y.shape,left_filter.shape,right_filter.shape)   
        return x[left_filter], y[left_filter], x[right_filter], y[right_filter]
        
    @staticmethod
    def best_split(x, y):
        best_gain, best_thresh, best_feature = 0, 0, 0
        num_features = x.shape[1]
        features = range(num_features)
        threshs = [value for feature in features for value in np.unique(x[:,feature])]
        results = []
        for feature, value in itertools.product(features,threshs):
        # for feature in range(num_features):
        #     for value in np.unique(x[:,feature]):
            _, y_left, _, y_right = DecisionTree.split(x, y, feature, value)
            if len(y_left) > 0 and len(y_right) > 0:
                ig = DecisionTree.info_gain(y, y_left, y_right)
                if ig > best_gain:
                    best_gain = ig
                    best_thresh, best_feature = value, feature
        return best_feature, best_thresh

    def grow_tree(self, x, y, depth=0):
        # print("currently on depth ",depth)
        #in case leaf isn't pure, majority value is safe choice
        majority_value = Counter(y).most_common(1)[0][0]
        if DecisionTree.entropy(y) == 0 or len(y) < 300 or depth == self.max_depth:
            return Node(label=majority_value)

        feature, thresh = DecisionTree.best_split(x, y)
        x_left, y_left, x_right, y_right = DecisionTree.split(x, y, feature, thresh)

        if len(y_left) > 0 and len(y_right) > 0:
            left = self.grow_tree(x_left, y_left, depth+1)
            right = self.grow_tree(x_right, y_right, depth+1)
            label = "feature # "+str(feature)+" > "+str(thresh)
            return Node(feature, thresh, label, left, right)
        else:
            return Node(label=majority_value)     

    def fit(self, x, y):
        """
        TODO: fit the model to a training set. Think about what would be 
        your stopping criteria
        """
        self.trained_tree = self.grow_tree(x, y)

    # @ray.remote  
    def predict_single(self, node, sample):
        # if isinstance(node.label, numbers.Number):
        #     return node.label
        # else:
        #     # print("pred single",node.label,node.feature,node.left,node.right)   
        #     node = node.left if sample[node.feature] > node.thresh else node.right
        #     return self.predict_single.remote(self,node, sample)
        #print("prediction!!",sample,sample[node.feature],sample.shape)
        cur_node = node
        while not isinstance(cur_node.label, numbers.Number):
            # print("pred single",node.label,node.feature,node.left,node.right)   
            cur_node = cur_node.left if sample[cur_node.feature] > cur_node.thresh else cur_node.right
            #print("node.label",cur_node.label)#,cur_node.left.label,cur_node.right.label)
        return cur_node.label

    def predict(self, x, mode="train",tree_type="decisiontree",rf=None):
        """
        TODO: predict the labels for input data 
        """
        num_samples = x.shape[0]
        y = np.zeros((num_samples))
        for row in range(num_samples):
            pred = None
            if tree_type == "decisiontree":
                pred = self.predict_single(self.trained_tree, x[row,:])
            elif tree_type == "randomforest":
                pred = self.predict_single(rf, x[row,:])
            # print("pred",pred)
            y[row] = pred
        if mode == "test" and tree_type == "decisiontree":
            preds_to_csv(y,self.header)
        else:
            return y

    def print_tree(self):
        def print_level(node):
          depth = 0   
          cur_level = [node]
          while cur_level:
            next_level = []
            for node in cur_level:
              print("Depth",str(depth)+":",str(node.label))
              if node.left: 
                next_level.append(node.left)
              if node.right: 
                next_level.append(node.right)
            depth += 1
            cur_level = next_level
            print("\n")
        print_level(self.trained_tree)


class RandomForest(DecisionTree):
    
    def __init__(self,num_trees=0,rand_features=None,max_depth=15,header=""):
        """
        TODO: initialization of a random forest
        """
        super(RandomForest, self).__init__()
        self.trained_trees = [None] * num_trees
        self.num_trees = num_trees
        self.rand_features = rand_features
        self.max_depth = max_depth  
        self.header = header
        ray.init()
        ray.register_custom_serializer(RandomForest, use_pickle=True)


    @ray.remote   
    def parallel_grow_tree(self, x, y):
        feature_indices = np.random.choice([i for i in range(x.shape[1])],\
            size=int(np.sqrt(x.shape[1])),replace=False)
        sample_indices = np.random.randint(x.shape[0], size=x.shape[0])
        x = x[:,feature_indices]
        x = x[sample_indices, :]
        return super(RandomForest, self).grow_tree(x, y)

    def fit(self, x, y):
        """
        TODO: fit the model to a training set.
        """
        self.trained_trees = ray.get([self.parallel_grow_tree.remote(self,x,y) for _ in range(self.num_trees)])

    @ray.remote
    def parallel_predict(self, x, model):
        return super(RandomForest, self).predict(x,tree_type="randomforest",rf=model)
    
    def predict(self, x, mode="train"):
        """
        TODO: predict the labels for input data 
        """
        # print("confirming tree properties",len(self.trained_trees),type(self.trained_trees[0]))
        all_preds = np.asarray(np.matrix(ray.get([self.parallel_predict.remote(self,x,self.trained_trees[i]) for i in range(self.num_trees)])))
        final_preds = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, all_preds)
        if mode == "test":
            preds_to_csv(final_preds,self.header)
        else:
            return final_preds










