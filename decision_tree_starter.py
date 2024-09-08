"""
Have Fun!
- 189 Course Staff
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from pydot import graph_from_dot_data
import io
from scipy import stats

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number
COUNT = [10]


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred, self.prob = None, None, None  # for leaf nodes
        self.y_valid_ind = []


    @staticmethod
    def entropy(y):
        # TODO
        h_s = 0
        for val in np.unique(y):
            p_c = sum(y == val)/len(y)
            h_s += p_c*np.log2(p_c)
        return -h_s

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO
        y_left = y[np.where(X < thresh)[0]]
        y_left_entrop = DecisionTree.entropy(y_left)
        y_right = y[np.where(X >= thresh)[0]]
        y_right_entrop = DecisionTree.entropy(y_right)
        if(len(y_left) == 0 or len(y_right) == 0 or y_left_entrop == y_right_entrop):
            return eps
        H_after = (len(y_left) * y_left_entrop + len(y_right) * y_right_entrop)/(len(y_left) + len(y_right))
        info_gain = DecisionTree.entropy(y) - H_after
        if info_gain < eps:
            return eps
        return info_gain

    @staticmethod
    def acc(y_pred, y_test):
        t = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                t+=1
        return t #/len(y_pred)

    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO
        pass

    @staticmethod
    def gini_purification(X, y, thresh):
        # TODO
        pass

#Uses split_test to also just grab the y values that were split during this
    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

#Basically just splits X into 2 datasets where X0 < thresh and X1 >= thresh
    def split_test(self, X, idx, thresh): 
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        # TODO
        #self.data = X
        if self.max_depth == -1:
            self.data = y
            self.pred = stats.mode(y)[0]
            self.prob = len(np.where(y == 0)[0])/len(y)
            return

        info_gain = 0
        best_thresh = 0
        best_col = 0
        for idx in range(X.shape[1]):
            #For Numerical categorical Features
            info = 0
            thresh = 0
            for temp_thresh in np.unique(X[:,idx]):
                temp_info = self.information_gain(X[:,idx],y,temp_thresh)
                if info < temp_info:
                    info = temp_info
                    thresh = temp_thresh
            
            if info_gain < info:
                info_gain = info
                best_thresh = thresh
                best_col = idx
                
        self.thresh = best_thresh
        self.split_idx = best_col
        X0, y0, X1, y1 = self.split(X, y, self.split_idx, self.thresh)

        if len(y0) == 0 or len(y1) == 0:
            self.data = y
            self.pred = stats.mode(y)[0]
            self.prob = len(np.where(y == 0)[0])/len(y)
            return
        

        self.left = DecisionTree(max_depth= self.max_depth - 1)
        self.right = DecisionTree(max_depth= self.max_depth - 1)
        self.left.fit(X0, y0)
        self.right.fit(X1, y1)

        return

    def print2DUtil(self, space):
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
 
    # Base case
        if self.pred is not None:
            for i in range(COUNT[0], space):
                print(end="   ")
            print('Prediction:',self.pred)
            return
    
        # Increase distance between levels
        space += COUNT[0]
    
        # Process right child first
        self.right.print2DUtil(space)
    
        # Print current node after space
        # count
        #print()
        for i in range(COUNT[0], space):
            print(end="  ")
        print('Threshold:',self.thresh)
        for i in range(COUNT[0], space):
            print(end="  ")
        print('Column:',features[self.split_idx])
        
    
        # Process left child
        self.left.print2DUtil(space)
        
 
    def print2D(self):
    
        # space=[0]
        # Pass initial space count as 0
        self.print2DUtil(3)


    def pruning(self, y_valid):
        #print(type(self.left.pred))
        if self.left and self.left.pred is not None:
            if self.right.data is not None:
                self.data = list(self.left.data) + list(self.right.data)
            else:
                self.data = list(self.left.data)

            if len(self.left.y_valid_ind) == 0 or len(self.right.y_valid_ind) == 0:
                #self.pred = stats.mode(self.data)[0]
                return
            y_valid_data = y_valid[self.left.y_valid_ind + self.right.y_valid_ind]
            left_valid_data = y_valid[self.left.y_valid_ind]
            right_valid_data = y_valid[self.right.y_valid_ind]
            y_pred = self.acc([stats.mode(self.data)[0]]*len(y_valid_data), y_valid_data)
            y_pred_children = (self.acc([self.left.pred]*len(left_valid_data), left_valid_data) + self.acc([self.right.pred]*len(right_valid_data), right_valid_data))
            if  y_pred > y_pred_children:
                #print('pruned')
                #del self.left
                #del self.right
                self.pred = stats.mode(self.data)[0]
            return
        
        if self.left and self.left.left:
            self.left.pruning(y_valid)
            self.right.pruning(y_valid)

        return
    
    def prune(self, X_valid, y_valid):
        def predict_one(self, X_valid_one, ind): 
            if self.pred is not None:
                self.y_valid_ind.append(ind)
                return self.pred
            if X_valid_one[self.split_idx] < self.thresh:
                return predict_one(self.left, X_valid_one, ind)
            else:
                return predict_one(self.right, X_valid_one, ind)
        
        for point in range(X_valid.shape[0]):
            predict_one(self, X_valid[point], point)
        for _ in range(10):
            self.pruning(y_valid)


    def predict(self, X_test, want_prob=False):
        # TODO
        ans = []
        def predict_one(self, X_test_one, ind): 
            if self.pred is not None:
                self.y_valid_ind.append(ind)
                if want_prob == True:
                    return self.prob
                return self.pred
            if X_test_one[self.split_idx] < self.thresh:
                return predict_one(self.left, X_test_one, ind)
            else:
                return predict_one(self.right, X_test_one, ind)
        '''
        if self.y_valid is not None and want_prune == True:
            for point in range(X_test.shape[0]):
                #ans.append(predict_one(self, X_test[point], point))
                predict_one(self, X_test[point], point)
            for _ in range(10):
                self.pruning()
        
        '''
        for point in range(X_test.shape[0]):
            #ans.append(predict_one(self, X_test[point], point))
            ans.append(predict_one(self, X_test[point], point))

        return ans

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


#class BaggedTrees(BaseEstimator, ClassifierMixin):
class BaggedTrees(DecisionTree):

    def __init__(self,max_depth = 10, n=20):
        self.n = n
        self.max_depth = max_depth
        self.decision_trees = [DecisionTree(self.max_depth) for i in range(self.n)]

    def fit(self, X, y):
        for i in range(self.n):
            random_points = np.random.randint(0, len(X), len(X))
            self.decision_trees[i].fit(X[random_points],y[random_points])
        pass

    def prune(self, X_valid, y_valid):
        for i in range(self.n):
            self.decision_trees[i].prune(X_valid, y_valid)

    def predict(self, X_test):
        ans = []
        for i in range(self.n):
            ans.append(self.decision_trees[i].predict(X_test))
        return stats.mode(ans)[0]


class RandomForest(DecisionTree):

    def __init__(self, max_depth = 25, n_features = 15, m=150):
        np.random.seed(101)
        self.m = m
        self.cols = []
        self.n_features = n_features
        #self.d = np.random.choice(32, size = 32, replace = False)
        self.decision_tree = [DecisionTree(max_depth) for i in range(self.m)]

    def fit(self, X, y):
        for i in range(self.m):
            d = np.random.choice(X.shape[1], size = self.n_features, replace = False)
            self.cols.append(d)
            n = np.random.randint(0, len(X), len(X))
            self.decision_tree[i].fit(X[np.ix_(n,d)], y[n])
        pass

    def prune(self, X_valid, y_valid):
        for i in range(self.m):
            self.decision_tree[i].prune(X_valid, y_valid)
    
    def predict(self, X_test):
        ans = []
        for i in range(self.m):
            ans.append(self.decision_tree[i].predict(X_test[:,self.cols[i]]))
        return stats.mode(ans)[0]



class BoostedRandomForest(DecisionTree):

    def __init__(self, max_depth = 20, num_trees = 100, num_features = 20, learning_rate = 0):
        np.random.seed(101)
        self.num_trees = num_trees
        self.num_features = num_features
        self.cols = []
        self.alpha = []
        self.learning_rate = learning_rate
        self.decision_tree = [DecisionTree(max_depth) for i in range(self.num_trees)]

    def fit(self, X, y):
        w = [1/len(y)]*len(y)
        y_preds = []
        for i in range(self.num_trees):
            d = np.random.choice(X.shape[1], size = self.num_features, replace = False)
            self.cols.append(d)
            n = np.random.randint(0, len(X), len(X))
            self.decision_tree[i].fit(X[np.ix_(n,d)], y[n])
            y_preds = self.decision_tree[i].predict(X[:,d])

            error_sum = sum(w)
            error = 0
            a = (~np.equal(y_preds, y)).astype(int)
            wrong_inds = np.where(a == 1)[0]
            for ind in wrong_inds:
                error += w[ind]/error_sum

            alpha = 0
            if self.learning_rate == 0:
                alpha = 0.5*np.log((len(y) - 1)*(1 - error)/error)
            else:
                alpha = self.learning_rate*np.log((1 - error)/error)
            
            #2 ways to do this
            #Way 1
            if alpha > 0:
                for i in range(len(w)):
                    if i in wrong_inds:
                        w[i] = w[i]*np.exp(alpha)
                    else:
                        w[i] = w[i] #* np.exp(-alpha) #Maybe try not multiplying by anything too
                self.alpha.append(alpha)
            else:
                self.alpha.append(0)
        pass

    def prune(self, X_valid, y_valid):
        for i in range(self.num_trees):
            self.decision_tree[i].prune(X_valid, y_valid)
    
    def predict(self, X_test):
        ans_0 = []
        ans_1 = []
        for ind in range(self.num_trees):
            probs = self.decision_tree[ind].predict(X_test[:,self.cols[ind]], want_prob=True)
            ans_0.append([self.alpha[ind] * i for i in probs])
            ans_1.append([self.alpha[ind] * (1 - i) for i in probs])
        ans_0 = np.sum(ans_0, axis = 0)/self.num_trees
        ans_1 = np.sum(ans_1, axis = 0)/self.num_trees
        true_ans = []
        for i in range(len(X_test)):
            if ans_0[i] > ans_1[i]:
                true_ans.append(0)
            else:
                true_ans.append(1)

        return true_ans

class GradientBoosting(DecisionTree):
    def __init__(self, max_depth = 20, num_trees = 100, num_features = 20, learning_rate = 1):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.decision_tree = []
        self.f0 = 0

    def fit(self, X, y):
        self.f0 = np.round(np.mean(y))
        model = np.round(np.mean(y))
        for i in range(self.num_trees):
            p = -np.exp(model)/(1 + np.exp(model))
            #loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            negative_grad = y / p - (1 - y) / (1 - p)

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X,negative_grad)
            h_i = tree.predict(X)
            model = model + self.learning_rate*h_i
            self.decision_tree.append(tree)
    
    def predict(self, X_test):
        return self.f0 + self.learning_rate*np.sum([tree.predict(X_test) for tree in self.decision_tree], axis=0)
        