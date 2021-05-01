import numpy as np
from DatasetUtils import extractFeatures
import math

class DecisionNode():

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

class ClassificationTree(object):

    def __init__(self):
        self.root = None
        self.MIN_IMPURITY = 1e-7
        self.max_depth = math.inf

    def fit(self, X, y):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self.buildTree(X, y)

    def buildTree(self, X, y, current_depth=0):
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= 2 and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    Xy1, Xy2 = self.divideFeatureByThreshold(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self.impurity(y, y1, y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }

        if largest_impurity > self.MIN_IMPURITY:
            true_branch = self.buildTree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self.buildTree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch)

        leaf_value = self.mostLabel(y)

        return DecisionNode(value=leaf_value)

    def predictValue(self, reqType, url, body):
        x = np.array(extractFeatures(reqType,url,body))
        return self.predict_value(x)

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]

        branch = tree.false_branch
        if feature_value >= tree.threshold:
            branch = tree.true_branch

        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)

    def impurity(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = self.entropy(y)
        info_gain = entropy - p * self.entropy(y1) - (1 - p) * self.entropy(y2)

        return info_gain

    def mostLabel(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def divideFeatureByThreshold(self, X, idxFeature, threshold):

        X_1 = np.array([sample for sample in X if sample[idxFeature] >= threshold])
        X_2 = np.array([sample for sample in X if sample[idxFeature] < threshold])

        return np.array([X_1, X_2])

    def entropy(self,y):
        log2 = lambda x: math.log(x) / math.log(2)
        labels = np.unique(y)
        entropy = 0
        for label in labels:
            count = len(y[y == label])
            p = count / len(y)
            entropy += -p * log2(p)
        return entropy