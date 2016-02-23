# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sets import Set
import matplotlib.pyplot as plt
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from scipy.optimize import fmin_powell
from sklearn.cross_validation import KFold 
import xgboost as xg
import operator
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
# remove comment from the line below if using ipython notebook
#%matplotlib inline



       
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def draw_feature_map(gbm):
	print "Feature importances"
	importance = gbm.get_fscore(fmap='xgb.fmap')
	importance = sorted(importance.items(), key=operator.itemgetter(1))
	df = pd.DataFrame(importance, columns=['feature', 'fscore'])
	df['fscore'] = df['fscore'] / df['fscore'].sum()
	plt.figure()
	df.plot()
	df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 20))
	plt.title('XGBoost Feature Importance')
	plt.xlabel('relative importance')
	plt.gcf().savefig('feature_importance_xgb.png')





def xgb_model(X_train,y_train,X_test,y_test,save=False):
	'''
		Function to apply the xgb model to the split train dataset to get the score
	'''



def add_features():
	print "Adding Features..."
	all_data = train.append(test)
	
	
	return train_new,test_new

def clean_data():
	categorical_cols = []
	numeric_cols = []
	for col in train.columns:
		if train[col].dtype == 'object':
			categorical_cols.append(col)
		else:
			numeric_cols.append(col)

	train.fillna(-1,inplace=True)
	# label encoding
	enc = preprocessing.LabelEncoder()
	for col in categorical_cols:
		train[col] = enc.fit_transform(train[col])

def plot_scatter():
	'''
		Function to save scatter plots 
	'''
	print "plotting correlation plots.."
	for col in train.iloc[:,1:].columns:
		plt.figure()
		train.plot(kind='hexbin',x=col,y='Response',gridsize=10)
		plt.savefig("histograms/"+col+'.png')
		plt.close()
	

if __name__ == "__main__":
	print "Reading Train Data..."
	train = pd.read_csv('train.csv')
	print "Cleaning Data..."
	clean_data()
	print "Done!! Exiting Now..."

