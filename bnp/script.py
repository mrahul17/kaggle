# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xg
from sets import Set
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold 
import matplotlib.pyplot as plt
import operator
from sklearn import preprocessing
import sys

class Data:
	x = pd.DataFrame()
	y = pd.DataFrame()
	z = pd.DataFrame()
       
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





def xgb_model(X_train,y_train,X_test,y_test=None,save=False):
	'''
		Function to apply the xgb model to the split train dataset to get the score
	'''
	if not save:
		num_rounds = 600
	else:
		num_rounds = 700
	
	# setup parameters for xgboost
	params = {}
	params['objective'] = 'binary:logistic'
	params["eta"] = 0.02
	#params["min_child_weight"] = 240
	params["subsample"] = 0.9
	params["colsample_bytree"] = 0.85
	params["silent"] = 1
	params["max_depth"] = 6

	print "Training the model now... This will take really long..."
	gbm = xg.train(params,xg.DMatrix(X_train, y_train),num_rounds)

	print "Predicting on test data..."
	test_preds = gbm.predict(xg.DMatrix(X_test),ntree_limit=gbm.best_iteration)
	
	if not save:
		return log_loss(y_test, test_preds, eps=1e-15)
	else:
#		create_feature_map(X_train.columns[1:])
#		draw_feature_map(gbm)
		submission = pd.DataFrame()
		submission['ID'] = test.z
		submission.loc[:,'PredictedProb'] = test_preds
		print "Saving output...."
		submission.to_csv('submissions/output_'+str(np.mean(scores))+'.csv',index=False)




def add_features():
	print "Adding Features..."
	global train
	global test
	train.x["CountNulls"] = np.sum(train.x[train.x.columns] == -999 , axis = 1)
	test.x["CountNulls"] = np.sum(test.x[test.x.columns] == -999 , axis = 1)
	

def clean_data(train, test):
	print "Cleaning Data..."
	categorical_cols = []
	numeric_cols = []
	for col in train.columns:
		if train[col].dtype == 'object':
			categorical_cols.append(col)
		else:
			numeric_cols.append(col)

	train.fillna(-999,inplace=True)
	test.fillna(-999, inplace = True)
	# label encoding
	enc = preprocessing.LabelEncoder()
	for col in categorical_cols:
		train[col] = enc.fit_transform(train[col])
		test[col] = enc.fit_transform(test[col])

	return train, test

def select_features():
	# remove constant columns
	
	cols_to_drop = []
	cols = train.x.columns
	for col in cols:
		if train.x[col].std() == 0:
			cols_to_drop.append(col)

	cols = list(Set(cols) - Set(cols_to_drop))

	# remove highly correlated columns
	
	print "Removing correlated factors"
	
	# replace code below, modifying the list over which we are iterating
	corr_cols = []
	col_pairs = []
	for i in range(len(cols)):
		for j in range(i+1, len(cols)):
			col_pairs.append((cols[i],cols[j]))

	corr_cols = [col2 for col1,col2 in col_pairs if abs(train.x[col1].corr(train.x[col2]))>0.999]
	cols_to_drop += corr_cols
	
	return cols_to_drop

def plot_scatter():
	'''
		Function to save scatter plots 
	'''
	print "plotting correlation plots.."
	temp = train.x
	temp['target'] = train.y
	for col in train.x.columns:
		plt.figure()
		temp.plot(kind='hexbin',x=col,y='target',gridsize=10)
		plt.savefig("histograms/"+col+'.png')
		plt.close()

if __name__ == "__main__":
	train = Data()
	test = Data()
	print "Reading Train Data..."
	data = pd.read_csv('train.csv')
	train.x = data.iloc[:,2:]
	train.y['PredictedProb'] = data.iloc[:,1]
	print "Reading Test Data..."
	data = pd.read_csv('test.csv')
	test.x = data.iloc[:,1:]
	test.y = None
	test.z['ID'] = data.iloc[:,0]
	train.x, test.x = clean_data(train.x, test.x)
	cols_to_drop = select_features()
	add_features()
	#cols_to_drop = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']
	print cols_to_drop
	train.x.drop(cols_to_drop, inplace=True, axis=1)
	test.x.drop(cols_to_drop, inplace=True, axis =1)
	#plot_scatter()
	#sys.exit()
	skf = KFold(len(train.x),n_folds=3,shuffle=True,random_state=2)
	#print "Begin 3 fold cross validation"
	scores = []
	
	for train_index,test_index in skf:
		X_train = train.x.iloc[train_index,:]
		X_test = train.x.iloc[test_index,:]
		y_train = train.y.iloc[train_index,:]['PredictedProb']
		y_test = train.y.iloc[test_index,:]['PredictedProb']
		score = xgb_model(X_train,y_train,X_test,y_test)
		print score
		scores.append(score)
	print np.mean(scores)

	proceed = raw_input("Train on entire Data? (T/F)")
	if proceed == 'T':		
		xgb_model(train.x,train.y['PredictedProb'],test.x,test.y,True)

	#Save scatterplot images
	#plot_scatter()
	
	print "Done!! Exiting Now..."

