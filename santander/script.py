import pandas as pd
import numpy as np
import xgboost as xg
from sets import Set
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold 
import matplotlib.pyplot as plt
import operator

def create_feature_map(features):
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
		num_rounds = 400
	else:
		num_rounds = 500
	
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
	gbm = xg.train(params,xg.DMatrix(X_train.iloc[:,1:],y_train),num_rounds)

	print "Predicting on test data..."
	test_preds = gbm.predict(xg.DMatrix(X_test.iloc[:,1:]),ntree_limit=gbm.best_iteration)
	
	if not save:
		return roc_auc_score(y_test, test_preds, average="macro")
	else:
#		create_feature_map(X_train.columns[1:])
#		draw_feature_map(gbm)
		submission = pd.DataFrame()
		submission['ID'] = X_test['ID']
		submission.loc[:,'TARGET'] = test_preds
		print "Saving output...."
		submission.to_csv('submissions/output_'+str(np.mean(scores))+'.csv',index=False)


def select_features(train,test):
	# remove constant columns
	
	cols_to_drop = []
	cols = train.columns
	for col in cols:
		if train[col].std() == 0:
			cols_to_drop.append(col)

	cols = list(Set(cols) - Set(cols_to_drop + ['ID', 'TARGET']))

	# remove duplicate columns
	print 'Removing Duplicates..'
	for col1 in cols:
		for col2 in cols:
			if col1 != col2 and train[col1].equals(train[col2]) :
				cols_to_drop.append(col2)
				cols.remove(col2) 
	return cols_to_drop

	# remove highly correlated columns
	
	print "Removing correlated factors"
	
	for col1 in cols:
		for col2 in cols:
			if col1 != col2:
				cor = train[col1].corr(train[col2])
				if abs(cor) > 0.999 :
					cols_to_drop.append(col2)
					cols.remove(col2) 

def add_features(train, test):
	for col in train.columns:
		if col not in ['ID', 'TARGET']:
			train[col + '2'] = np.square(train[col])
			test[col+'2'] = np.square(test[col])

	return train, test

if __name__ == "__main__":
	print "Reading Train Data..."
	#train = pd.read_csv('train.csv')
	train = pd.read_csv('train_edited1.csv')
	print "Reading Test Data..."
	#test = pd.read_csv('test.csv')
	test = pd.read_csv('test_edited1.csv')
	#train,test = add_features()

	
	#ceate_feature_map(features)
	#cols_to_drop = select_features(train, test)
	#train.drop(cols_to_drop, inplace=True, axis=1)
	#test.drop(cols_to_drop, inplace = True, axis =1)	
	#train,test = add_features(train, test)
	
	print "Number of features" + str(len(train.columns))
	skf = KFold(len(train),n_folds=3,shuffle=True,random_state=2)
	#print "Begin 3 fold cross validation"
	scores = []
	
	for train_index,test_index in skf:
		train_part = train.iloc[train_index,:]
		test_part = train.iloc[test_index,:]
		X_train =  train_part.drop(['TARGET'],axis=1)
		X_test = test_part.drop(['TARGET'],axis=1)
		y_train = train_part['TARGET']
		y_test = test_part['TARGET']
		score = xgb_model(X_train,y_train,X_test,y_test)
		print score
		scores.append(score)
	print np.mean(scores)
	del train_part
	del test_part
	proceed = raw_input("Train on entire Data? (T/F)")
	if proceed == 'T':
		X_train =  train.drop(['TARGET'],axis=1)
		X_test = test
		y_train = train['TARGET'] 
		y_test = None
		
		xgb_model(X_train,y_train,X_test,y_test,True)

	#Save scatterplot images
	#plot_scatter()
	
	print "Done!! Exiting Now..."