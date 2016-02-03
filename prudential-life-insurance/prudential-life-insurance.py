import pandas as pd
import numpy as np
from sets import Set
#import matplotlib.pyplot as plt
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import xgboost as xg
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
# remove comment from the line below if using ipython notebook
#%matplotlib inline

def read_data():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	features = train.columns.tolist()
	return (train,test,features)


def clean_data(train,test):
	obj_cols = [] #list to store columns that have been read as objects
	for col in train.columns:
		if train[col].dtype==object:
			obj_cols.append(col)
	preprocessor = preprocessing.LabelEncoder()	
	
	for col in obj_cols:
		preprocessor.fit(list(train[col].values) + list(test[col].values))
		train[col] = preprocessor.transform(list(train[col].values))
		test[col] = preprocessor.transform(list(test[col].values))	

	return train,test

def select_features():
	# select 250 best features
	selectf = SelectKBest(f_classif	, k=250)
	selectf.fit(train.iloc[:,2:], train.iloc[:,1])
	
	#train_new = (train.iloc[:,2:])[:,selectf.get_support()]
	#test_new = (test.iloc[:,1:])[:,selectf.get_support()]
	train_column_list = list(train.columns)
	univ_col_list = train_column_list[2:]
	cols = []
	i = 0
	mask = selectf.get_support()
	for col in mask:
		if col:
			cols.append(univ_col_list[i])
		i += 1

	train_new = train[train_column_list[:2]+cols]
	test_new = test[train_column_list[:1]+cols]
	
	return (train_new,test_new)
	

	

def xgb_model():

	# setup parameters for xgboost
	params = {}
	# use softmax multi-class classification
	params['objective'] = 'multi:softmax'
	# scale weight of positive examples
	params["eta"] = 0.05
	params["min_child_weight"] = 240
	params["subsample"] = 0.9
	params["colsample_bytree"] = 0.67
	params["silent"] = 1
	params["max_depth"] = 6
	params['num_class'] = 8
	print "Training the model now... This will take really long..."

	gbm = xg.train(params,xg.DMatrix(train.iloc[:,1:].drop('Response',axis=1),train.iloc[:,127]),800)
	print "Predicting..... "
	y_pred = gbm.predict(xg.DMatrix(test.iloc[:,1:]))
	submission = test[['Id']]
	submission.loc[:,'Response'] = y_pred + 1
	print "Saving output...."
	# fix to remove floats
	submission = submission.astype(int)
	submission.to_csv('submissions/output3.csv',index=False)

def add_features():
	# count number of zeroes
	print "Adding Features..."
	#cols = [col for col in train.columns if col != "Response"]
	#train["CountNulls"]=np.sum(train[cols] == -1 , axis = 1)
	#test["CountNulls"]=np.sum(test[cols] == -1 , axis = 1) 
	med_keyword_columns = [col for col in train.columns if col.startswith('Medical_Keyword_')]
	train['Med_Keywords_Count'] = train[med_keyword_columns].sum(axis=1)
	train['BMI_Age'] = train['BMI'] * train['Ins_Age']
	test['Med_Keywords_Count'] = test[med_keyword_columns].sum(axis=1)
	test['BMI_Age'] = test['BMI'] * test['Ins_Age']

	#test[col] = np.square(test[col])

	#train.to_csv('train_with_edited_featues.csv',index=False)
	#test.to_csv('test_with_edited_featues.csv',index=False)



def prepare_data():
	
	print "Reading Original Train and Test Data..."
	train,test,features = read_data()
	
	print "Cleaning Data..."
	train,test = clean_data(train,test)
	# because labels need to start from 0
	train.loc[:,'Response'] = train.loc[:,'Response'] - 1 
	print "Filling na values.."
	train.fillna(-1,inplace=True)
	test.fillna(-1,inplace=True)


	print "Saving Cleaned Data on disk.."
	train.to_csv('train_cleaned.csv',index=False)
	test.to_csv('test_cleaned.csv',index=False)	
	print "Clean Data ready and saved on disk. Exiting..."



if __name__ == "__main__":
	# COMMENT THE  LINES BELOW AFTER THE PROGRAM HAS BEEN RUN ONCE
	#prepare_data()

	# THE LINES BELOW WILL BE COMMENTED WHEN THE ABOVE IS USED
	print "Reading Train Data..."
	train = pd.read_csv('train_cleaned.csv')
	#X_train,X_test,y_train,y_test = train_test_split(train.iloc[:,1:127],train.iloc[:,127],test_size=0.4, random_state=0)	

	print "Reading Test Data..."
	test = pd.read_csv('test_cleaned.csv')
	#prepare_sample()
	#print "Selecting Features..."
	#train,test = select_features()
	#apply xgboost
	add_features()
	xgb_model()
	print "Done!! Exiting Now..."


	#X = train.drop('QuoteConversion_Flag')
	#y = train.QuoteConversion_Flag

	#classifier = GradientBoostingClassifier(random_state=0)

	# Field10 and Original_Quote_Date need special mention



#		mappings_universal[col] = create_mapping(col)
#		train.replace({col:mappings_universal[col]},inplace=True)

#	test_obj_cols = obj_cols

#	for col in test_obj_cols:
#		test.replace({col:mappings_universal[col]},inplace=True)

	# after this the values which were not in 

	

	#delete useless

#	train.replace({'PropertyField4':create_mapping('PropertyField4')},inplace=True)
