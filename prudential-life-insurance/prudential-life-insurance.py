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

# credits to @zeroblue

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

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)      
    return quadratic_weighted_kappa(yhat, y)

def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

def xgb_model(X_train,y_train,X_test,y_test,save=False):
	'''
		Function to apply the xgb model to the split train dataset to get the score
	'''
	# setup parameters for xgboost
	params = {}
	# use softmax multi-class classification
	#params['objective'] = 'multi:softmax'
	params['objective'] = 'reg:linear'
	# scale weight of positive examples
	params["eta"] = 0.05
	params["min_child_weight"] = 240
	params["subsample"] = 0.9
	params["colsample_bytree"] = 0.67
	params["silent"] = 1
	params["max_depth"] = 6
	#params['num_class'] = 8
	print "Training the model now... This will take really long..."
	gbm = xg.train(params,xg.DMatrix(X_train,y_train),800)

#	draw_feature_map(gbm)

	print "Predicting on train data..."
	train_preds = gbm.predict(xg.DMatrix(X_train),ntree_limit=gbm.best_iteration)

	print "Predicting on test data..."
	test_preds = gbm.predict(xg.DMatrix(X_test),ntree_limit=gbm.best_iteration)
	

	train_preds = np.clip(train_preds, -0.99, 8.99)
	test_preds = np.clip(test_preds, -0.99, 8.99)
	num_classes = 8
	# train offsets 
	offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
	data = np.vstack((train_preds, train_preds, y_train.values))
	for j in range(num_classes):	
		data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
	for j in range(num_classes):
		train_offset = lambda x: -apply_offset(data, x, j)
		offsets[j] = fmin_powell(train_offset, offsets[j])  

	# apply offsets to test
	#return test_preds.shape,test['Response'].values
	
	data = np.vstack((test_preds, test_preds, y_test.values))
	for j in range(num_classes):
		data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

	final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

	# thanks @inversion https://www.kaggle.com/inversion/prudential-life-insurance-assessment/digitize/code
	#preds = np.clip(y_pred,0.1,8.1)
	#splits = [0, 1.5, 2.5, 3, 4.2, 5.8, 6.5, 7]
	#y_pred = np.digitize(preds, splits)
	if not save:
		return eval_wrapper(final_test_preds,y_test)
	else:
		submission = X_test[['Id']]
		submission.loc[:,'Response'] = final_test_preds
		print "Saving output...."
		# fix to remove floats
		submission = submission.astype(int)
		submission.to_csv('submissions/output12.csv',index=False)


def add_features():
	# count number of zeroes
	print "Adding Features..."
	all_data = train.append(test)
	
	# @credits zeroblue
	# Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
	# create any new variables    
	all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
	all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

	# factorize categorical variables
	all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
	all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
	all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

	all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

	med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
	all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)
	
	# inspired by https://www.kaggle.com/mariopasquato/prudential-life-insurance-assessment/linear-model/code
	all_data['BMI_Prod4'] = all_data['BMI'] * all_data['Product_Info_4']
	all_data['BMI_Med_Key3'] = all_data['BMI'] * all_data['Medical_Keyword_3']


	#all_data['Age_Med_Keywords_Count']  = all_data['Med_Keywords_Count'] * all_data['Ins_Age']
	print 'Filling Missing values'
	all_data.fillna(-1, inplace=True)


	all_data['Response'] = all_data['Response'].astype(int)
	cols = [col for col in train.columns if col != "Response" and col != "Id"]
	all_data["CountNulls"]=np.sum(all_data[cols] == -1 , axis = 1)
	
	insured_info_columns = all_data.columns[all_data.columns.str.startswith('InsuredInfo_')]
	all_data['UA_InsuredInfo'] = np.sum(all_data[insured_info_columns] == -1 , axis = 1)
	medical_history_columns = all_data.columns[all_data.columns.str.startswith('Medical_History_')]
	all_data['UA_Medical_History'] = np.sum(all_data[medical_history_columns] == -1 , axis = 1)
	family_hist_columns = all_data.columns[all_data.columns.str.startswith('Family_Hist_')]
	all_data['UA_Family_Hist'] = np.sum(all_data[family_hist_columns] == -1 , axis = 1)
	employment_info_columns = all_data.columns[all_data.columns.str.startswith('Employment_Info_')]
	all_data['UA_Employment_Info'] = np.sum(all_data[employment_info_columns] == -1 , axis = 1)
	for col in cols:
		all_data[col+"square"] = np.square(all_data[col])
	train_new = all_data[all_data['Response']>0].copy()
	test_new = all_data[all_data['Response']<1].copy()

	#train_new.to_csv('train_prepared.csv',index=False)
	#test_new.to_csv('test_prepared.csv',index=False)
	
	return train_new,test_new


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
	# COMMENT THE  LINES BELOW AFTER THE PROGRAM HAS BEEN RUN ONCE
	#prepare_data()

	# THE LINES BELOW WILL BE COMMENTED WHEN THE ABOVE IS USED
	print "Reading Train Data..."
	train = pd.read_csv('train.csv')
	print "Reading Test Data..."
	test = pd.read_csv('test.csv')
	train,test = add_features()

	# COMMENT THE LINE BELOW
	columns_to_drop = ['Response','Medical_History_10','Medical_History_24']
	features = list(Set(train.columns)-Set(columns_to_drop))
	ceate_feature_map(features)
	#train,test = select_features()
	skf = KFold(len(train),n_folds=3)
	print "Begin 3 fold cross validation"
	for train_index,test_index in skf:
		train_part = train.iloc[train_index,:]
		test_part = train.iloc[test_index,:]
		X_train =  train_part.iloc[:,1:].drop(columns_to_drop,axis=1)
		X_test = test_part.iloc[:,1:].drop(columns_to_drop,axis=1)
		y_train = train_part['Response']
		y_test = test_part['Response']
		print xgb_model(X_train,y_train,X_test,y_test)

	proceed = raw_input("Train on entire Data? (T/F)")
	if proceed == 'T':
		X_train =  train.iloc[:,1:].drop(columns_to_drop,axis=1)
		X_test = test.iloc[:,1:].drop(columns_to_drop,axis=1)
		y_train = train['Response'] 
		y_test = test['Response']
		
		xgb_model(X_train,y_train,X_test,y_test,True)

	#prepare_sample()
	#print "Selecting Features..."
	
	#apply xgboost
	#print "Predicting on test data"
	
	
	
	#Save scatterplot images
	#plot_scatter()
	
	print "Done!! Exiting Now..."

