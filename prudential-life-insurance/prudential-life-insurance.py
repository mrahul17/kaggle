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
import xgboost as xg
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
# remove comment from the line below if using ipython notebook
#%matplotlib inline

# credits to @zeroblue
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

def xgb_model():
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
	print "Predicting on train data..."
	train_preds = gbm.predict(xg.DMatrix(X_train),ntree_limit=gbm.best_iteration)

	print "Predicting on test data..."
	test_preds = gbm.predict(xg.DMatrix(X_test),ntree_limit=gbm.best_iteration)
	

	train_preds = np.clip(train_preds, -0.99, 8.99)
	test_preds = np.clip(test_preds, -0.99, 8.99)
	num_classes = 8
	# train offsets 
	offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
	data = np.vstack((train_preds, train_preds, train['Response'].values))
	for j in range(num_classes):	
		data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
	for j in range(num_classes):
		train_offset = lambda x: -apply_offset(data, x, j)
		offsets[j] = fmin_powell(train_offset, offsets[j])  

	# apply offsets to test
	#return test_preds.shape,test['Response'].values
	
	data = np.vstack((test_preds, test_preds, test['Response'].values))
	for j in range(num_classes):
		data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

	final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

	# thanks @inversion https://www.kaggle.com/inversion/prudential-life-insurance-assessment/digitize/code
	#preds = np.clip(y_pred,0.1,8.1)
	#splits = [0, 1.5, 2.5, 3, 4.2, 5.8, 6.5, 7]
	#y_pred = np.digitize(preds, splits)
	print eval_wrapper(train_preds,y_train)
	submission = test[['Id']]
	submission.loc[:,'Response'] = final_test_preds
	print "Saving output...."
	# fix to remove floats
	submission = submission.astype(int)
	submission.to_csv('submissions/output10.csv',index=False)


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

	print 'Filling Missing values'
	all_data.fillna(-1, inplace=True)

	all_data['Response'] = all_data['Response'].astype(int)
	train_new = all_data[all_data['Response']>0].copy()
	test_new = all_data[all_data['Response']<1].copy()

	train_new.to_csv('train_prepared.csv',index=False)
	test_new.to_csv('test_prepared.csv',index=False)
	
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
	#X_train,X_test,y_train,y_test = train_test_split(train.iloc[:,1:].drop((['Response']+columns_to_drop),axis=1),train['Response'],test_size=0.3, random_state=0)
	
	#train,test = select_features()
	X_train =  train.iloc[:,1:].drop(columns_to_drop,axis=1)
	X_test = test.iloc[:,1:].drop(columns_to_drop,axis=1)
	y_train = y_test = train['Response']
	
	xgb_model()

	#prepare_sample()
	#print "Selecting Features..."
	
	#apply xgboost
	#print "Predicting on test data"
	#X_train = train.iloc[:,1:].drop(columns_to_drop,axis=1)
	#y_train = train['Response']
	#X_test = test.iloc[:,1:].drop(columns_to_drop,axis=1)
	#xgb_model()
	
	#Save scatterplot images
	#plot_scatter()
	
	print "Done!! Exiting Now..."

