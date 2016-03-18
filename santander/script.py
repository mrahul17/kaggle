import pandas as pd
import xgboost as xg


def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)      
    return quadratic_weighted_kappa(yhat, y)

def xgb_model(X_train,y_train,X_test,y_test=None,save=False):
	'''
		Function to apply the xgb model to the split train dataset to get the score
	'''
	#if not save:
	#	num_rounds = 800
	#else:
	#	num_rounds = 1000
	num_rounds = 600
	# setup parameters for xgboost
	params = {}
	params['objective'] = 'binary:logistic'

	params["eta"] = 0.05
	params["min_child_weight"] = 240
	params["subsample"] = 0.9
	params["colsample_bytree"] = 0.67
	params["silent"] = 1
	params["max_depth"] = 6
	print "Training the model now... This will take really long..."
	gbm = xg.train(params,xg.DMatrix(X_train,y_train),num_rounds)

#	draw_feature_map(gbm)

	print "Predicting on train data..."
	train_preds = gbm.predict(xg.DMatrix(X_train),ntree_limit=gbm.best_iteration)

	print "Predicting on test data..."
	test_preds = gbm.predict(xg.DMatrix(X_test.iloc[:,1:]),ntree_limit=gbm.best_iteration)
	


	#if not save and y_test:
	#	return eval_wrapper(final_test_preds,y_test)
	#else:
	submission = X_test[['ID']]
	submission.loc[:,'TARGET'] = test_preds
	print "Saving output...."
	submission.to_csv('submissions/output.csv',index=False)


def select_features(train,test):
	# remove constant columns
	cols_to_drop = []
	cols = train.columns
	for col in cols:
		if train[col].std() == 0:
			cols_to_drop.append(col)
	return cols_to_drop


if __name__ == "__main__":
	print "Reading Train Data..."
	train = pd.read_csv('train.csv')
	print "Reading Test Data..."
	test = pd.read_csv('test.csv')
	#train,test = add_features()

	# COMMENT THE LINE BELOW
	columns_to_drop = []
	
	#ceate_feature_map(features)
	cols_to_drop = select_features(train, test)
	#train,test = select_features()
	#skf = KFold(len(train),n_folds=3,shuffle=True,random_state=2)
	#print "Begin 3 fold cross validation"
	#scores = []
	train.drop(cols_to_drop+['ID'], inplace=True, axis=1)
	test.drop(cols_to_drop, inplace = True, axis =1)
	'''for train_index,test_index in skf:
		train_part = train.iloc[train_index,:]
		test_part = train.iloc[test_index,:]
		X_train =  train_part.iloc[:,1:].drop(columns_to_drop,axis=1)
		X_test = test_part.iloc[:,1:].drop(columns_to_drop,axis=1)
		y_train = train_part['Response']
		y_test = test_part['Response']
		score = xgb_model(X_train,y_train,X_test,y_test)
		print score
		scores.append(score)
	print np.mean(scores)'''
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