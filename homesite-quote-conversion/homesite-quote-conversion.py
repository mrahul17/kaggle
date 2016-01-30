import pandas as pd
import numpy as np
from sets import Set
#import matplotlib.pyplot as plt
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
#from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
# remove comment from the line below if using ipython notebook
#%matplotlib inline

def read_data():

	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	features = train.columns.tolist()
	return (train,test,features)


#def create_mapping():
	#arr = pd.unique(train[label])
	#map = {}
	#for i in range(len(arr)):
    #        if not pd.isnull(arr[i]):
	#        map[arr[i]] = i
	#return map
def clean_data():
	obj_cols = [] #list to store columns that have been read as objects
	for col in train.columns:
		if train[col].dtype==object:
			obj_cols.append(col)
	preprocessor = preprocessing.LabelEncoder()	
	obj_cols = list(Set(obj_cols)-Set(['Field10','Original_Quote_Date']))
	
	for col in obj_cols:
		preprocessor.fit(list(train[col].values) + list(test[col].values))
		train[col] = preprocessor.transform(list(train[col].values))
		test[col] = preprocessor.transform(list(test[col].values))	



#def replace(label):
#	train.replace({label:create_mapping(label)},inplace=True)


if __name__ == "__main__":
	print "Reading Data..."
	train,test,features = read_data()
	
	# list to store columns that have null values
	print "Cleaning Data..."
	clean_data()
	print "Filling na values.."
	train.fillna(-1,inplace=True)
	test.fillna(-1,inplace=True)

	# Deal with Field10
	print "Converting Field10.."
	map_field10 = {}

	train_f10_unique = pd.unique(train['Field10'])
	test_f10_unique = pd.unique(test['Field10'])


	for item in train_f10_unique:
		map_field10[item] = int("".join(item.split(",")))
	for item in test_f10_unique:
		map_field10[item] = int("".join(item.split(",")))

	train.replace({'Field10':map_field10},inplace=True)
	test.replace({'Field10':map_field10},inplace=True)

	#Deal with Original_Quote_Date
	print "Converting Original_Quote_Date..."
	train_ymd = train['Original_Quote_Date'].apply(lambda x: pd.Series(x.split('-')))
	test_ymd = test['Original_Quote_Date'].apply(lambda x: pd.Series(x.split('-')))
	train['Year'],train['Month'],train['Day'] = train_ymd[0],train_ymd[1],train_ymd[2]
	test['Year'],test['Month'],test['Day'] = test_ymd[0],test_ymd[1],test_ymd[2]
	train.drop('Original_Quote_Date',axis=1,inplace=True)
	test.drop('Original_Quote_Date',axis=1,inplace=True)

	print "Saving Cleaned Data on disk.."
	train.to_csv('train_cleaned.csv')
	test.to_csv('test_cleaned.csv')
	print "Clean Data ready and saved on disk. Exiting..."



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