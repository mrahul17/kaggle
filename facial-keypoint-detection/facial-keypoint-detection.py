import pandas as pd
import numpy as np

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)



def read_data():
	train = pd.read_csv('training.csv')
	test = pd.read_csv('test.csv')
	submission = pd.read_csv('SampleSubmission.csv')
	idlookup = pd.read_csv('IdLookupTable.csv')
	features = train.columns.tolist()
	return (train,test,submission,features,idlookup)




if __name__ == "__main__":
	train,test,submission,features,idlookup = read_data()
	
	for col in train:
		print col,train[col].dtypes
    
	bunch = Bunch()
	bunch.image = train.Image

	train = train.drop('Image',axis=1)
	# image data has been read as string. so convert to integer
	for i in range(len(bunch.image)):
		bunch.image[i] = [int(pix) for pix in bunch.image[i].split(" ")]

	#http://stackoverflow.com/questions/12760797/imshowimg-cmap-cm-gray-shows-a-white-for-128-value
	
	for i in range(len(test)):
		test.iloc[i].Image = [int(num) for num in test.iloc[i].Image.split(" ")]

	means = train.mean(axis=0)
	means_hor = np.reshape(means,(1,30))
	x = pd.DataFrame(means_hor,columns = train.columns)
	predictions = pd.concat([x]*len(test),ignore_index=True)


	for i in range(len(idlookup)):
		feature = idlookup.loc[i,'FeatureName']
		imgId = idlookup.loc[i,'ImageId']
		submission.loc[i,'Location'] = predictions.loc[imgId-1,feature]