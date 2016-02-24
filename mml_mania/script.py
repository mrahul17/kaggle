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


def match(x):
	season = x['Season']
	team1 = x['Team1']
	team2 = x['Team2']
	seasonMatrix = temp_df.loc[temp_df['Season'] == int(season)]
	seed_team1 = int(seasonMatrix.loc[seasonMatrix['Team'] == int(team1)]['Seed'].item())
	seed_team2 = int(seasonMatrix.loc[seasonMatrix['Team'] == int(team2)]['Seed'].item())
	
	return (0.5 + 0.03 * (seed_team2 - seed_team1))

if __name__ == "__main__":
	
	teams = pd.read_csv("Teams.csv")
	seasons = pd.read_csv("Seasons.csv")
	tourney_seeds = pd.read_csv('TourneySeeds.csv')
	sample_submission = pd.read_csv('SampleSubmission.csv')
	season_list = (2012,2013,2014,2015)

	# select data for only 4 years
	season_list_seeds = tourney_seeds.loc[tourney_seeds['Season'].isin(season_list)]
	temp_df = pd.DataFrame()
	matches = sample_submission['Id'].apply(lambda x: pd.Series(x.split('_')))
	matches.rename(columns={0:'Season',1:'Team1',2:'Team2'},inplace=True)
	# sample_submission now has 3 columns : Season,Team1,Team2
	temp_df['Season'] = season_list_seeds['Season']
	temp_df['Team'] = season_list_seeds['Team']
	div_of_team = season_list_seeds['Seed'].str.replace('[0-9]','')
	temp_df['Div'] = div_of_team.str.replace('[a-z]','')
	temp_df['Seed'] = season_list_seeds['Seed'].str.replace('[a-zA-Z]','')	
	# temp_df has 4 columns: Season,Team,Div,Seed

	sample_submission['Pred'] = matches.apply(lambda x: match(x),axis=1)
	sample_submission.to_csv('submission/out.csv',index=False)

	print "Done!! Exiting Now..."

