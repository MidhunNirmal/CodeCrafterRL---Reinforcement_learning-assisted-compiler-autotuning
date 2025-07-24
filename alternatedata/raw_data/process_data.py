import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('../data/data.csv')
relevent_data = data.drop('code_size', axis=1)
test_app = 'consumer_tiffmedian'

def train_test_split(test_app):
	train_data  = relevent_data[relevent_data['APP_NAME'] != test_app]
	test_data = relevent_data[relevent_data['APP_NAME'] == test_app]
	scaled_train_predictors = MinMaxScaler(feature_range=(0,1)).fit_transform(relevent_data[relevent_data['APP_NAME'] != test_app].iloc[:,1:-5])
	train_targets = relevent_data[relevent_data['APP_NAME'] != test_app].iloc[:,-5:]
	scaled_train_targets = []
	for app in range(0, train_targets.shape[0], 128):
		scaled_train_targets.append(MinMaxScaler(feature_range=(0,1)).fit_transform(train_targets.iloc[app:app+128]))
	scaled_train_targets = np.array(scaled_train_targets).reshape(train_targets.shape[0],5)
	scaled_test_predictors = MinMaxScaler(feature_range=(0,1)).fit_transform(relevent_data[relevent_data['APP_NAME'] == test_app].iloc[:,1:-5])
	scaled_test_targets = MinMaxScaler(feature_range=(0,1)).fit_transform(relevent_data[relevent_data['APP_NAME'] == test_app].iloc[:,-5:])
	train_set = np.concatenate(
	      (
	          scaled_train_predictors,
	          scaled_train_targets
	      ),
	      axis=1
	  )
	test_set = np.concatenate(
	    (
	        scaled_test_predictors,
	        scaled_test_targets
	    ),
	    axis=1
	)
	return train_set, test_set
train_set, test_set = train_test_split(test_app)
pd.DataFrame(train_set).to_csv(path_or_buf='../data/train.csv', index=False)
pd.DataFrame(test_set).to_csv(path_or_buf='../data/test.csv', index=False)

train, test = pd.read_csv('../data/train.csv'), pd.read_csv('../data/test.csv')
print(train.shape, test.shape)
print(train.head())
print(test.head())

