
# Importing the libraries

import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

#_______________________________________________________________________#

# Read csv file

dataset = pd.read_csv('hiring.csv')


#_______________________________________________________________________#

# Replace null values experience and test_score columns

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)


#_______________________________________________________________________#
#Splitting Training and Test Set

X = dataset.iloc[:, :3]


#Converting words to integer values

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda X : convert_to_int(X))

y = dataset.iloc[:, -1]


#_______________________________________________________________________#


# we have a very small dataset.
# So, we will train our model with all availabe data.


regressor = LinearRegression()

#_______________________________________________________________________#

#Fitting model with trainig data
regressor.fit(X, y)

#_______________________________________________________________________#

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

#_______________________________________________________________________#



# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 2200, 5]]))

#_______________________________________________________________________#
