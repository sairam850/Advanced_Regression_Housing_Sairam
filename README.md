# Advanced_Regression_Housing_Sairam
import numpy as np
import pandas as pd
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os

import warnings
warnings.filterwarnings('ignore')

training_data = pd.read_csv("C://Users//Sairam//Downloads//train.csv",encoding = 'utf-8')
training_data.head()

training_data.shape

training_data.info()

training_data.describe()

training_data.isnull().sum()

# Outlier Check

training_data.describe(percentiles = [0.25,0.5,0.75,0.90,0.95,0.99])

# Method to remove outliers.

def remove_outliers(x,y):
    q1 = x[y].quantile(0.25)
    q3 = x[y].quantile(0.75)
    value = q3-q1
    low = q1-(1.5*value)
    high = q3+(1.5*value)
    out= x[(x[y]<high) & (x[y]>low)]
    return out

training_data.shape

training_data.columns[training_data.isnull().any()]

null = training_data.isnull().sum()/len(training_data)*100
null = null[null>0]
null.sort_values(inplace=True, ascending=False)
null

# Categorical Value Imputation

null_with_meaning = ["Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]
for i in null_with_meaning:
    training_data[i].fillna("none", inplace=True)



training_data.columns[training_data.isnull().any()]

null2 = training_data.isnull().sum()/len(training_data)*100
null2 = null2[null2>0]
null2.sort_values(inplace=True, ascending=False)
null2

training_data['LotFrontage'].describe()

training_data['GarageYrBlt'].describe()

training_data['MasVnrArea'].describe()

training_data['Electrical'].describe()

training_data['LotFrontage'] = training_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
training_data["GarageYrBlt"].fillna(training_data["GarageYrBlt"].median(), inplace=True)
training_data["MasVnrArea"].fillna(training_data["MasVnrArea"].median(), inplace=True)
training_data["Electrical"].dropna(inplace=True)

training_data['LotFrontage'].describe()

training_data['GarageYrBlt'].describe()

training_data['MasVnrArea'].describe()

training_data['Electrical'].describe()

# EDA on Cleaned Data

training_data_numeric = training_data.select_dtypes(include=['float64', 'int64'])
training_data_numeric.head()

training_data_numeric = training_data_numeric.drop(['Id'],axis =1)
training_data_numeric.head()

# Price vs Selected Columns

plt.figure(figsize = (20,12))
sns.barplot(x = "Neighborhood", y = "SalePrice" , data = training_data)
plt.title("Sales Price vs Neighbourhood")
plt.xticks(rotation=90)

plt.figure(figsize = (20,12))
sns.barplot(x = "OverallCond", y = "SalePrice" , data = training_data)
plt.title("Sales Price vs OverallCondition")
plt.xticks(rotation=90)

plt.figure(figsize = (20,12))
sns.barplot(x = "OverallQual", y = "SalePrice" , data = training_data)
plt.title("Sales Price vs OverallQuality")
plt.xticks(rotation=90)

sns.distplot(training_data['SalePrice'])

training_data_raw = training_data.copy

# Logarthamic Transformation Applied to Centralize Data

training_data['SalePrice'] = np.log1p(training_data['SalePrice'])

sns.distplot(training_data['SalePrice'])

# Correlation Determination

correlation = training_data_numeric.corr()
correlation

plt.figure(figsize = (30,20))
sns.heatmap(correlation,cmap="YlGnBu", annot=True)
plt.show()

sns.set()
cols = ['SalePrice', 'GrLivArea', 'GarageCars', 'BsmtUnfSF', 'BsmtFinSF1', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'TotRmsAbvGrd', 'GarageYrBlt']
sns.pairplot(training_data[cols], size = 2.5)
plt.show()

# Dropping Columns that are irrelevant to Sales

training_data = training_data.drop(['GarageCars'], axis = 1)
training_data = training_data.drop(['BsmtUnfSF'], axis = 1)
training_data = training_data.drop(['TotRmsAbvGrd'], axis = 1)
training_data = training_data.drop(['GarageYrBlt'], axis = 1)    

training_data.head()

training_data.select_dtypes(exclude = ['object'])

sns.jointplot(x='GrLivArea', y='SalePrice', data=training_data)
plt.show()

training_data = remove_outliers(training_data,'GrLivArea')

training_data.shape

sns.jointplot(x='GrLivArea', y='SalePrice', data=training_data)
plt.show()

sns.jointplot(x='LotFrontage', y='SalePrice', data=training_data)
plt.show()

sns.jointplot(x='1stFlrSF', y='SalePrice', data=training_data)
plt.show()

training_data['PropAge'] = (training_data['YrSold'] - training_data['YearBuilt'])
training_data.head()

sns.jointplot(x='PropAge', y='SalePrice', data=training_data)
plt.show()

# Dropping Irrelevant Columns

training_data = training_data.drop(['MoSold'], axis = 1)
training_data = training_data.drop(['YrSold'], axis = 1)
training_data = training_data.drop(['YearBuilt'], axis = 1)
training_data = training_data.drop(['YearRemodAdd'], axis = 1)
training_data.head()

training_data.Street.value_counts()

training_data.Utilities.value_counts()

training_data = training_data.drop(['Street'], axis = 1)
training_data = training_data.drop(['Utilities'], axis = 1)

# Just to check the variance of these columns

l2= training_data.select_dtypes(include=['float64', 'int64'])
l2

for i in l2:
    print(training_data[i].value_counts())

# 3. Data Preparation

training_data = training_data.drop(['PoolQC','MiscVal', 'Alley', 'RoofMatl', 'Condition2', 'Heating', 'GarageCond', 'Fence', 'Functional' ], axis = 1)

training_data.shape

training_data = training_data.drop(['Id'],axis = 1)
training_data.head()

types = training_data.dtypes
numeric_type = types[(types == 'int64') | (types == float)] 
categorical_type = types[types == object]

pd.DataFrame(types).reset_index().set_index(0).reset_index()[0].value_counts()

numerical_columns = list(numeric_type.index)
print(numerical_columns)

categorical_columns = list(categorical_type.index)
print(categorical_columns)

# Creating Dummy columns to convert categorical into numerical

training_data = pd.get_dummies(training_data, drop_first=True )
training_data.head()

X = training_data.drop(['SalePrice'],axis = 1)
X.head()

y = training_data['SalePrice']
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=50)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PropAge']] = scaler.fit_transform(X_train[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PropAge']])

X_test[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PropAge']] = scaler.fit_transform(X_test[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PropAge']])

X_train.head()

X_test.head()

# 4. Model Evaluation using Recursive Feature Elimnation

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


import statsmodels.api as sm

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 100)             # running RFE
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col = X_train.columns[rfe.support_]
col

X_train.columns[~rfe.support_]

col = X_train.columns[rfe.support_]
col

X_train.columns[~rfe.support_]


X_train_rfe = X_train[col]

X_train_rfe = pd.DataFrame(X_train[col])

X_train_rfe.head()

X_train_rfe.shape

y_train_pred = lm.predict(X_train)
metrics.r2_score(y_true=y_train, y_pred=y_train_pred)

y_test_pred = lm.predict(X_test)
metrics.r2_score(y_true=y_test, y_pred=y_test_pred)

list(zip(X_test.columns,rfe.support_,rfe.ranking_))

col1 = X_test.columns[rfe.support_]
col1

X_test_rfe = X_test[col1]

X_test_rfe.head()

# LASSO AND RIDGE REGRESSION

# LASSO REGRESSION

print("X_train", X_train.shape)
print("y_train", y_train.shape)


params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
lasso = Lasso()

folds = 5
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=1]
cv_results.head()

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()

alpha = 0.01
lasso = Lasso(alpha=alpha)

lasso.fit(X_train, y_train)
lasso.coef_

model_parameters = list(lasso.coef_ )
model_parameters.insert(0, lasso.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))

lm = Lasso(alpha=0.01)
lm.fit(X_train, y_train)

# prediction on the test set(Using R2)
y_train_pred = lm.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

mod = list(zip(cols, model_parameters))

para = pd.DataFrame(mod)
para.columns = ['Variable', 'Coeff']
para.head()

para = para.sort_values((['Coeff']), axis = 0, ascending = False)
para

pred = pd.DataFrame(para[(para['Coeff'] != 0)])
pred

pred.shape

Lassso_var = list(pred['Variable'])
print(Lassso_var)

X_train_lasso = X_train[['GrLivArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'Fireplaces', 'LotArea', 'LotFrontage', 'BsmtFullBath', 'Foundation_PConc', 'OpenPorchSF', 'FullBath', 'ScreenPorch', 'WoodDeckSF']]
                        
X_train_lasso.head()


X_train_lasso.shape

X_test_lasso = X_test[['GrLivArea', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'Fireplaces', 'LotArea', 'LotFrontage', 'BsmtFullBath', 'Foundation_PConc', 'OpenPorchSF', 'FullBath', 'ScreenPorch', 'WoodDeckSF']]
                        
X_test_lasso.head()

# RIDGE REGRESSION

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=5]
cv_results.head()

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')


plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


alpha = 2
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_

model_parameters = list(ridge.coef_)
model_parameters.insert(0, ridge.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))

lm = Ridge(alpha=2)
lm.fit(X_train, y_train)


y_train_pred = lm.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

mod_ridge = list(zip(cols, model_parameters))

paraRFE = pd.DataFrame(mod_ridge)
paraRFE.columns = ['Variable', 'Coeff']
res=paraRFE.sort_values(by=['Coeff'], ascending = False)
res.head(20)

paraRFE = paraRFE.sort_values((['Coeff']), axis = 0, ascending = False)
paraRFE

predRFE = pd.DataFrame(paraRFE[(paraRFE['Coeff'] != 0)])
predRFE

predRFE.shape

# Observation:
Though the model performance by Ridge Regression was better in terms of R2 values of Train and Test,
it is better to use Lasso, since it brings and assigns a zero value to insignificant features, enabling us to choose
the predictive variables.
It is always advisable to use simple yet robust model.
Equation can be formulated using the features and coefficients obtained by Lasso

pred.set_index(pd.Index(['C','x1', 'x2', 'x3', 'x4', 'x5' , 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']), inplace = True) 
pred

