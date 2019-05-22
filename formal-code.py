import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder
from sklearn.linear_model import RidgeCV,LinearRegression,ElasticNetCV,LassoCV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
data=pd.read_csv("DATA.csv")
data=data.dropna()# remove the observations containing missing sort_values
# Part  1
print(data.shape)
# observe the distribution of the target variables "SalePrice"
sns.distplot(data["SalePrice"])
plt.show()
#the nutural logrithmn of "SalePrice"
sns.distplot(np.log1p(data["SalePrice"]))
plt.show()
# Relationship between continuous variables snd SalePrice
data.plot.scatter(x="GrLivArea",y="SalePrice",ylim=(0,800000))
plt.show()
data.plot.scatter(x="TotalBsmtSF",y="SalePrice",ylim=(0,800000))
plt.show()
#Relationship between discrete variables and SalePrice
sns.boxplot(x="OverallQual",y="SalePrice",data=data)
plt.show()
sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
plt.show()
# the heatmap of the correlation Coefficients
corrmat=data.corr()
sns.heatmap(corrmat,vmax=0.8,square=True)
plt.show()

# Part 2
data['SalePrice']=np.log(data['SalePrice'])
#tranform into exponential, it seem more likely to normal distribution
y=data['SalePrice']
x=data.drop(['SalePrice'], axis=1)
scaler = StandardScaler()
mse_elastic=[]
mse_lr=[]
mse_lasso=[]
mse_ridge=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=i)
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    Y_test=np.exp(Y_test)

    lasso=LassoCV(cv=5)
    lasso.fit(X_train,Y_train)
    ridge=RidgeCV(cv=5)
    ridge.fit(X_train,Y_train)
    lr=LinearRegression().fit(X_train,Y_train)
    ElasticNet= ElasticNetCV(cv=5, random_state=0)
    ElasticNet.fit(X_train,Y_train)

    Y_pred_lasso=lasso.predict(X_test)
    Y_pred_ridge=ridge.predict(X_test)
    Y_pred_lr=lr.predict(X_test)
    Y_pred_Elastic=ElasticNet.predict(X_test)

    Y_pred_lasso=np.exp(Y_pred_lasso)
    mse_lasso.append(mean_squared_error(y_true=Y_test,y_pred=Y_pred_lasso))

    Y_pred_ridge=np.exp(Y_pred_ridge)
    mse_ridge.append(mean_squared_error(y_true=Y_test,y_pred=Y_pred_ridge))

    Y_pred_lr=np.exp(Y_pred_lr)
    mse_lr.append(mean_squared_error(y_true=Y_test,y_pred=Y_pred_lr))

    Y_pred_Elastic=np.exp(Y_pred_Elastic)
    mse_elastic.append(mean_squared_error(y_true=Y_test,y_pred=Y_pred_Elastic))

s1=pd.Series(np.array(mse_lasso))
s2=pd.Series(np.array(mse_ridge))
s3=pd.Series(np.array(mse_lr))
s4=pd.Series(np.array(mse_elastic))
data_mse=pd.DataFrame({"Linear Regression":s3,"Ridge":s2,"Lasso":s1,"Elastic Net":s4})
data_mse.boxplot()
plt.xlabel("Regression Methods")
plt.ylabel("MSE")
plt.show()
