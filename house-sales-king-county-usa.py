# %% [markdown]
# 
# <h1 align="center"><font size="5">Final Project: House Sales in King County, USA </font></h1>
# 

# %% [markdown]
# <h2>Table of Contents</h2>
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ul>
#     <li><a href="#Instructions">Instructions</a></li>
#     <li><a href="#About-the-Dataset">About the Dataset</a></li>
#     <li><a href="#Module-1:-Importing-Data-Sets">Module 1: Importing Data </a></li>
#     <li><a href="#Module-2:-Data-Wrangling">Module 2: Data Wrangling</a> </li>
#     <li><a href="#Module-3:-Exploratory-Data-Analysis">Module 3: Exploratory Data Analysis</a></li>
#     <li><a href="#Module-4:-Model-Development">Module 4: Model Development</a></li>
#     <li><a href="#Module-5:-Model-Evaluation-and-Refinement">Module 5: Model Evaluation and Refinement</a></li>
# </a></li>
# </div>
# <p>Estimated Time Needed: <strong>75 min</strong></p>
# </div>
# 
# <hr>
# 

# %% [markdown]
# # Instructions
# 

# %% [markdown]
# In this assignment, you are a Data Analyst working at a Real Estate Investment Trust. The Trust would like to start investing in Residential real estate. You are tasked with determining the market price of a house given a set of features. You will analyze and predict housing prices using attributes or features such as square footage, number of bedrooms, number of floors, and so on. This is a template notebook; your job is to complete the ten questions. Some hints to the questions are given.
# 
# As you are completing this notebook, take and save the **screenshots** of the final outputs of your solutions (e.g., final charts, tables, calculation results etc.). They will need to be shared in the following Peer Review section of the Final Project module.
# 

# %% [markdown]
# # About the Dataset
# 
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015. It was taken from [here](https://www.kaggle.com/harlfoxem/housesalesprediction?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-wwwcourseraorg-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01). It was also slightly modified for the purposes of this course. 
# 

# %% [markdown]
# | Variable      | Description                                                                                                 |
# | ------------- | ----------------------------------------------------------------------------------------------------------- |
# | id            | A notation for a house                                                                                      |
# | date          | Date house was sold                                                                                         |
# | price         | Price is prediction target                                                                                  |
# | bedrooms      | Number of bedrooms                                                                                          |
# | bathrooms     | Number of bathrooms                                                                                         |
# | sqft_living   | Square footage of the home                                                                                  |
# | sqft_lot      | Square footage of the lot                                                                                   |
# | floors        | Total floors (levels) in house                                                                              |
# | waterfront    | House which has a view to a waterfront                                                                      |
# | view          | Has been viewed                                                                                             |
# | condition     | How good the condition is overall                                                                           |
# | grade         | overall grade given to the housing unit, based on King County grading system                                |
# | sqft_above    | Square footage of house apart from basement                                                                 |
# | sqft_basement | Square footage of the basement                                                                              |
# | yr_built      | Built Year                                                                                                  |
# | yr_renovated  | Year when house was renovated                                                                               |
# | zipcode       | Zip code                                                                                                    |
# | lat           | Latitude coordinate                                                                                         |
# | long          | Longitude coordinate                                                                                        |
# | sqft_living15 | Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area |
# | sqft_lot15    | LotSize area in 2015(implies-- some renovations)                                                            |
# 

# %%
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# %%
#!pip install -U scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline

# %% [markdown]
# # Module 1: Importing Data Sets
# 

# %% [markdown]
# Download the dataset by running the cell below.
# 

# %%
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

# %%
filepath='https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?select=kc_house_data.csv'

# %%
await download(filepath, "housing.csv")
file_name="housing.csv"

# %% [markdown]
# Load the csv:
# 

# %%
df = pd.read_csv(file_name)

# %%
#filepath='https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?select=kc_house_data.csv'
#df = pd.read_csv(filepath, header=None)

# %% [markdown]
# We use the method <code>head</code> to display the first 5 columns of the dataframe.
# 

# %%
df.head()

# %% [markdown]
# ### Question 1
# 
# Displayed data types of each column
# 

# %%
# Question 1
df.dtypes

# %% [markdown]
# Describe method to obtain a statistical summary of the dataframe.
# 

# %%
df.describe()

# %% [markdown]
# # Module 2: Data Wrangling
# 

# %% [markdown]
# ### Question 2
# 
# Dropped columns <code>"id"</code> and <code>"Unnamed: 0"</code> from axis 1 with <code>drop()</code> method.  
# 

# %%
# Question 2
df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)
df.describe()

# %% [markdown]
# Missing values for the columns <code> bedrooms</code> and <code> bathrooms </code>
# 

# %%
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

# %% [markdown]
# Replaced missing values of column <code>'bedrooms'</code> with the mean of <code>'bedrooms' </code>.
# 

# %%
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)

# %% [markdown]
# Replaced missing values of column <code>'bathrooms'</code> with mean of <code>'bathrooms' </code>.
# 

# %%
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

# %%
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

# %% [markdown]
# # Module 3: Exploratory Data Analysis
# 

# %% [markdown]
# ### Question 3
# 
# Used <code>value_counts</code> method to count the number of houses with unique floor values.  
# 

# %%
# Question 3
num_of_houses= df['floors'].value_counts()
num_of_houses.to_frame()

# %% [markdown]
# ### Question 4
# 
# Plotting with <code>boxplot</code> in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.
# 

# %%
# Question 4
sns.boxplot(x='waterfront', y='price', data=df)

# %% [markdown]
# ### Question 5
# 
# Plotting with <code>regplot</code> in the seaborn library to determine if the feature <code>sqft_above</code> is negatively or positively correlated with price.
# 

# %%
# Question 5
sns.regplot(x='sqft_above', y='price', line_kws={"color":"red"}, data=df)
plt.ylim(0,)

# %% [markdown]
# Pandas <code>corr()</code> method to find the feature that is most correlated with price.
# 

# %%
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.corr()['price'].sort_values()

# %% [markdown]
# # Module 4: Model Development
# 

# %% [markdown]
# Fitting a linear regression model using the longitude feature <code>'long'</code> and calculated R^2.
# 

# %%
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

# %% [markdown]
# ### Question  6
# 
# Fitting a linear regression model to predict <code>'price'</code> using the feature <code>'sqft_living'</code> and calculates R^2.
# 

# %%
#Question 6

X1= df[['sqft_living']]
Y1= df['price']

lm1 = LinearRegression()
lm1.fit(X1,Y1)
r_squared = lm1.score(X1, Y1)

print("R Squared is:", r_squared)

# %% [markdown]
# ### Question 7
# 
# Fitting a linear regression model to predict <code>'price'</code> using all features:
# 

# %%
# Question 7A

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]

Z = df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
Y2= df['price']

lm2 = LinearRegression()
lm2.fit(Z, Y2)

# %% [markdown]
# Calculated R^2 for linear regression model. 
# 

# %%
# Question 7B
r2_lm2 = lm2.score(Z, Y2)

print("R Squared value is:", r2_lm2)

# %% [markdown]
# ### For Question 8
# 
# Estimator names:
# 
# <code>'scale'</code>
# 
# <code>'polynomial'</code>
# 
# <code>'model'</code>
# 
# Model constructor:
# 
# <code>StandardScaler()</code>
# 
# <code>PolynomialFeatures(include_bias=False)</code>
# 
# <code>LinearRegression()</code>
# 

# %%
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

# %% [markdown]
# ### Question 8
# 
# Created pipeline object to predict 'price'. Fitting object using list of <code>features</code>, and calculated R^2.
# 

# %%
# Question 8

pipe=Pipeline(Input)
Z=Z.astype(float)
pipe.fit(Z,Y2)
ypipe=pipe.predict(Z)

print(ypipe[0:4])

print("R Squared is:", pipe.score(Z, Y2))

# %% [markdown]
# # Module 5: Model Evaluation and Refinement
# 

# %% [markdown]
# Import the necessary modules:
# 

# %%
from sklearn.model_selection import train_test_split
print("done")

# %% [markdown]
# Splitting data into training and testing sets:
# 

# %%
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

# %% [markdown]
# ### Question 9
# 
# Fitting Ridge regression object using training data. Regularization parameter set to 0.1. Calculated R^2 using test data.
# 

# %%
from sklearn.linear_model import Ridge

# %%
# Question 9

rr_martin = Ridge(alpha=0.1)

rr_martin.fit(x_train, y_train)
r2test = rr_martin.score(x_test,y_test)

print("R^2:", r2test)

# %% [markdown]
# ### Question 10
# 
# Performed second order polynomial transform on training data and testing data. Fitting a Ridge regression object using the training data. Regularisation parameter set to 0.1. Calculated R^2 utilizing test data.
# 

# %%
# Question 10

pr= PolynomialFeatures(degree=2, include_bias=False)

x_train_pr= pr.fit_transform(x_train) # oops. I only needed to transform the x_train 
x_test_pr = pr.fit_transform(x_test) # oops same here. I only needed to transform the x_test

rmodel=Ridge(alpha=0.1)
rmodel.fit(x_train_pr, y_train)

test_score = rmodel.score(x_test_pr, y_test)

print("R^2 Test Model:", test_score)

# %% [markdown]
# <h2>About the IBM Project Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

# %% [markdown]
# Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01">Michelle Carey</a>, <a href="https://www.linkedin.com/in/jiahui-mavis-zhou-a4537814a?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01">Mavis Zhou</a>
# 

# %% [markdown]
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# <!--## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By      | Change Description                           |
# | ----------------- | ------- | --------------- | -------------------------------------------- |
# | 2020-12-01        | 2.2     | Aije Egwaikhide | Coverted Data describtion from text to table |
# | 2020-10-06        | 2.1     | Lakshmi Holla   | Changed markdown instruction of Question1    |
# | 2020-08-27        | 2.0     | Malika Singla   | Added lab to GitLab                          |
# | 2022-06-13        | 2.3     | Svitlana Kramar | Updated Notebook sharing instructions        |
# | <hr>              |         |                 |                                              |
# 
# 
# --!>
# <p>
# 


