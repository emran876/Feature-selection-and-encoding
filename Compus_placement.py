#!/usr/bin/env python
# coding: utf-8

# # Classification problem

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
get_ipython().run_line_magic('matplotlib', 'inline')


#  Data scurce: Kaggle

# In[3]:


df = pd.read_csv('Placement_Data_Full_Class.csv')


# In[4]:


df.head(2)


# I'm going to drop few columns for better prediction

# In[5]:


# Label column: admission score: Placed / Not placed in status column


# # Exploratory Data Analysis

# In[6]:


df['salary'].isnull().sum()


# # Replace missing data

#  Replace Nan values with median value. Mean is sensitive to the outliers

# In[7]:


df1=df.fillna(df.median())
#df1=df.fillna(df.mean())


# In[8]:


df1['salary'].isnull().sum()


# In[9]:


#for col in df.columns:
 #   print(col, ' :', len(df[col].unique()), 'labels' )


# # Visualize data with pandas ProfileReport

# In[10]:


profile= ProfileReport(df, title='Pandas Profile Report', explorative=True)


# In[12]:


profile.to_widgets()


# # Save profile report to html file

# In[11]:


profile.to_file('Campus_logistics_profile.html')


# In[41]:


#df['hsc_s'].unique()


# In[12]:


df['hsc_s'].value_counts()


# In[13]:


df.set_index('sl_no',inplace=True)


# # Encoding

# In[14]:


def hot_encoding(df,col, prefix):
    hot= pd.get_dummies(df[col], prefix=prefix, drop_first=True)
    df=pd.concat([hot,df], axis=1)
    df.drop(col, axis=1, inplace=True)
    return df
    


# In[15]:


def hot_encodingr(df,col, prefix):
    hot= pd.get_dummies(df[col], prefix=prefix, drop_first=True)
    df=pd.concat([df,hot], axis=1)
    df.drop(col, axis=1, inplace=True)
    return df


# # These categorical columns are encoded

# Features' columns (categorical)

# In[16]:


cat_cols=('gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation' )


# Encoding the output column

# In[17]:


df1=hot_encodingr(df1, 'status', 'status')


# In[18]:


df1.head(2)


# In[19]:


means1 = df1.groupby('degree_t')['status_Placed'].mean().to_dict()


# In[20]:


means2 = df1.groupby('specialisation')['status_Placed'].mean().to_dict()


# In[21]:


means3 = df1.groupby('degree_t')['status_Placed'].mean().to_dict()


# In[22]:


means3


# In[23]:


#means1


# In[24]:


df1.head(2)


# Encoding features columns

# In[25]:


for col in cat_cols:
    df1=hot_encoding(df1, col, col)


# After encoding

# In[26]:


df1.head(2)


# # Scaling the numerical values

# In[27]:


from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler


# In[28]:


zscore_cols=['ssc_p', 'hsc_p', 'degree_p', 'mba_p']


# In[29]:


#df1['ssc_p']=zscore(df1['ssc_p'])


# In[30]:


#df1.head(2)


# In[31]:


for col in zscore_cols:
    df1[col]=zscore(df1[col])


# In[32]:


df1.head(2)


# In[33]:


scaler = MinMaxScaler()


# In[34]:


df1[['etest_p', 'salary']] = scaler.fit_transform(df1[['etest_p', 'salary']])


# In[35]:


df1.head(3)


# In[36]:


# Set features and output matrices


# In[37]:


X=df1.iloc[:, 0:15].values


# In[38]:


y=df1.iloc[:, -1].values


# In[39]:


df1.shape


# In[40]:


X.shape


# In[41]:


y.shape


# # Training 

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)


# In[44]:


from sklearn.linear_model import LogisticRegression


# # Logistic regression

# In[45]:


lr = LogisticRegression()


# In[46]:


lr.fit(X_train,y_train)


# In[47]:


lr.score(X_test,y_test)


# # Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier


# In[49]:


#random_forest = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rfc = RandomForestClassifier(n_estimators=200, random_state=3)


# In[50]:


rfc.fit(X_train,y_train)


# In[51]:


rfc.score(X_test,y_test)


# # Xgboost classifier

# In[52]:


from  xgboost import XGBClassifier


# In[53]:


xgb=XGBClassifier(random_state=1,learning_rate=0.01)
xgb.fit(X_train, y_train)
xgb.score(X_test,y_test)


# In[54]:


from sklearn.metrics import precision_score


# In[55]:


y_pred=xgb.predict(X_test)


# In[56]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# #dfcm= pd.DataFrame(
#     confusion_matrix(y_test, y_pred, labels=['yes', 'no']), 
#     index=['true:yes', 'true:no'], 
#     columns=['pred:yes', 'pred:no']
# )

# In[57]:


cm = confusion_matrix(y_test, y_pred)


# In[58]:


#dfc=pd.DataFrame(cm, index=['Not Placed', 'Placed'], index=['Not Placed', 'Placed'])


# In[59]:


dfc=pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True) .transpose()


# In[60]:


dfc


# In[61]:


report=classification_report(y_test,y_pred, output_dict=True )


# In[62]:


dfr = pd.DataFrame(report).transpose()


# In[63]:


dfr


# # Hyper parameter tuning

# # Logistic regression

# In[159]:


param_grid_lr=[
    {'penalty': ['l1', 'l2', 'elasticnet', 'none'] ,
    'C': np.logspace(-4,4, 20),
    'solver': ['lbfgs', 'newtog-cg', 'liblinear', 'sag', 'saga'],
     'max_iter': [1, 10, 100, 1000, 2000]
    }
]


# In[160]:


lreg = LogisticRegression()


# In[161]:


from sklearn.model_selection import GridSearchCV


# In[162]:


cvlrge= GridSearchCV(lreg, param_grid=param_grid_lr, cv=5, verbose=True, n_jobs=-1)


# In[163]:


#param_grid_lr


# In[164]:


best_lreg=cvlrge.fit(X,y)


# In[167]:


best_lreg.best_estimator_


# In[168]:


best_lreg.score(X,y)


# In[169]:


best_lreg.best_score_


# # Random forest

# In[171]:


rfc= RandomForestClassifier()


# In[172]:


n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

param_rfc = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)


# In[173]:


cv_rfc = GridSearchCV(rfc, param_rfc, cv = 5, verbose = 1, n_jobs = -1) # multi-threaded


# In[174]:


best_rfc= cv_rfc.fit(X,y)


# In[175]:


best_rfc.best_estimator_


# In[176]:


best_rfc.score(X,y)


# In[177]:


best_rfc.best_score_


# # xgboost

# In[178]:


xgb = XGBClassifier(objective = 'binary:logistic')


# In[179]:


param_xgb={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[180]:


cv_xgb = GridSearchCV(xgb, param_xgb, cv = 5, verbose = 1, n_jobs = -1)


# In[181]:


best_xgb= cv_xgb.fit(X,y)


# In[182]:


best_xgb.best_estimator_


# In[183]:


best_xgb.score(X,y)


# In[184]:


best_xgb.best_score_


# In[ ]:




