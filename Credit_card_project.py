#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
End to End Model Development for Credit card project
'''


# In[2]:


import os
import sys
import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score,roc_curve


# # Loading the data

# In[3]:


df=pd.read_csv(r"C:\Users\jayan\Downloads\creditcard.csv")


# # Preprocessing the data

# In[4]:


df.sample(7)


# In[5]:


print(f'number of rows in the data : {df.shape[0]} and number of columns : {df.shape[1]}')


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.tail()


# In[10]:


df=df.drop([150000,150001],axis=0)


# In[11]:


df.isnull().sum()


# In[12]:


df['NumberOfDependents']=pd.to_numeric(df['NumberOfDependents'])
df['NumberOfDependents'].dtype


# In[13]:


X=df.iloc[:,:-1]
Y=df.iloc[:,-1]


# In[14]:


# Due to overcome the data leakage problem we are going to split the data in initial stage only
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=42)


# In[15]:


len(X_train),len(Y_train)


# In[16]:


len(X_test),len(Y_test)


# # Feature_Engineering_Stage
# - Handling misssing values
# - converting cat_to_num data
# - Handling outliers
# - Variable Transformation
# - Feature Scaling

# In[17]:


X_train.isnull().sum()


# In[18]:


# since MonthlyIncome and MonthlyIncome.1 columns looks similar so lets compare
c=[]
for i in X_train.index:
    if np.isnan(X_train['MonthlyIncome'][i])==np.isnan(X_train['MonthlyIncome.1'][i]):
        pass
    elif X_train['MonthlyIncome'][i]==X_train['MonthlyIncome.1'][i]:
        pass
    else:
        c.append(i)
if len(c)>0:
    print('not same')
else:
    print('same')


# In[19]:


X_train=X_train.drop(['MonthlyIncome.1'],axis=1)
X_test=X_test.drop(['MonthlyIncome.1'],axis=1)


# In[20]:


X_train.isnull().sum()


# In[21]:


# We are going to apply random_sample imputation technique on both columns which are having null values
def random_sample(X_train,var):
    X_train[var+"_replaced"]=X_train[var].copy()
    s=X_train[var].dropna().sample(X_train[var].isnull().sum() , random_state=42)
    s.index=X_train[X_train[var].isnull()].index
    X_train.loc[X_train[var].isnull(),var+'_replaced']=s


# In[22]:


col=['MonthlyIncome','NumberOfDependents']
for j in col:
    random_sample(X_train,j)


# In[23]:


X_train.head(7)


# In[24]:


X_train.isnull().sum()


# In[25]:


plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.title('MonthlyIncome')
X_train['MonthlyIncome'].plot(kind='kde',color='g',label='Original_monthlyIncome')
X_train['MonthlyIncome_replaced'].plot(kind='kde',color='g',label='Monthly_Replaced')
plt.legend(loc=0)

plt.subplot(1,2,2)
plt.title('NumberOfDependents')
X_train['NumberOfDependents'].plot(kind='kde',color='g',label='Original_NumberOfDependents')
X_train["NumberOfDependents_replaced"].plot(kind='kde',color='r',label='NumberOfDependents_replaced')
plt.legend(loc=0)
print(f'Std of MonthlyIncome Feature : {X_train["MonthlyIncome"].std()}')
print(f'Std of MonthlyIncome_replaced Feature : {X_train["MonthlyIncome_replaced"].std()}')
print()
print(f'Std of NumberOfDependents Feature : {X_train["NumberOfDependents"].std()}')
print(f'Std of NumberOfDependents_replaced Feature : {X_train["NumberOfDependents_replaced"].std()}')
plt.show()


# In[26]:


# same process repeat on the X_test null value columns
def random_sample(X_test,var):
    X_test[var+"_replaced"]=X_test[var].copy()
    s=X_test[var].dropna().sample(X_test[var].isnull().sum() , random_state=42)
    s.index=X_test[X_test[var].isnull()].index
    X_test.loc[X_test[var].isnull(),var+'_replaced']=s


# In[27]:


col=['MonthlyIncome','NumberOfDependents']
for j in col:
    random_sample(X_test,j)


# In[28]:


plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.title('MonthlyIncome')
X_test['MonthlyIncome'].plot(kind='kde',color='g',label='Original_monthlyIncome')
X_test['MonthlyIncome_replaced'].plot(kind='kde',color='g',label='Monthly_Replaced')
plt.legend(loc=0)

plt.subplot(1,2,2)
plt.title('NumberOfDependents')
X_test['NumberOfDependents'].plot(kind='kde',color='g',label='Original_NumberOfDependents')
X_test["NumberOfDependents_replaced"].plot(kind='kde',color='r',label='NumberOfDependents_replaced')
plt.legend(loc=0)
print(f'Std of MonthlyIncome Feature : {X_test["MonthlyIncome"].std()}')
print(f'Std of MonthlyIncome_replaced Feature : {X_test["MonthlyIncome_replaced"].std()}')
print()
print(f'Std of NumberOfDependents Feature : {X_test["NumberOfDependents"].std()}')
print(f'Std of NumberOfDependents_replaced Feature : {X_test["NumberOfDependents_replaced"].std()}')
plt.show()


# In[29]:


# since null values cleared we can remove original features because proper data was maintained in another feature
X_train=X_train.drop(['MonthlyIncome','NumberOfDependents'],axis=1)
X_test=X_test.drop(['MonthlyIncome','NumberOfDependents'],axis=1)


# In[30]:


X_train.isnull().sum()


# # From the X_train and X_test seperate Numerical features and Categorical features

# In[121]:


X_train.info()


# In[122]:


X_train_num_cols=X_train.select_dtypes(exclude='object')
X_train_cat_cols=X_train.select_dtypes(include='object')


# In[123]:


X_test_num_cols=X_test.select_dtypes(exclude='object')
X_test_cat_cols=X_test.select_dtypes(include='object')


# In[124]:


# Converting X_train_cat_cols to numbers and same we need to apply on X_test_cat_cols
X_train_cat_cols.head()


# In[125]:


# Gender and region comes under nominal part
# Rented_OwnHouse,Occupation and Education
# fit wiil do the math operations but it will not update, transfrom will update the solutions
X_train_cat_cols.reset_index(drop=True,inplace=True)


# In[130]:


from sklearn.preprocessing import OneHotEncoder
one_hot=OneHotEncoder(handle_unknown='ignore')
one_hot.fit(X_train_cat_cols[['Gender','Region']])
f=one_hot.transform(X_train_cat_cols[['Gender','Region']]).toarray()
v=pd.DataFrame(data=f)
v.columns=one_hot.get_feature_names_out()
X_train_cat_cols=pd.concat([X_train_cat_cols,v],axis=1)
X_train_cat_cols.head(7)


# In[37]:


# We are going to use odinal encoding on rented_ownhouse, occupation and education features
from sklearn.preprocessing import OrdinalEncoder
od_en=OrdinalEncoder()
od_en.fit(X_train_cat_cols[['Rented_OwnHouse','Occupation','Education']])
p=od_en.transform(X_train_cat_cols[['Rented_OwnHouse','Occupation','Education']])
f=pd.DataFrame(data=p)
f.columns=od_en.get_feature_names_out()


# In[38]:


od_en.categories_


# In[39]:


f


# In[40]:


X_train_cat_cols.columns


# In[41]:


X_train_cat_cols=X_train_cat_cols.drop(['Gender', 'Region', 'Rented_OwnHouse', 'Occupation', 'Education'],axis=1)
X_train_cat_cols.head()


# In[42]:


X_train_cat_cols=pd.concat([X_train_cat_cols,f],axis=1)
X_train_cat_cols.head()


# In[43]:


# same things we can assign to test data
X_test_cat_cols.head()


# In[44]:


p=one_hot.transform(X_test_cat_cols[['Gender','Region']]).toarray()
f=od_en.transform(X_test_cat_cols[['Rented_OwnHouse','Occupation','Education']])


# In[45]:


p


# In[46]:


p_=pd.DataFrame(data=p)
p_.columns=one_hot.get_feature_names_out()
p_.head()


# In[47]:


f_=pd.DataFrame(data=f)
f_.columns=od_en.get_feature_names_out()
f_.head()


# In[48]:


g=pd.concat([p_,f_],axis=1)
g.head()


# In[49]:


X_test_cat_cols=X_test_cat_cols.drop(['Gender', 'Region', 'Rented_OwnHouse', 'Occupation', 'Education'],axis=1)
X_test_cat_cols.head()


# In[50]:


X_test_cat_cols.reset_index(drop=True,inplace=True)


# In[51]:


X_test_cat_cols=pd.concat([X_test_cat_cols,g],axis=1)
X_test_cat_cols.head()


# In[52]:


for i in X_train_num_cols.columns:
    sns.displot(X_train_num_cols[i],kind='kde')


# In[53]:


def fun(data_c,var):
    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    data_c[var].plot(kind='kde')
    plt.subplot(1,3,2)
    sns.boxplot(x=data_c[var])
    plt.subplot(1,3,3)
    stats.probplot(data_c[var],dist='norm',plot=plt)
    plt.show()


# In[54]:


for i in X_train_num_cols.columns:
    fun(X_train_num_cols,i)


# In[55]:


# Applying Log for X_train_num_cols to main Normal Distribution and reduce the outliers
X_train_num_cols.head()


# In[56]:


for i in X_train_num_cols.columns:
    X_train_num_cols[i+'_log']=np.log(X_train_num_cols[i]+1)


# In[57]:


X_train_num_cols.head()


# In[58]:


for i in X_train_num_cols.columns:
    if '_log' in i:
        fun(X_train_num_cols,i)


# In[59]:


X_train_num_cols=X_train_num_cols.drop(['NPA Status', 'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'MonthlyIncome_replaced', 'NumberOfDependents_replaced'],axis=1)


# In[60]:


X_train_num_cols.columns


# In[61]:


for i in X_test_num_cols.columns:
    X_test_num_cols[i+'_log']=np.log(X_test_num_cols[i]+1)
    
X_test_num_cols=X_test_num_cols.drop(["NPA Status", "RevolvingUtilizationOfUnsecuredLines", "age",
       "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
       "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
       "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
       "MonthlyIncome_replaced", "NumberOfDependents_replaced"],axis=1)


# In[62]:


X_test_num_cols.columns


# # Handling_Outliers

# In[63]:


def han_out(X_train_num_cols,var):
    plt.figure(figsize=(5,3))
    sns.boxplot(x=X_train_num_cols[var])
    plt.show()


# In[64]:


for i in X_train_num_cols.columns:
    han_out(X_train_num_cols,i)


# In[65]:


# trimming on X_train_num_cols
def fun_1(df,var):
    iqr=df[var].quantile(0.75)-df[var].quantile(0.25)
    upper=df[var].quantile(0.75)+(1.5*iqr)
    lower=df[var].quantile(0.25)-(1.5*iqr)
    return upper ,lower
for i in X_train_num_cols.columns:
    upper_value,lower_value=fun_1(X_train_num_cols,i)
    X_train_num_cols[i+'_trimming']=np.where(X_train_num_cols[i]>upper_value,upper_value,
        np.where(X_train_num_cols[i]<lower_value,lower_value,X_train_num_cols[i]))
for j in X_train_num_cols.columns:
    if '_trimming' in j:
        han_out(X_train_num_cols,j)


# In[66]:


X_train_num_cols=X_train_num_cols.drop(['NPA Status_log', 'RevolvingUtilizationOfUnsecuredLines_log', 'age_log',
       'NumberOfTime30-59DaysPastDueNotWorse_log', 'DebtRatio_log',
       'NumberOfOpenCreditLinesAndLoans_log', 'NumberOfTimes90DaysLate_log',
       'NumberRealEstateLoansOrLines_log',
       'NumberOfTime60-89DaysPastDueNotWorse_log',
       'MonthlyIncome_replaced_log', 'NumberOfDependents_replaced_log'],axis=1)


# In[67]:


X_train_num_cols.columns


# In[68]:


# same operations we are going to update on the X_test_num_cols
# trimming on X_test_num_cols
def fun_1(df,var):
    iqr=df[var].quantile(0.75)-df[var].quantile(0.25)
    upper=df[var].quantile(0.75)+(1.5*iqr)
    lower=df[var].quantile(0.25)-(1.5*iqr)
    return upper ,lower
for i in X_test_num_cols.columns:
    upper_value,lower_value=fun_1(X_test_num_cols,i)
    X_test_num_cols[i+'_trimming']=np.where(X_test_num_cols[i]>upper_value,upper_value,
        np.where(X_test_num_cols[i]<lower_value,lower_value,X_test_num_cols[i]))
for j in X_test_num_cols.columns:
    if '_trimming' in j:
        han_out(X_test_num_cols,j)


# In[69]:


X_test_num_cols=X_test_num_cols.drop(['NPA Status_log', 'RevolvingUtilizationOfUnsecuredLines_log', 'age_log',
       'NumberOfTime30-59DaysPastDueNotWorse_log', 'DebtRatio_log',
       'NumberOfOpenCreditLinesAndLoans_log', 'NumberOfTimes90DaysLate_log',
       'NumberRealEstateLoansOrLines_log',
       'NumberOfTime60-89DaysPastDueNotWorse_log',
       'MonthlyIncome_replaced_log', 'NumberOfDependents_replaced_log'],axis=1)


# In[70]:


X_test_num_cols.columns


# In[71]:


X_train_num_cols.reset_index(drop=True,inplace=True)


# In[72]:


# join the data as it is
X_train_cleaned=pd.DataFrame()
X_train_cleaned=pd.concat([X_train_num_cols,X_train_cat_cols],axis=1)
X_train_cleaned.head(7)


# In[73]:


X_train_cleaned.shape


# In[74]:


X_test_num_cols.reset_index(drop=True,inplace=True)


# In[75]:


# final X_test
X_test_cleaned=pd.DataFrame()
X_test_cleaned=pd.concat([X_test_num_cols,X_test_cat_cols],axis=1)
X_test_cleaned.head(7)


# In[76]:


X_test_cleaned.shape


# In[77]:


# checking columns names are matching or not exactly
c1=[]
for i,j in enumerate(X_train_cleaned.columns):
    if X_test_cleaned.columns[i]==j:
        pass
    else:
        c1.append(j)
print(c1)


# # Feature_Selection
# - feature selection can be done in mainly 2 tecniques
# - filtermethods
#       - constant
#       - quesi constant technique
# - correlation
# - hypothesis testing

# In[78]:


# constant -> any column variance is zero means we can remove the column
vt=VarianceThreshold(threshold=0.0)
vt.fit(X_train_cleaned)


# In[79]:


X_train_cleaned.columns[~vt.get_support()]


# In[80]:


X_train_cleaned=X_train_cleaned.drop(['NPA Status_log_trimming',
       'NumberOfTime30-59DaysPastDueNotWorse_log_trimming',
       'NumberOfTimes90DaysLate_log_trimming',
       'NumberOfTime60-89DaysPastDueNotWorse_log_trimming'],axis=1)
X_test_cleaned=X_test_cleaned.drop(['NPA Status_log_trimming',
       'NumberOfTime30-59DaysPastDueNotWorse_log_trimming',
       'NumberOfTimes90DaysLate_log_trimming',
       'NumberOfTime60-89DaysPastDueNotWorse_log_trimming'],axis=1)


# In[81]:


X_test_cleaned.shape


# In[82]:


X_train_cleaned.shape


# In[83]:


# quasi_constant -> variance should be 0.1
vt=VarianceThreshold(threshold=0.1)
vt.fit(X_train_cleaned)


# In[84]:


X_train_cleaned.columns[~vt.get_support()]


# In[85]:


X_train_cleaned=X_train_cleaned.drop(['RevolvingUtilizationOfUnsecuredLines_log_trimming', 'age_log_trimming'],axis=1)
X_test_cleaned=X_test_cleaned.drop(['RevolvingUtilizationOfUnsecuredLines_log_trimming', 'age_log_trimming'],axis=1)


# In[86]:


X_train_cleaned.shape


# In[87]:


# using the label encoding technique we are going to convert into numbers
lb_t=LabelEncoder()
lb_t.fit(Y_train)
print(lb_t.classes_)
f=lb_t.transform(Y_train)
Y_train_n=pd.DataFrame(data=f,columns=['target'])
Y_test_n=pd.DataFrame(data=lb_t.transform(Y_test),columns=['target'])


# - We have 15 column know we are going to apply hypothesis testing and find the best columns for the model development

# In[88]:


# hypothesis testing correlation [pearson_correlation] and p_value


# In[89]:


co=[]
for i in X_train_cleaned.columns:
    s=pearsonr(X_train_cleaned[i],Y_train_n['target'])
    co.append(s)
co=np.array(co)
co


# In[90]:


p_value=pd.Series(co[:,1] , index=X_train_cleaned.columns)
p_value


# In[91]:


k=[]
g=list(p_value)
for i in p_value:
    if i>0.05:
        k.append(list(g).index(i))
print(k)


# In[92]:


p_value.sort_values(ascending=True).plot.bar()


# In[93]:


X_train_cleaned.columns


# In[94]:


X_train_cleaned=X_train_cleaned.drop(['Rented_OwnHouse','Gender_Female','Gender_Male','DebtRatio_log_trimming','Occupation','Region_South'],axis=1)


# In[95]:


X_test_cleaned=X_test_cleaned.drop(['Rented_OwnHouse','Gender_Female','Gender_Male','DebtRatio_log_trimming','Occupation','Region_South'],axis=1)


# In[96]:


X_test_cleaned.shape


# In[97]:


Y_train_n['target'].value_counts()


# In[98]:


Y_train_n=np.array(Y_train_n).ravel()
Y_test_n=np.array(Y_test_n).ravel()


# In[99]:


# using upsampling we are going to balance the data
get_ipython().system('pip install imblearn')


# In[100]:


print('labels count for Bad 0 ={}'.format(sum(Y_train_n==0)))
print('labels count for Good 0 ={}'.format(sum(Y_train_n==1)))
print()
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2)
X_train_up,Y_train_up=sm.fit_resample(X_train_cleaned,Y_train_n)
print('labels count for Bad 0 ={}'.format(sum(Y_train_up==0)))
print('labels count for Good 0 ={}'.format(sum(Y_train_up==1)))


# In[101]:


X_train_up.shape


# In[102]:


Y_train_up.shape


# - Since the Data is balanced we need to work on Model Development

# In[103]:


def knn_algo(train_x,train_y,test_x,test_y):
    knn=KNeighborsClassifier(n_neighbors=5) #default k_value 5.
    knn.fit(train_x,train_y)
    Y_train_pred=knn.predict(train_x)
    Y_test_pred=knn.predict(test_x)
    print(f'Train accuracy : {accuracy_score(train_y,Y_train_pred)}')
    print(f'Test accuracy : {accuracy_score(test_y,Y_test_pred)}')
    print(f'confusion matrix : {confusion_matrix(test_y,Y_test_pred)}')
    print(f'classification Report : {classification_report(test_y,Y_test_pred)}')


# In[104]:


def log_algo(train_x,train_y,test_x,test_y):
    lr=LogisticRegression()
    lr.fit(train_x,train_y)
    Y_train_pred=lr.predict(train_x)
    Y_test_pred=lr.predict(test_x)
    print(f'Train accuracy : {accuracy_score(train_y,Y_train_pred)}')
    print(f'Test accuracy : {accuracy_score(test_y,Y_test_pred)}')
    print(f'confusion matrix : {confusion_matrix(test_y,Y_test_pred)}')
    print(f'classification Report : {classification_report(test_y,Y_test_pred)}')


# In[105]:


def nb_algo(train_x,train_y,test_x,test_y):
    nb=GaussianNB()
    nb.fit(train_x,train_y)
    Y_train_pred=nb.predict(train_x)
    Y_test_pred=nb.predict(test_x)
    print(f'Train accuracy : {accuracy_score(train_y,Y_train_pred)}')
    print(f'Test accuracy : {accuracy_score(test_y,Y_test_pred)}')
    print(f'confusion matrix : {confusion_matrix(test_y,Y_test_pred)}')
    print(f'classification Report : {classification_report(test_y,Y_test_pred)}')


# In[106]:


def dt_algo(train_x,train_y,test_x,test_y):
    dt=DecisionTreeClassifier()
    dt.fit(train_x,train_y)
    Y_train_pred=dt.predict(train_x)
    Y_test_pred=dt.predict(test_x)
    print(f'Train accuracy : {accuracy_score(train_y,Y_train_pred)}')
    print(f'Test accuracy : {accuracy_score(test_y,Y_test_pred)}')
    print(f'confusion matrix : {confusion_matrix(test_y,Y_test_pred)}')
    print(f'classification Report : {classification_report(test_y,Y_test_pred)}')


# In[107]:


def calling(train_x,train_y,test_x,test_y):
    print('----------knn----------')
    knn_algo(train_x,train_y,test_x,test_y)
    print('----------logistic----------')
    log_algo(train_x,train_y,test_x,test_y)
    print('----------Naive_bayes----------')
    nb_algo(train_x,train_y,test_x,test_y)
    print('----------Decision_Tree----------')
    dt_algo(train_x,train_y,test_x,test_y)


# In[108]:


calling(X_train_up,Y_train_up,X_test_cleaned,Y_test_n)


# In[109]:


k_value=np.arange(3,25,2)
k_value


# In[110]:


test_accuracy=[]
for i in k_value:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_up,Y_train_up)
    test_accuracy.append(knn.score(X_test_cleaned,Y_test_n))
test_accuracy


# # Best Model
# - AUC -> Area under the curve
# - roc -> Rectifier operating charateristics

# In[111]:


# knn 
knn_algo=KNeighborsClassifier()
knn_algo.fit(X_train_up,Y_train_up)
knn_pred=knn_algo.predict(X_test_cleaned)
# Logistic regression
lr_algo=LogisticRegression()
lr_algo.fit(X_train_up,Y_train_up)
lr_pred=lr_algo.predict(X_test_cleaned)
# Naive bayes
nb_algo=GaussianNB()
nb_algo.fit(X_train_up,Y_train_up)
nb_pred=nb_algo.predict(X_test_cleaned)
# Decision Tree
dt_algo=DecisionTreeClassifier()
dt_algo.fit(X_train_up,Y_train_up)
dt_pred=dt_algo.predict(X_test_cleaned)


# In[112]:


# give the Model outcomes to auc and roc curves
fpr_knn,tpr_knn,treshold=roc_curve(Y_test_n,knn_pred)
fpr_lr,tpr_lr,treshold=roc_curve(Y_test_n,lr_pred)
fpr_nb,tpr_nb,treshold=roc_curve(Y_test_n,nb_pred)
fpr_dt,tpr_dt,treshold=roc_curve(Y_test_n,dt_pred)


# In[113]:


plt.figure(figsize=(5,3))
plt.plot([0,1],[0,1],'k--',label='50% AUC')
plt.plot(fpr_knn,tpr_knn,color='g',label='knn')
plt.plot(fpr_lr,tpr_lr,color='r',label='LR')
plt.plot(fpr_nb,tpr_nb,color='b',label='NB')
plt.plot(fpr_dt,tpr_dt,color='y',label='DT')
plt.legend(loc=0)
plt.show()


# In[114]:


# finally the Best Model is Logistic Regression


# In[115]:


X_train_up.columns


# # New Data Point

# In[116]:


if lr_algo.predict([[4.2,3,1200,2,0,0,1,0,1]])[0]==1:
    print('Good Customer')
else:
    print('Bad Customer')


# # Save the Model

# In[117]:


import pickle


# In[118]:


with open('credit_card_project.pkl','wb') as f:
    pickle.dump(lr_algo,f)


# In[ ]:




