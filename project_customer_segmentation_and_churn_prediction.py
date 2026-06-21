

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel("Telco_customer_churn.xlsx")

df.shape

df.info()

df['Churn Label'].value_counts()

plt.figure(figsize=(8,5))
sns.histplot(df['Tenure Months'],bins=30,kde=True)
plt.xlabel('Tenure Months')
plt.ylabel('Customer count')
plt.title('Distribution of Tenure Months vs no. of Customers')
plt.show()

df['Tenure Months'].max()
df['Tenure Months'].min()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn Label',y='Tenure Months',data=df)
plt.xlabel('Churn Label')
plt.ylabel('Tenure Months')
plt.title('Distribution of Tenure Months vs Churn Label')
plt.show()

df['Churn Label'].unique()

df[df['Churn Label']=='Yes']

plt.figure(figsize=(8,5))
sns.histplot(df['Monthly Charges'], bins=30, kde=True)
plt.xlabel('months charges')
plt.ylabel('Customer count')
plt.title('Distribution of months charges vs no of customer')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn Label',y='Monthly Charges',data=df)
plt.xlabel('Churn Label')
plt.ylabel('Tenure Months')
plt.title('Distribution of Tenure Months vs Monthly charges')
plt.show()

df[df['Churn Label']=='Yes']['Monthly Charges'].quantile([0.25,0.5,0.75])

df[df['Churn Label']=='No']['Monthly Charges'].quantile([0.25,0.5,0.75])

df['Monthly Charges'].describe()

df['Monthly Charges'].quantile([0.25,0.5,0.75])

df[df['Churn Label']=='Yes']['Monthly Charges'].quantile([0.25,0.5,0.75])
df[df['Churn Label']=='No']['Monthly Charges'].quantile([0.25,0.5,0.75])



df['Contract'].unique()

df['Contract'].value_counts()

plt.figure(figsize=(8,5))
sns.countplot(x='Contract',hue ='Churn Label',data=df)
plt.xlabel('Contract')
plt.ylabel('Customer count')
plt.title('Distribution of Contract vs no of customer')
plt.show()

df['Internet Service'].unique()

plt.figure(figsize=(8,5))
sns.countplot(x='Internet Service',hue ='Churn Label',data=df)
plt.xlabel('Internet Service')
plt.ylabel('Customer count')
plt.title('Distribution of CInternet Service vs no of customer')
plt.show()

df.info()

df['Payment Method'].unique()

plt.figure(figsize=(8,6))
sns.countplot(x='Payment Method',hue ='Churn Label',data=df)
plt.xlabel('Payment Method')
plt.ylabel('Customer count')
plt.title('Distribution of Payment Method vs no of customer')
plt.show()

df['Tech Support'].unique()

plt.figure(figsize=(8,5))
sns.countplot(x='Tech Support',hue ='Churn Label',data=df)
plt.xlabel('Tech Support')
plt.ylabel('Customer count')
plt.title('Distribution of Tech Support vs no of customer')
plt.show()

avg_tenure=df.groupby('Churn Label')['Tenure Months'].mean()

avg_tenure

df.info()

numerical_cols=['Tenure Months','Monthly Charges','Churn Value','Churn Score','CLTV']
correlation_matrix=df[numerical_cols].corr()

correlation_matrix

contract_churn=pd.crosstab(df['Contract'],df['Churn Label'],normalize='index')
contract_churn

"""data cleaning"""

df.info()

df['Total Charges']

df['Total Charges']=pd.to_numeric(df['Total Charges'],errors='coerce')
df['Total Charges']

df['Total Charges'].isnull().sum()

df[df['Total Charges'].isnull()]['Tenure Months']

df[df['Total Charges'].isnull()]['Tenure Months'].shape

df['Total Charges']=df['Total Charges'].fillna(0)

df['Total Charges'].isnull().sum()

drop_columns=['CustomerID','Count','Country','State','Zip Code','Lat Long','Latitude','Longitude','Churn Label','Churn Score','CLTV','Churn Reason']

df=df.drop(columns=drop_columns)

df.info()

df_encoded=pd.get_dummies(df,drop_first=True)

df_encoded.head()

df_encoded.shape

df=df.drop(columns=['City'])

df.shape

df.info()

df_encoded=pd.get_dummies(df,drop_first=True)

df_encoded.shape

df_encoded

x=df_encoded.drop(columns=['Churn Value'])
y=df_encoded['Churn Value']

x.shape

y.shape

print(x.head())

print(y.head())

""""Machine learning Implementation"
"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_test.shape

x_train.shape

y_train.shape

y_test.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

rf_model=RandomForestClassifier(n_estimators=100,random_state=42)

rf_model.fit(x_train,y_train)

y_pred=rf_model.predict(x_test)

y_test

y_pred

accuracy=accuracy_score(y_test,y_pred)

accuracy

cm=confusion_matrix(y_test,y_pred)

cm

print(classification_report(y_test,y_pred))



"""approach 1- Handle class imbalance"""

rf_balance=RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')
rf_balance.fit(x_train,y_train)
y_pred_balance=rf_balance.predict(x_test)
accuracy_balance=accuracy_score(y_test,y_pred_balance)
cm_balance=confusion_matrix(y_test,y_pred_balance)
print("Accuracy with class balancing:",accuracy_balance)
print("Confusion Matrix with class balancing:")
print(cm_balance)
print(classification_report(y_test,y_pred_balance))

"""approach 2: hyperparameter tunning"""

rf_tuned=RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42,class_weight='balanced')
rf_tuned.fit(x_train,y_train)
y_pred_tuned=rf_tuned.predict(x_test)
accuracy_tuned=accuracy_score(y_test,y_pred_tuned)
cm_tuned=confusion_matrix(y_test,y_pred_tuned)
print("Accuracy with class balancing:",accuracy_tuned)
print("Confusion Matrix with class balancing:")
print(cm_tuned)
print(classification_report(y_test,y_pred_tuned))

for w in [3,4,5]:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight={0:1,1:w},
        random_state=42
    )

    rf.fit(x_train,y_train)
    pred = rf.predict(x_test)

    print(f"\nWeight={w}")
    print(classification_report(y_test,pred))

"""approach 3 : feature importance"""

import pandas as pd
feature_importance=pd.DataFrame({
    'features':x.columns,
    'importance':rf_tuned.feature_importances_

})
feature_importance=feature_importance.sort_values(by='importance',ascending=False)
print(feature_importance)

print(feature_importance.tail(15))

x_selected=x.drop(['Phone Service_Yes','Multiple Lines_No phone service'],axis=1)

x_train_sel,x_test_sel,y_train_sel,y_test_sel=train_test_split(x_selected,y,test_size=0.2,random_state=42)

rf_selected=RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42,class_weight='balanced')
rf_selected.fit(x_train_sel,y_train_sel)
y_pred_selected=rf_selected.predict(x_test_sel)
print(classification_report(y_test_sel,y_pred_selected))

"""approach combination of tree and depth

"""

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

n_estimators_list=[100,200,300,400,500]
max_depth_list=[5,10,15,20]
results=[]
for n in n_estimators_list:
    for d in max_depth_list:
        rf=RandomForestClassifier(n_estimators=n,max_depth=d,random_state=42,class_weight='balanced')
        rf.fit(x_train,y_train)
        y_pred_tuned=rf.predict(x_test)
        accuracy_tuned=accuracy_score(y_test,y_pred_tuned)
        recall=recall_score(y_test,y_pred_tuned)
        precision=precision_score(y_test,y_pred_tuned)
        f1=f1_score(y_test,y_pred_tuned)
        results.append({
            'n_estimators':n,
            'max_depth':d,
            'accuracy':accuracy_tuned,
            'recall':recall,
            'precision':precision,
            'f1_score':f1
        })
result_df=pd.DataFrame(results)
result_df=result_df.sort_values(by=['recall','accuracy'],ascending =False)
print(  result_df.head(20))

from sklearn.model_selection import cross_val_score
final_rf=RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42,class_weight='balanced')

cv_accuracy=cross_val_score(final_rf,x,y,cv=5,scoring='accuracy')
cv_accuracy

cv_accuracy.mean()

cv_recall=cross_val_score(final_rf,x,y,cv=5,scoring='recall')
cv_recall

cv_recall.mean()

"""ROC-AUC Evaluation

"""

from sklearn.metrics import roc_curve,roc_auc_score

import matplotlib.pyplot as plt

"""# Generate churn probability scores for customer segmentation"""

y_prob=rf_model.predict_proba(x)

y_prob1=rf_tuned.predict_proba(x_test)

churn_probability=y_prob[:,1]

churn_prob=y_prob1[:,1]
fpr,tpr,threshhold=roc_curve(y_test,churn_prob)
auc_score=roc_auc_score(y_test,churn_prob)
print(auc_score)

"""CUSTOMER SEGMENTATION"""

segmentation_data=pd.DataFrame({
    'tenure':x['Tenure Months'],
    'monthly_charges':x['Monthly Charges'],
    'total_charges':x['Total Charges'],
    'churn probability':churn_probability
})

segmentation_data

"""implementation of kmeans"""

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaled_data=scaler.fit_transform(segmentation_data)
print(scaled_data[:5])

from sklearn.cluster import KMeans
wcss=[]
for k in range(1,16):
  kmeans=KMeans(n_clusters=k,random_state=42)
  kmeans.fit(scaled_data)
  wcss.append(kmeans.inertia_)
  plt.figure(figsize=(8,6))
plt.plot(range(1,16),wcss,marker='o',linestyle='--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method to find optimal number of clusters')
plt.show()

kmeans=KMeans(n_clusters=3,random_state=42)

clusters=kmeans.fit_predict(scaled_data)

segmentation_data

segmentation_data['Cluster']=clusters

segmentation_data

cluster_summary=segmentation_data.groupby('Cluster').mean()

cluster_summary

print(segmentation_data.shape)
print(segmentation_data.columns)
print(segmentation_data['Cluster'].value_counts(dropna=False))

cluster_summary={
    0:'Budget Loyal Customers',
    1:'Loyal premium customers',
    2:'High risk new customers'
}

segmentation_data['Cluster Segment'] = segmentation_data['Cluster'].map(cluster_summary)

segmentation_data

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='tenure',
    y='churn probability',
    hue='Cluster',
    data=segmentation_data,
    palette='Spectral'
)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='monthly_charges',
    y='churn probability',
    hue='Cluster',
    data=segmentation_data,
    palette='Spectral'
)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='tenure',
    y='churn probability',
    hue='Cluster',
    data=segmentation_data,
    palette='Spectral'
)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='total_charges',
    y='churn probability',
    hue='Cluster',
    data=segmentation_data,
    palette='Spectral'
)

plt.show()

plt.savefig("roc_curve.png")
