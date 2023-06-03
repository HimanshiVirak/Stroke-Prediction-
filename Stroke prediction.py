#!/usr/bin/env python
# coding: utf-8

# ## Assignment Details
# <hr style="border:1px solid black"> </hr>
# 
# **Module Code: CSMAI21**
# <br>**Assignment Report Title:Artificial Intelligence and Machine Learning**
# <br>**Student Number:30852881**
# <br>**Date (when work was completed): 8th March 2023**
# <br>**Actual hours spent on assignment: 20 hours**
# <hr style="border:1px solid black"> </hr>

# In[1]:

#Importing Libraries

# library for prepare the dataset
import os
import zipfile

# library for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
pal1 = sns.color_palette("icefire")
sns.set_palette(pal1) 

# library for data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# library for modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold

# library for model evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")


# In[2]:


from configparser import ConfigParser
#Load Config File 
config = ConfigParser()
config.read('configfile.ini')


# In[3]:


for i, j in config.items():
    print(f"{i}: {j}")


# In[4]:


#Reading the data 
stroke_prediction_data = pd.read_csv("healthcare-dataset-stroke-data.csv")


# In[5]:


stroke_prediction_data.head()


# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Exploratory Data Information</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[6]:


stroke_prediction_data.info()


# In[7]:


stroke_prediction_data.shape


# In[8]:


stroke_prediction_data.describe()


# In[9]:


stroke_prediction_data.isna().sum()
# The "id" feature does not add value in the analysis because it is different for each instance.

# We have categorical features that will have to be coded for analysis.

# The characteristic "bmi" has null values.

# The data does not contain infinite values


# In[10]:


stroke_prediction_data.columns


# In[11]:


import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(stroke_prediction_data.corr(), annot=True, cmap='nipy_spectral_r')
plt.show()


# In[12]:


sns.pairplot(stroke_prediction_data)
print('We have data with a high level of imbalance')
print('95% of patients have not had a stroke')
print('Only 5% of patients in the entire dataset have had a stroke')


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
# Plot the distribution of Age, BMI, and Average Glucose Level in one plot
sns.histplot(data=stroke_prediction_data, x="age", kde=True, label="Age")
sns.histplot(data=stroke_prediction_data, x="bmi", kde=True, label="BMI",color='red')
sns.histplot(data=stroke_prediction_data, x="avg_glucose_level", kde=True, label="Average Glucose Level",color='lightgreen')
plt.legend()
plt.show()


# In[14]:


# plot a scatterplot of age versus BMI
sns.scatterplot(data=stroke_prediction_data, x="age", y="bmi", color='green')
plt.title('Age vs. BMI')
plt.show()


# In[15]:


# plot a scatterplot of age versus average glucose level
sns.scatterplot(data=stroke_prediction_data, x="age", y="avg_glucose_level", color='lavender')
plt.title('Age vs. avg_glucose_level')
plt.show()


# In[16]:


sns.FacetGrid(stroke_prediction_data, hue="stroke", height = 8).map(sns.distplot, "age").add_legend()
plt.title("Distplot for patients' age")
plt.show()
print('Generally, higher age increases the likely hood of a stroke')


# In[17]:


sns.FacetGrid(stroke_prediction_data, hue="stroke", height = 8,).map(sns.distplot, "bmi",).add_legend()
plt.title("Distplot for patients' bmi")
plt.show()
print('The bmi column by itself cannot be used to predict the likely hood of a stroke')


# In[18]:


plt.figure(figsize=(30,10))

plt.subplot(1,3,1)
plt.title("Strokes Shown by Age and Gender",fontsize=18)
sns.stripplot(x="stroke",y="age",hue="gender",data= stroke_prediction_data, palette='terrain')
plt.xlabel("Stroke",fontsize=18)
plt.ylabel("Age",fontsize=18)
plt.legend(loc=1, prop={'size': 11})

plt.subplot(1,3,2)
plt.title("Strokes Shown by Bmi and Gender",fontsize=18)
sns.stripplot(x="stroke",y="bmi",hue="gender",data= stroke_prediction_data,palette='pastel')
plt.xlabel("Stroke",fontsize=18)
plt.ylabel("Bmi",fontsize=18)
plt.legend(loc=1, prop={'size': 11})

plt.subplot(1,3,3)
plt.title("Strokes Shown by Average Glucose Level and Gender",fontsize=18)
sns.stripplot(x="stroke",y="avg_glucose_level",hue="gender", data = stroke_prediction_data,palette='terrain_r')
plt.xlabel("Stroke",fontsize=18)
plt.ylabel("Average Glucose Level",fontsize=18)
plt.legend(loc=1, prop={'size':11})

plt.show()


# In[19]:



fig, ax = plt.subplots(1,2, figsize = (14,5))
((ax1, ax2)) = ax

stroke_prediction_data[stroke_prediction_data['stroke'] ==0].plot.scatter(ax=ax1, x='bmi', y='avg_glucose_level', alpha = 0.2, c='gray', label='no stroke')
stroke_prediction_data[stroke_prediction_data['stroke'] ==1].plot.scatter(ax=ax1, x='bmi', y='avg_glucose_level', alpha = 0.8, c='orange', label='stroke')
ax1.legend()
ax1.set_title('stroke by combination of bmi and avg_glucose_level')

stroke_prediction_data[stroke_prediction_data['stroke'] ==0].plot.scatter(ax=ax2, x='bmi', y='age', alpha = 0.3, c='gray', label='no stroke')
stroke_prediction_data[stroke_prediction_data['stroke'] ==1].plot.scatter(ax=ax2, x='bmi', y='age', alpha = 0.6, c='orange', label='stroke')
ax2.legend()
ax2.set_title('stroke by combination of bmi and age')

plt.tight_layout()
plt.show()


# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Feature Engineering</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[20]:


print('The dataset is a mix of both categorical and numeric data and since Machine learning algorithms understand data of numeric nature')
# Check datatype of the column
cats = list(stroke_prediction_data.select_dtypes(include=['object','bool']) )
nums = list(stroke_prediction_data.select_dtypes(include=['int64','float64']))
print(cats)
print(nums)
    


# In[21]:


# classify data for the encoding
encoder = []
onehot = []

for col in cats:
    if len(stroke_prediction_data[col].unique()) == 2:
        encoder.append(col)
    else:
        onehot.append(col)

print(encoder)
print(onehot)


# In[22]:


stroke_prediction_data_labencoded = stroke_prediction_data.copy()

for col in encoder:
    stroke_prediction_data_labencoded[col] = stroke_prediction_data_labencoded[col].astype('category').cat.codes
stroke_prediction_data_labencoded.head()


# In[23]:


# check feature correlation to the target
stroke_prediction_data_labencoded.corr().round(2)


# In[24]:


# visualize feature correlation to the target
for col in onehot:
    stroke_prediction_data_loop = stroke_prediction_data_labencoded[[col,'stroke']].copy()
    onehots = pd.get_dummies(stroke_prediction_data_loop[col], prefix=col)
    stroke_prediction_data_loop = stroke_prediction_data_loop.join(onehots)
    plt.figure(figsize=(15, 8))
    print(sns.heatmap(stroke_prediction_data_loop.corr(), cmap='twilight_shifted_r', annot=True, fmt='.2f'))


# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Data Preparation</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[25]:


print('To Handle the null values we will be using KNN')
print('We get the value of neigh_params from the config dictionary,create KNNImputer object with the specified number of neighbors')
from sklearn.impute import KNNImputer

n_config = config.getint('preprocessing','neigh_params')

# Get the parameters from the config file
neigh_params = n_config
# = ['imputation']['neigh_params']

# Create a KNNImputer object with the specified parameters
imputer = KNNImputer(n_neighbors=neigh_params)

# Impute missing values in the 'bmi' column of the 'stroke_prediction_data' DataFrame
stroke_prediction_data['bmi'] = imputer.fit_transform(stroke_prediction_data[['bmi']])
# **The imputed bmi values are assigned back to the bmi column of the stroke_prediction_data DataFrame.**


# In[26]:


n_config


# In[27]:


stroke_prediction_data.isna().sum()


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Remove Unnecessary Columns</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[28]:


stroke_prediction_data= stroke_prediction_data.drop(['id'], axis=1)
nums.remove('id')
stroke_prediction_data.head()
# **We removed the "id" feature because it generates noise in the analysis**
    


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Remove Outliers of the dataset</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[29]:


#Lets See Outliers
plt.figure(figsize=(15, 7))
for i in range(0, len(nums)):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=stroke_prediction_data[nums[i]],color='green',orient='v')
    plt.tight_layout()
    


# In[30]:


outlier = ['avg_glucose_level', 'bmi']


# In[31]:


Q1 = stroke_prediction_data[outlier].quantile(0.25)
Q3 = stroke_prediction_data[outlier].quantile(0.75)
IQR = Q3 - Q1
stroke_prediction_data = stroke_prediction_data[~((stroke_prediction_data[outlier]<(Q1-1.5*IQR))|(stroke_prediction_data[outlier]>(Q3+1.5*IQR))).any(axis=1)]
stroke_prediction_data.reset_index(drop=True)
    


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Convert Categorical Column to Numerical</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[32]:


#onehot encoding for categorical feature
stroke_prediction_data = pd.get_dummies(stroke_prediction_data)
stroke_prediction_data.head()


# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Data Imbalance</h2>
# <hr style="border:1px solid black"> </hr>

# In[33]:


# Exploratory data analysis (EDA) is used to analyze and investigate datasets and summarize their main characteristics,
# often employing data visualization methods.
fig, axes = plt.subplots(figsize=(8,4))
stroke_prediction_data['stroke'].value_counts(normalize=True).plot.bar(width=0.2,color=('blue','green'))
plt.tight_layout()
plt.show()


# In[34]:


cols = stroke_prediction_data[['age','hypertension','heart_disease','avg_glucose_level','bmi']]
cols.head()


# In[35]:


stroke_prediction_data.columns


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Pre Modelling Steps</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[36]:


# separate feature and target
X = stroke_prediction_data.drop(columns = ['stroke'])
y = stroke_prediction_data['stroke']
    


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Split train data and test data</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[37]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[38]:


#### **Balancing Dataset**
# As we know, dataset is imbalanced. So let’s balance our data. I am going to use SMOTE method for this. It will populate our data with records similar to our minor class. Usually, we perform this on the whole dataset but as we have very fewer records of minor class.


# In[39]:


# Apply SMOTE to the training set only
sm = SMOTE(random_state=111)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f'''Shape of X before SMOTE:{X.shape}
Shape of X after SMOTE:{X_train_sm.shape}''',"\n\n")

print(f'''Target Class distributuion before SMOTE:\n{y.value_counts(normalize=True)}
Target Class distributuion after SMOTE :\n{y_train_sm.value_counts(normalize=True)}''')

# Concatenate the training and testing sets
X_balanced = pd.concat([X_train_sm, X_test])
y_balanced = pd.concat([y_train_sm, y_test])

# Create a new dataframe with the balanced target variable
data_balanced = pd.DataFrame({'stroke': y_balanced})

# Plot the countplot
sns.countplot(x='stroke', data=data_balanced)
plt.title('Class Distribution of Stroke Data (after balancing)')
plt.xlabel('Stroke (0 = no stroke, 1 = stroke)')
plt.ylabel('Count')
plt.show()


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Normalize data with StandardScaler</h2>
# <hr style="border:1px solid black"> </hr>
# 
# 

# In[40]:


scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_balanced)
X_test_scaled = scaler.transform(X_test)


# In[41]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Normalize the features in X_balanced
X_balanced_norm = scaler.fit_transform(X_balanced)

# Convert X_balanced_norm back to a pandas DataFrame
X_balanced_norm = pd.DataFrame(X_balanced_norm, columns=X_balanced.columns)

# Print the mean and standard deviation of the normalized features
print(X_balanced_norm.mean())
print(X_balanced_norm.std())


# In[42]:


from sklearn.decomposition import PCA
pca_check = PCA()
pca_check.fit(X_train_scaled)
pca_components = range(pca_check.n_components_)
plt.bar(pca_components, pca_check.explained_variance_)
plt.xlabel('dimensions')
plt.ylabel('variance')
plt.show()


#4.6B variance explained by each principal component, along with the corresponding dimension number using the sum of explained_variance_

variance_sum = sum(pca_check.explained_variance_)
combined_variance = 0
for i, component in enumerate(pca_check.explained_variance_):
    combined_variance += (component/variance_sum)*100
    print(f'{round(combined_variance)}% variance is explained by {i+1} dimensions')


# In[43]:


from sklearn.decomposition import PCA

# Create a PCA object with 2 components
pca = PCA(n_components=14)

# Fit the PCA object to the normalized features in X_balanced_norm
X_train_pca = pca.fit(X_balanced_norm)

# Transform the features in X_balanced_norm using the fitted PCA object
X_balanced_pca = pca.transform(X_balanced_norm)

# Create a scatter plot of the transformed features
plt.scatter(X_balanced_pca[:,0], X_balanced_pca[:,1], c=y_balanced)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA plot of Stroke Data (after balancing)')
plt.show()


# In[44]:


from sklearn.decomposition import PCA
import numpy as np

# Reshape input data to 2D array
X_train_2d = np.reshape(X_train, (-1, X_train.shape[1]))

# Apply PCA to the reshaped data
pca = PCA(n_components=14)
X_train_pca = pca.fit_transform(X_train_2d)

# Reshape test data to 2D array and apply PCA
X_test_2d = np.reshape(X_test, (-1, X_test.shape[1]))
X_test_pca = pca.transform(X_test_2d)


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:TWILIGHT;">Machine Learning Modeling</h2>
# <hr style="border:1px solid black"> </hr>
# 

# <h2 style="color:crimson;">Using K-Nearest-Neighbors-Classifier</h2>
# <hr style="border:1px solid black"> </hr>

# In[45]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score



from configparser import ConfigParser
#Load Config File 
config = ConfigParser()
config.read('configfile.ini')

# Define pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
#     ('pca', PCA(n_components=14)),
    ('knn', KNeighborsClassifier())
])

# Extract model parameters from config file
n_neighbors = int(config['model_params']['n_neighbors'])
weights = config['model_params']['weights']
metric = config['model_params']['metric']

# Set pipeline parameters
pipe.set_params(
    knn__n_neighbors=n_neighbors,
    knn__weights=weights,
    knn__metric=metric
)

# Extract cross-validation parameters from config file
cv = int(config['cv_params']['cv'])
scoring = config['cv_params']['scoring']

# Perform cross-validation on the training data
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
cv_mean = cv_scores.mean()
print("Cross-validation mean accuracy:", cv_mean)

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Test the pipeline on the testing data
y_pred = pipe.predict(X_test)

# Print the accuracy score and classification report
acc = accuracy_score(y_test, y_pred)
print('Testing-set Accuracy score is:', acc)
print('Training-set Accuracy score is:', accuracy_score(y_train, pipe.predict(X_train)))
print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))

# Plot the confusion matrix
cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot=True, fmt="d")


# In[46]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the predicted probabilities for the test set
y_pred_proba = pipe.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[47]:


import matplotlib.pyplot as plt

# Calculate cross-validation accuracy scores
cv_scores = cross_val_score(pipe,X_train_pca, y_train, cv=cv, scoring=scoring)

# Plot cross-validation results
plt.plot(cv_scores)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation results')
plt.show()


# In[48]:


from sklearn.model_selection import cross_val_score
# Perform cross-validation on the training data
cv_results = cross_val_score(pipe, X_train_pca, y_train, cv=cv, scoring=scoring)

# Plot the results using a boxplot
import matplotlib.pyplot as plt
plt.boxplot(cv_results)
plt.title('Cross-validation Results')
plt.xlabel('Model')
plt.ylabel('Recall')
plt.show()


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;"> Develop the Machine Learning model (KNN CLASSIFIER) with Hyperparameter Tuning HalvingGridSearchCV</h2>
# <hr style="border:1px solid black"> </hr>
# 

# In[49]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from configparser import ConfigParser
#Load Config File 
config = ConfigParser()
config.read('configfile.ini')

# Define the pipeline
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Get the hyperparameters from the config file
n_neighbors = config.getint('hyperparameters', 'n_neighbors')
p = config.getint('hyperparameters', 'p')
weights = config.get('hyperparameters', 'weights')
algorithm = config.get('hyperparameters', 'algorithm')

# Set hyperparameter
param_grid = {'knn__n_neighbors': [n_neighbors],
            'knn__p': [p],
            'knn__weights': [weights],
            'knn__algorithm': [algorithm],
            }

# Define cross-validation splitter
cv_splitter = StratifiedKFold(n_splits=3, random_state=123, shuffle=True)

# Seek for the best hyperparameter with HalvingGridSearchCV
new_param = HalvingGridSearchCV(model_pipeline, 
                                param_grid, 
                                cv=cv_splitter,
                                resource='knn__leaf_size',
                                max_resources=20,
                                scoring='recall',
                                aggressive_elimination=False).fit(X_train, y_train)

# Result of the hyperparameter tuning
print(f"Best Hyperparameter {new_param.best_estimator_} with score {new_param.best_score_}")

# Set the model with the best hyperparameter
model = KNeighborsClassifier(algorithm=algorithm, leaf_size=new_param.best_params_['knn__leaf_size'], 
                            n_neighbors=n_neighbors, p=p, weights=weights)

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])


# Fit the model pipeline to the training data
model_pipeline.fit(X_train, y_train)

# Test tuned model with test data
y_pred = model_pipeline.predict(X_test)

# Tuned model report
acc = accuracy_score(y_test, y_pred)
print('Testing-set Accuracy score is:', acc)
print('Training-set Accuracy score is:', accuracy_score(y_train, model_pipeline.predict(X_train)))

improvement_report = classification_report(y_test, y_pred, output_dict=True, target_names=['No Stroke', 'Stroke'])
pd.DataFrame(improvement_report).transpose()

# generate the classification report
report = classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke'])
print(report)
# Confusion matrix
hyper_cf = confusion_matrix(y_test, y_pred)
sns.heatmap(hyper_cf, annot=True, fmt="d")


# In[50]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get the probabilities for each class
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calculate the FPR, TPR, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate the AUC score
auc_score = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc_score))
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[51]:


from sklearn.model_selection import cross_val_score
# Perform cross-validation on the training data
results = []
model_names = []
cv_results = cross_val_score(model_pipeline, X_train, y_train, cv=cv_splitter, scoring='recall')

# Plot the results using a boxplot
import matplotlib.pyplot as plt
plt.boxplot(cv_results)
plt.plot(cv_results)
plt.title('Cross-validation Results')
plt.xlabel('Model')
plt.ylabel('Recall')
plt.show()
#Boxplot showing the distribution of recall scores across the different folds of the cross-validation.
#This will give us an idea of how well the model generalizes to new data.


# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Logistic Regression</h2>
# <hr style="border:1px solid black"> </hr>

# In[52]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from configparser import ConfigParser

# Load Config File 
config = ConfigParser()
config.read('configfile.ini')

# Create pipeline
logpipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit the pipeline to training data
logpipeline.fit(X_train, y_train)

# Make predictions on test data
y_pred = logpipeline.predict(X_test)

# Perform cross-validation on the training data
cv_params = config['CrossValidation']
cv = int(cv_params.get('cv'))
scoring = cv_params.get('scoring')
lr_cv_scores = cross_val_score(logpipeline, X_train, y_train, cv=cv, scoring=scoring)
mean_cv_score = lr_cv_scores.mean()

# Simple model report
acc = accuracy_score(y_test, y_pred)
print('Testing-set Accuracy score is:', acc)
print('Training-set Accuracy score is:', accuracy_score(y_train, logpipeline.predict(X_train)))

# Generate classification report
target_names = config['ClassificationReport']['target_names'].split(',')
report = classification_report(y_test, y_pred, output_dict=True, target_names=target_names)
report_df = pd.DataFrame(report).transpose()

# Print classification report
print(report_df)

# Generate confusion matrix
log_reg_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(log_reg_cm, annot=True, fmt="d")


# In[53]:


from sklearn.metrics import roc_curve, roc_auc_score

# Calculate the predicted probabilities
y_pred_prob = logpipeline.predict_proba(X_test)[:,1]

# Calculate the FPR and TPR for different threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[54]:


from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Calculate cross-validation accuracy scores
cv_scores = cross_val_score(logpipeline, X_train, y_train, cv=cv, scoring=scoring)

# Plot cross-validation results
plt.plot(cv_scores)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation results')
plt.show()


# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;"> Develop the Machine Learning model (logistic regression) with Hyperparameter Tuning HalvingGridSearchCV</h2>
# <hr style="border:1px solid black"> </hr>

# In[55]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

#Load Config File 
config = ConfigParser()
config.read('configfile.ini')

# Set hyperparameters
n_components = int(config.get('PCA', 'n_components'))

# Create pipeline
hyplogpipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components)),
    ('classifier', LogisticRegression())
])

# Define the range of hyperparameters to tune
logreg_params = config['LogisticRegressiontuned']
penalty = [logreg_params.get('penalty')]
C = [float(c) for c in logreg_params.get('C').split(',')]
param_grid = {'classifier__penalty': penalty, 'classifier__C': C}

# Define the cross-validation object
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the grid search object
grid_search = GridSearchCV(estimator=hyplogpipeline, param_grid=param_grid, cv=cv)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by GridSearchCV
print('Best hyperparameters:', grid_search.best_params_)

# Create a new logistic regression model with the best hyperparameters found by GridSearchCV
log_reg = grid_search.best_estimator_

# Fit the model to the training data
log_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg.predict(X_test)

# Tuned model report
acc = accuracy_score(y_test, y_pred)
print('Testing-set Accuracy score is:', acc)
print('Training-set Accuracy score is:',accuracy_score(y_train,log_reg.predict(X_train)))

improvement_report = classification_report(y_test, y_pred, output_dict=True, target_names=['No Stroke', 'Stroke'])
pd.DataFrame(improvement_report).transpose()

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke'])
print(report)

# Generate confusion matrix
log_reg_cm = confusion_matrix(y_test, y_pred)
sns.heatmap(log_reg_cm, annot=True, fmt="d")


# In[65]:


from sklearn.metrics import roc_curve, roc_auc_score

# Calculate the probabilities of class 1
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Calculate the FPR, TPR, and threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate the AUC score
roc_auc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[57]:


# Compute the cross-validation scores
cv_params = config['CrossValidation']
cv = int(cv_params.get('cv'))
scoring = cv_params.get('scoring')
cv_scores = cross_val_score(hyplogpipeline, X_train, y_train, cv=cv, scoring=scoring)



# Plot the results
fig, ax = plt.subplots()
ax.plot(range(1, cv+1), cv_scores)
ax.set_xlabel('Fold')
ax.set_ylabel('Score')
ax.set_title('Cross-validation Results')
plt.show()


# In[58]:


import numpy as np
import matplotlib.pyplot as plt

# get the results of the grid search
results = grid_search.cv_results_

# extract the mean test score and standard deviation for each combination of hyperparameters
mean_scores = results['mean_test_score']
std_scores = results['std_test_score']
params = results['params']

# convert the hyperparameters into a string format for plotting
param_strings = [str(p) for p in params]

# create a horizontal bar plot of the mean test score and standard deviation for each combination of hyperparameters
y_pos = np.arange(len(param_strings))
plt.barh(y_pos, mean_scores, xerr=std_scores, align='center')
plt.yticks(y_pos, param_strings)
plt.xlabel('Mean test score')
plt.ylabel('Hyperparameters')
plt.title('Cross-validation results')
plt.show()


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Deep Learning Model (ANN) with RandomsearchCV </h2>
# <hr style="border:1px solid black"> </hr>

# In[59]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from scipy import stats

# Define the hyperparameters
Ep_deep = config.getint('Deep','epochs')
batc_deep = config.getint('Deep', 'batch_size')
opt_deep = [config.get('Deep','optimizer')]
drop_deep = [float(c) for c in config.get('Deep', 'dropout_rate').split(',')]
act_deep = [config.get('Deep','activation')]
iter_deep = config.getint('Deep','n_iter')
cv_deep = config.getint('Deep','cv')

# Define the Keras model
def create_ann_model(dropout_rate=0.0, activation='relu', optimizer='adam', hidden_layers=1, neurons=64):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(X_train_pca.shape[1],), activation=activation))
    model.add(Dropout(dropout_rate))
    for i in range(hidden_layers):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define the ANN model as a KerasClassifier object
model = KerasClassifier(build_fn=create_ann_model, verbose=0)

# Define the randomized search object
param_dist = {
    'dropout_rate': [0.1, 0.2, 0.3],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'hidden_layers': [1, 2, 3],
    'neurons': [32, 64, 128, 256]
}
search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=5,
    verbose=2, 
    error_score='raise'
)

# Fit the randomized search object to the data
search.fit(X_train_pca, y_train, validation_split=0.1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

# Predict classes for the test set
y_pred = search.predict(X_test_pca)
accc = accuracy_score(y_test, y_pred)
print("ACCu:", accc)
msee = mean_squared_error(y_test, y_pred)
print("Mean squared error:", msee)
r2c = r2_score(y_test, y_pred)
print("R2 score:", r2c)

#Define the y_test_classes variable as a binary vector to compute the confusion matrix correctly
y_test_classes = label_binarize(y_test, classes=[0, 1])[:, 0]
# Compute the confusion matrix
cmc = confusion_matrix(y_test_classes, y_pred)
print("Confusion matrix:\n",cmc)

# Plot the confusion matrix
plt.figure()
plt.imshow(cmc, interpolation='nearest', cmap=plt.cm.Blues)
plt.show()


# In[60]:


from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, roc_curve, auc


# Predict classes for the test set
y_pred = search.predict(X_test_pca)

# Compute the false positive rate, true positive rate, and threshold for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_classes, y_pred)

# Compute the area under the ROC curve
roc_auc = auc(fpr, tpr)

# Print the AUC score
print("ROC AUC score:", roc_auc)

# Plot the ROC curve
plt.figure(figsize=(8, 6)) # Set the figure size
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0]) # Set the x-axis limits
plt.ylim([0.0, 1.05]) # Set the y-axis limits
plt.xlabel('False Positive Rate', fontsize=14) # Set the x-axis label and font size
plt.ylabel('True Positive Rate', fontsize=14) # Set the y-axis label and font size
plt.title('Receiver Operating Characteristic', fontsize=16) # Set the plot title and font size
plt.tick_params(labelsize=12) # Set the tick label font size
plt.legend(loc="lower right", fontsize=14) # Set the legend font size
plt.show()


# In[61]:


import matplotlib.pyplot as plt

# Get the cross-validation results
cv_results = search.cv_results_

# Extract the mean test scores and standard deviations
mean_test_scores = cv_results['mean_test_score']
std_test_scores = cv_results['std_test_score']
params = cv_results['params']

# Extract the hyperparameter values that were tested
dropout_rates = [params[i]['dropout_rate'] for i in range(len(params))]
activations = [params[i]['activation'] for i in range(len(params))]

# Plot the results using an errorbar plot
fig, ax = plt.subplots()
ax.errorbar(range(len(mean_test_scores)), mean_test_scores, yerr=std_test_scores, fmt='o', color='blue', ecolor='gray', capsize=5)

# Set the x-axis ticks and labels
ax.set_xticks(range(len(mean_test_scores)))
ax.set_xticklabels([f"Dropout Rate: {dropout_rates[i]}, Activation: {activations[i]}" for i in range(len(mean_test_scores))], rotation=90)

# Set the plot title and axis labels
ax.set_title('Cross-validation Results')
ax.set_xlabel('Hyperparameters')
ax.set_ylabel('Accuracy')

plt.show()


# Plot with the mean test scores as dots and error bars showing the standard deviation. The x-axis shows the iteration number, and the y-axis shows the accuracy. 

# In[62]:


import matplotlib.pyplot as plt
import seaborn as sns

# # Get the cross-validation results
# cv_results = search.cv_results_

# Extract the mean and standard deviation of the test scores for each combination of hyperparameters
means = cv_results['mean_test_score']
stds = cv_results['std_test_score']
params = cv_results['params']

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(range(len(params)), means, yerr=stds, align='center')
plt.xticks(range(len(params)), [str(p) for p in params], rotation=90)
plt.xlabel('Hyperparameters')
plt.ylabel('Mean Test Score')
plt.title('Cross-Validation Results')
plt.show()


# 
# <hr style="border:1px solid black"> </hr>
# <h2 style="color:Crimson;">Evaluation of three Models</h2>
# <hr style="border:1px solid black"> </hr>

# In[63]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
# Scale the input features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Initialize the three classifiers with the desired hyperparameters
logistic_reg = LogisticRegression(C=1)
ann = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, alpha=0.01, solver='sgd', learning_rate='constant', learning_rate_init=0.01)
knn = KNeighborsClassifier(n_neighbors=5)

# Combine the three classifiers into a list
classifiers = [logistic_reg, ann, knn]

# Define a KFold cross-validation object with 5 folds
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation for each classifier and calculate mean and standard deviation of scores
scores = []
for clf in classifiers:
    cv_score = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    scores.append(cv_score)

means = [score.mean() for score in scores]
stds = [score.std() for score in scores]

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))
# Plot the results using a box plot
plt.boxplot(scores, labels=['Logistic Regression', 'ANN', 'KNN'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Cross-validation Results')
plt.show()


# In[64]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Split the data into training and testing sets

# Define the cross-validation technique and the scoring metric
cv = KFold(n_splits=10, shuffle=True, random_state=42)
scoring = 'accuracy'

# Define the three models
logreg = LogisticRegression()
ann = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000)
knn = KNeighborsClassifier()

# Fit each model on the training set and evaluate its performance using cross-validation
logreg_scores = cross_val_score(logreg, X_train, y_train, cv=cv, scoring=scoring)
ann_scores = cross_val_score(ann, X_train, y_train, cv=cv, scoring=scoring)
knn_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring=scoring)

# Compute the mean and standard deviation of the cross-validation scores for each model
logreg_mean = logreg_scores.mean()
logreg_std = logreg_scores.std()
ann_mean = ann_scores.mean()
ann_std = ann_scores.std()
knn_mean = knn_scores.mean()
knn_std = knn_scores.std()

# Compare the cross-validation scores of the three models to select the best performing model
print('Logistic Regression: {} +/- {}'.format(logreg_mean, logreg_std))
print('ANN: {} +/- {}'.format(ann_mean, ann_std))
print('KNN Classifier: {} +/- {}'.format(knn_mean, knn_std))


import matplotlib.pyplot as plt

# Create a bar plot of the mean cross-validation scores for each model
models = ['Logistic Regression', 'ANN', 'KNN Classifier']
means = [logreg_mean, ann_mean, knn_mean]
stds = [logreg_std, ann_std, knn_std]
fig, ax = plt.subplots()
ax.bar(models, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Accuracy')
ax.set_title('Cross-validation scores for different models')
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

models = ['Logistic Regression', 'ANN', 'KNN Classifier']
scores = [logreg_scores, ann_scores, knn_scores]
fig, ax = plt.subplots()
ax.boxplot(scores, labels=models, showmeans=True)
ax.set_ylabel('Accuracy')
ax.set_title('Cross-validation scores for different models')
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()


# 

# %%
