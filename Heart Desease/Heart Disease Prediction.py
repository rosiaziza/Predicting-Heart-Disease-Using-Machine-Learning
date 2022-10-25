#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Prediction
# Ainiya Aziza

# ### **Introduction**
# 
# The goal of this final project is to predict a person whether or not someone has heart disease based on their medical attributes. The original data came from the Cleveland database from UCI Machine Learning Repository and consists of a number of features including the information about health indicators. The original database contains 76 attributes, but here only 14 attributes will be used.
# 
# **Let's start by importing some necessary libraries**

# In[1]:


## Data Analysis
import numpy as np 
import pandas as pd 

## Data Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

## Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# **Here we load the dataset as follows**

# In[2]:


# Let's check the data dimension
df = pd.read_csv("../data/heart-disease.csv")
df.shape


# In[3]:


df = df.rename(columns={"sex": "gender"})


# **Exploratory Data Analysis**

# In[4]:


# Let's check the top 5 rows of our dataframe
df.head(5)


# In[5]:


# Let's see how many positive (1) and negative (0) samples we have in our dataframe
df.target.value_counts()


# In[6]:


# Normalized value counts
df.target.value_counts(normalize=True)


# In[7]:


# Plot the value counts with a bar graph
fig = plt.figure(figsize = (4,4))
sns.set_style("whitegrid")
plt.title("Histogram of Target")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
df.target.value_counts().plot(kind="bar", color=["lightsteelblue", "pink"]);
plt.xticks(rotation=0); # keep the labels on the x-axis vertical


# In[8]:


df.describe()


# **Heart Disease Frequency according to Gender**

# In[9]:


# Compare target column with sex column
pd.crosstab(df.target, df.gender)


# In[10]:


# Create a plot
pd.crosstab(df.target, df.gender).plot(kind="bar", figsize=(4,4), color=["pink", "lightsteelblue"])

# Add some attributes to it
plt.title("Heart Disease Frequency for Gender")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0);


# **Age vs Max Heart rate for Heart Disease**

# In[11]:


# Create another figure
plt.figure(figsize=(8,6))

# Start with positve examples
plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1], 
            c="lightsteelblue") # define it as a scatter figure

# Now for negative examples, we want them on the same plot, so we call plt again
plt.scatter(df.age[df.target==0], 
            df.thalach[df.target==0], 
            c="pink") # axis always come as (x, y)

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.legend(["Disease", "No Disease"])
plt.ylabel("Max Heart Rate");


# In[12]:


# Check the distribution of the age column with a histogram
df.age.plot.hist(color = "lightsteelblue");
plt.xlabel("Age");


# **Heart Disease Frequency per Chest Pain Type**

# In[13]:


pd.crosstab(df.cp, df.target)


# In[14]:


# Create a new crosstab and base plot
pd.crosstab(df.cp, df.target).plot(kind="bar", 
                                   figsize=(6,4), 
                                   color=["lightsteelblue", "pink"])

# Add attributes to the plot to make it more readable

plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation = 0);


# **Correlation between independent variables**

# In[15]:


corr_matrix = df.corr()
corr_matrix 


# In[16]:


corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="Blues");


# **Modelling**

# In[17]:


df.head(3)


# In[18]:


# Everything except target variable
x = df.drop("target", axis=1)

# Target variable
y = df['target']


# In[19]:


# Independent variables (no target column)
x.head(3)


# **Training and test split**

# In[20]:


# Split data into train and test sets
np.random.seed(42)

# Split into train & test set
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2)


# **Model choices**
# 1. Logistic Regression - [`LogisticRegression()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# 2. K-Nearest Neighbors - [`KNeighboursClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
# 3. RandomForest - [`RandomForestClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

# In[21]:


# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(solver = "lbfgs"), 
          "Random Forest": RandomForestClassifier()}

# Create function to fit and score models
def fit_and_score(models, x_train, x_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(x_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(x_test, y_test)
    return model_scores


# In[22]:


model_scores = fit_and_score(models=models,
                             x_train=x_train,
                             x_test=x_test,
                             y_train=y_train,
                             y_test=y_test)


# In[23]:


plt.figure(figsize=(4,4))
model_compare = pd.DataFrame(model_scores, index=["accuracy"]);
model_compare.T.plot.bar(color = "lightsteelblue");
plt.xticks(rotation = 0);


# **Hyperparameter Tuning**

# In[24]:


# Let's tune KNN

train_scores = []
test_scores = []

# Create a list of differnt values for n_neighbors
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm
    knn.fit(x_train, y_train)
    
    # Update the training scores list
    train_scores.append(knn.score(x_train, y_train))
    
    # Update the test scores list
    test_scores.append(knn.score(x_test, y_test))


# In[25]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# **Hypreparameter Tuning with `RandomizedSearchCV`**

# In[26]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# In[27]:


# Tune LogisticRegression

np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(x_train, y_train)


# In[28]:


rs_log_reg.best_params_


# In[29]:


rs_log_reg.score(x_test, y_test)


# In[30]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(x_train, y_train)


# In[31]:


# Find the best hyperparameters
rs_rf.best_params_


# In[32]:


# Evaluate the randomized search RandomForestClassifier model
rs_rf.score(x_test, y_test)


# **Hyperparameter Tuning with GridSearchCV**

# In[33]:


# Different hyperparameters for our LogisticRegression model
log_reg_grid = {"C": np.logspace(-4, 4, 30),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(x_train, y_train);


# In[34]:


# Check the best hyperparmaters
gs_log_reg.best_params_


# In[35]:


# Evaluate the grid search LogisticRegression model
gs_log_reg.score(x_test, y_test)


# **Evaluating our tuned machine learning classifier**
# 
# * ROC curve and AUC score
# * Confusion matrix
# * Classification report
# * Precision
# * Recall
# * F1-score

# In[36]:


# Make predictions with tuned model
y_preds = gs_log_reg.predict(x_test)


# In[37]:


# Plot ROC curve and calculate and calculate AUC metric
plot_roc_curve(gs_log_reg, x_test, y_test)


# In[38]:


# Confusion matrix
print(confusion_matrix(y_test, y_preds))


# In[39]:


sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
plot_conf_mat(y_test, y_preds)


# In[40]:


print(classification_report(y_test, y_preds))


# **Calculate evaluation metrics using cross-validation**

# In[41]:


# Check best hyperparameters
gs_log_reg.best_params_


# In[42]:


# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")


# In[43]:


# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="accuracy")


# In[44]:


cv_acc = np.mean(cv_acc)


# In[45]:


# Cross-validated precision
cv_precision = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision=np.mean(cv_precision)
cv_precision


# In[46]:


# Cross-validated recall
cv_recall = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# In[47]:


# Cross-validated f1-score
cv_f1 = cross_val_score(clf,
                         x,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)


# In[48]:


# Visualize cross-validated metrics
plt.figure(figsize=(1,1))
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1},
                          index=[0])

cv_metrics.T.plot.bar(title = "Cross-validated classification metrics", 
                      legend=False, 
                      color = "lightsteelblue"
                     );
plt.xticks(rotation = 0);


# In[49]:


# Fit an instance of LogisticRegression
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

clf.fit(x_train, y_train);


# In[50]:


# Check coef_
clf.coef_


# In[51]:


# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[52]:


# Visualize feature importance

feature_df = pd.DataFrame(feature_dict, index=[0])                                             
feature_df.T.plot.bar(title="Feature Importance", legend=False, color = "lightsteelblue");
plt.xticks(rotation = (48));


# In[53]:


pd.crosstab(df["gender"], df["target"])


# In[54]:


pd.crosstab(df["slope"], df["target"])


# slope - the slope of the peak exercise ST segment
# * 0: Upsloping: better heart rate with excercise (uncommon)
# * 1: Flatsloping: minimal change (typical healthy heart)
# * 2: Downslopins: signs of unhealthy heart
