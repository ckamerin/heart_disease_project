
###Predicting heart disease using machine learning

#   This notebook will look into various libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes

# We're going to take the following approach
    # 1. Problem definition
    # 2. Data
    # 3. Evaluation
    # 4. Features
    # 5. Modelling
    # 6. Experimentation

## 1. Problem Definition

#   In a statment, given someones medical attributes can we predict whether or not they have heart disease.

## 2. Data

#   The original data came from the Cleavland data from the UCI Machine Learning Repository. 
#   https://archive.ics.uci.edu/m1/datasets/heart+Disease

## 3. Evaluation

#   If we can reach 95% accuracy at predicting whether or not someone has heart disease during the proof of concept, we'll pusure the project.

## 4. Features

#   Data Dictionary

# age - age in years
# sex - (1 = male; 0 = female)
# cp - chest pain type
# 0: Typical angina: chest pain related decrease blood supply to the heart
# 1: Atypical angina: chest pain not related to heart
# 2: Non-anginal pain: typically esophageal spasms (non heart related)
# 3: Asymptomatic: chest pain not showing signs of disease
# trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# anything above 130-140 is typically cause for concern
# chol - serum cholestoral in mg/dl
# serum = LDL + HDL + .2 * triglycerides
# above 200 is cause for concern
# fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# '>126' mg/dL signals diabetes
# restecg - resting electrocardiographic results
# 0: Nothing to note
# 1: ST-T Wave abnormality
# can range from mild symptoms to severe problems
# signals non-normal heart beat
# 2: Possible or definite left ventricular hypertrophy
# Enlarged heart's main pumping chamber
# thalach - maximum heart rate achieved
# exang - exercise induced angina (1 = yes; 0 = no)
# oldpeak - ST depression induced by exercise relative to rest
# looks at stress of heart during excercise
# unhealthy heart will stress more
# slope - the slope of the peak exercise ST segment
# 0: Upsloping: better heart rate with excercise (uncommon)
# 1: Flatsloping: minimal change (typical healthy heart)
# 2: Downslopins: signs of unhealthy heart
# ca - number of major vessels (0-3) colored by flourosopy
# colored vessel means the doctor can see the blood passing through
# the more blood movement the better (no clots)
# thal - thalium stress result
# 1,3: normal
# 6: fixed defect: used to be defect but ok now
# 7: reversable defect: no proper blood movement when excercising
# target - have disease or not (1=yes, 0=no) (= the predicted attribute)

## Preparing the tools

# Regular EDA and plotting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, plot_roc_curve


#Load Data

df = pd.read_csv("heart-disease.csv").reset_index(drop=True)
df.shape

## EDA Checklist
#   1. What question(s) are we trying to solve?
#   2. What kind of data do we have and how do we treat different types?
#   3. What's missing from the data and how do we deal with that?
#   4. Where are the outliers and why/should we care about them?
#   5. How can you add, change or remove features to get more out of our data?

df.head()
df.tail()

# %%
df["target"].value_counts()

df["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"])

df.info()

df.isna().sum()

df.describe()

df.sex.value_counts()

pd.crosstab(df.target, df.sex)

#women have roughly 75% of having heart disease, men have roughly 50% chance based on data set
# %%
pd.crosstab(df.target, df.sex).plot(kind="bar",
                                    figsize=(10,6),
                                    color=["salmon","lightblue"])

plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"])


df["thalach"].value_counts()


# %%
plt.figure(figsize=(10,6))

plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon")

plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue")

plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease","No Disease"])


# %%
# Check the distribution of the age column with a histogram
df.age.plot.hist();

# %%
### Heart Disease Frequency per Chest Pain type
# cp - chest pain type
# 0: Typical angina: chest pain related decrease blood supply to the heart
# 1: Atypical angina: chest pain not related to heart
# 2: Non-anginal pain: typically esophageal spasms (non heart related)
# 3: Asymptomatic: chest pain not showing signs of disease

pd.crosstab(df.cp, df.target)
pd.crosstab(df.cp, df.target).plot( kind="bar",
                                    figsize=(10,6),
                                    color=["salmon","lightblue"])
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease","Disease"])
plt.xticks(rotation=0);

# %%
print(df.corr().target)

corr_matrix=df.corr()
fig, ax=plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=1)
                 

plt.yticks(rotation=90);

# 5. Modelling


np.random.seed(5)
df1=df.sample(frac=1)

X=df1.drop("target", axis=1)

y=df1["target"]



train, validate, test = np.split(df1, [int(.7 *len(df1)),int(.85*len(df1))])

X_train, X_val, X_test = train.drop("target", axis=1), validate.drop("target", axis=1), test.drop("target", axis=1)
y_train, y_val, y_test = train["target"], validate["target"], test["target"]


## Going to try 3 different ML models
#   1. Logistic Regression
#   2. KNeighbors Classifier
#   3. Random Forest Classifier

models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forrest": RandomForestClassifier()}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given ML models.
    models: a dict of different SKlearn ML models
    X_train : training data (no labels)
    X_val : validation data (no labels)
    X_test : test data (no labels)
    y_train : training labels
    y_val : validation labels
    y_test : test labels
    """
    #Set Random Seed
    np.random.seed(5)
    #Make Dict for model scores
    model_scores = {}
    #Loop through models
    for name, model in models.items():
        #Fit the model to data
        model.fit(X_train,y_train)
        #Evaluate the model and append scores 
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# %%
model_scores = fit_and_score(models=models,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test)

model_scores


# %%
model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();

# We shouldnt base out model off baseline predicitons time to check other classifications of accuracy.

# Look at the following:
# * Hyperparameter tuning
# * Feature imporatnce
# * Confusion matrix
# * Cross-validation
# * Precision 
# * Recall 
# * F1 score 
# * Classification report 
# * ROC Curve
# * Area under the curve (AUC)

#%% 
train_scores = []
test_scores = []

#List of different n_neighbors values
neighbors = range(1,21)

knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params( n_neighbors=i)

    #Fit the algorithm
    knn.fit(X_train, y_train)

    #Update the score lists
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

train_scores

# %%
test_scores  

# %%
plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="=Test score")
plt.xticks(np.arange(1,21,1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


## Hyper Parameter Tuning with RandomSearchCV

# We're going to tune LogisticRegression() RandomForestClassifer() using RandomizedSearchCV

#Creat hyperparameter grid for LogisticRegression and RandomForestClassifer

log_reg_grid = {"C" : np.logspace(-4,4,20),
                "solver" : ["liblinear"]}

rf_grid ={"n_estimators": np.arange(10,1000,50),
          "max_depth": [None, 3, 5, 10],
          "min_samples_split": np.arange(2,20,2),
          "min_samples_leaf": np.arange(1,20,2)}



# %%
# Tune using RandomizedSearchCv

np.random.seed(5)

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)
rs_log_reg.fit(X_train,y_train)


# %%
rs_log_reg.best_params_

# %%
rs_log_reg.score(X_test, y_test)


# %%
np.random.seed(5)

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

rs_rf.fit(X_train,y_train)
# %%
rs_rf.best_params_

# %%
rs_rf.score(X_test,y_test)

# LogisticRegression still the best
# Use GridSearchCV to figure tune that.

#%%
log_reg_grid = {"C" : np.logspace(-4,4,30),
                "solver" : ["liblinear"]}

gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

gs_log_reg.fit(X_train, y_train);

# %%
gs_log_reg.best_params_

# %%
gs_log_reg.score(X_test,y_test)
## 5.Evaluation
## Evaluating our tuned ML clasifer beyond accuracy
 
# %%
y_preds = gs_log_reg.predict(X_test)


# %%
plot_roc_curve(gs_log_reg, X_test,y_test)

# %%
 print(confusion_matrix(y_test,y_preds))


# %%
def plot_conf_mat(y_test, y_preds):

    fig, ax = plt.subplots()
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                    annot=True,
                    cbar=False)
plt.xlabel("True Label")
plt.ylabel("Predicted Label")

# %%
plot_conf_mat(y_test, y_preds)

# %%
print(classification_report(y_test,y_preds))

# %%
gs_log_reg.best_params_

# %%
clf = LogisticRegression(C=62.10169418915616,
                        solver="liblinear")


# %%
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")

cv_prec = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")

cv_rec = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="recall")

cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")

# %%
cv_acc = np.mean(cv_acc)
cv_prec = np.mean(cv_prec)
cv_rec = np.mean(cv_rec)
cv_f1 = np.mean(cv_f1)

# %%
cv_metrics = pd.DataFrame({"Accuracy" : cv_acc,
                           "Precision" : cv_prec,
                           "Recall" : cv_rec,
                           "F1" : cv_f1},
                           index = [0])

cv_metrics.T.plot.bar(title= "Cross-validated classification metrics", legend = False)

# Feature Importance

# %%
clf = LogisticRegression(C=62.10169418915616,
                        solver="liblinear")

clf.fit(X_train,y_train);

# %%
clf.coef_

# %%
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict

# %%
feature_df= pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False)

# %%
##6. Experimentation
# * Could we collect more data?
# * Could we try a better model? Like CatBoost or XGBoost
# * Could we improve the current models? 
# * If your model is good enough and hit the evaluation metric, export and pass it on. 
