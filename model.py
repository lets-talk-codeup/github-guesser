import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# -------------------------- the shotgun -------------------------- #

def nlp_shotgun(X_insample, y_insample, X_outsample, y_outsample):
    """ Take in Pandas Series for NLP content and target (pass columns!),
        Create several DecisionTree, RandomForest, LogisticRegression, Naive Bayes,
        and KNearest classification models, 
        Push model predictions to originating dataframe, return dataframe """
    # convert predictions column (usually Series) to dataframe
    if type(y_insample) != 'pandas.core.frame.DataFrame':
        y_insample = pd.DataFrame(y_insample.rename('in_actuals'))
    if type(y_outsample) != 'pandas.core.frame.DataFrame':
        y_outsample = pd.DataFrame(y_outsample.rename('out_actuals'))
    # Baseline
    y_insample, y_outsample = nlp_bl(y_insample, y_outsample)
    # Decision Tree classifier
    y_insample, y_outsample = decisiontree(X_insample, y_insample, X_outsample, y_outsample)
    # Random Forest classifier
    y_insample, y_outsample = randomforest(X_insample, y_insample, X_outsample, y_outsample)
    # Logistic Regression classifier
    y_insample, y_outsample = logisticregression(X_insample, y_insample, X_outsample, y_outsample)
    # Naive Bayes classifier
    y_insample, y_outsample = naivebayes(X_insample, y_insample, X_outsample, y_outsample)
    # K-Nearest Neighbors classifier
    y_insample, y_outsample = knearestneighbors(X_insample, y_insample, X_outsample, y_outsample)
    
    return y_insample, y_outsample

# -------------------------- the models -------------------------- #

def nlp_bl(y_insample, y_outsample):
    mode = y_insample.in_actuals.mode().item()
    y_insample['baseline'] = mode
    y_outsample['baseline'] = mode
    return y_insample, y_outsample

def decisiontree(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates decision trees with max_depth 1,2,3,5,10 and random_state=123 """
    X_cv_insample, X_cv_outsample = count_vectorizer(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = tfidf(X_insample, X_outsample)
    max_depths = [1,2,3,5,10]
    # loop through max depths
    for depth in max_depths:
        # create tree
        tree_cv = DecisionTreeClassifier(max_depth=depth, random_state=123)\
            .fit(X_cv_insample, y_insample.in_actuals)
        tree_tfidf = DecisionTreeClassifier(max_depth=depth, random_state=123)\
            .fit(X_tfidf_insample, y_insample.in_actuals)
        # make predictions in new column
        y_insample['cv_tree_maxdepth' + str(depth)] = tree_cv.predict(X_cv_insample)
        y_outsample['cv_tree_maxdepth' + str(depth)] = tree_cv.predict(X_cv_outsample)
        y_insample['tfidf_tree_maxdepth' + str(depth)] = tree_tfidf.predict(X_tfidf_insample)
        y_outsample['tfidf_tree_maxdepth' + str(depth)] = tree_tfidf.predict(X_tfidf_outsample)

    return y_insample, y_outsample # return dataframe with preds appended

def randomforest(X_insample, y_insample, X_outsample, y_outsample):
    X_cv_insample, X_cv_outsample = count_vectorizer(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = tfidf(X_insample, X_outsample)
    max_depths = [1,2,3,5,10]
    # loop through max depths
    for depth in max_depths:
        # create forest
        rf_cv = RandomForestClassifier(max_depth=depth, random_state=123)\
            .fit(X_cv_insample, y_insample.in_actuals)
        rf_tfidf = RandomForestClassifier(max_depth=depth, random_state=123)\
            .fit(X_tfidf_insample, y_insample.in_actuals)
        # make predictions in new column
        y_insample['cv_rf_depth' + str(depth)] = rf_cv.predict(X_cv_insample)
        y_outsample['cv_rf_depth' + str(depth)] = rf_cv.predict(X_cv_outsample)
        y_insample['tfidf_rf_depth' + str(depth)] = rf_tfidf.predict(X_tfidf_insample)
        y_outsample['tfidf_rf_depth' + str(depth)] = rf_tfidf.predict(X_tfidf_outsample)
    
    return y_insample, y_outsample # return dataframe with preds appended

def logisticregression(X_insample, y_insample, X_outsample, y_outsample):
    X_cv_insample, X_cv_outsample = count_vectorizer(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = tfidf(X_insample, X_outsample)
    logit_cv = LogisticRegression(random_state=123)\
        .fit(X_cv_insample, y_insample.in_actuals)
    logit_tfidf = LogisticRegression(random_state=123)\
        .fit(X_tfidf_insample, y_insample.in_actuals)
    y_insample['cv_logit'] = logit_cv.predict(X_cv_insample)
    y_outsample['cv_logit'] = logit_cv.predict(X_cv_outsample)
    y_insample['tfidf_logit'] = logit_tfidf.predict(X_tfidf_insample)
    y_outsample['tfidf_logit'] = logit_tfidf.predict(X_tfidf_outsample)
    
    return y_insample, y_outsample # return dataframe with preds appended

def naivebayes(X_insample, y_insample, X_outsample, y_outsample):
    X_cv_insample, X_cv_outsample = dense_cv(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = dense_tfidf(X_insample, X_outsample)
    smooth_levels = [.001, .01, 10, 100]
    # loop through smoothing levels
    for smooth_level in smooth_levels:
        nb_cv = GaussianNB(var_smoothing=smooth_level)\
            .fit(X_cv_insample, y_insample.in_actuals)
        nb_tfidf = GaussianNB(var_smoothing=smooth_level)\
            .fit(X_tfidf_insample, y_insample.in_actuals)
        # make predictions in new column
        y_insample['cv_nb_vsmooth' + str(smooth_level)] = nb_cv.predict(X_cv_insample)
        y_outsample['cv_nb_vsmooth' + str(smooth_level)] = nb_cv.predict(X_cv_outsample)
        y_insample['tfidf_nb_vsmooth' + str(smooth_level)] = nb_tfidf.predict(X_tfidf_insample)
        y_outsample['tfidf_nb_vsmooth' + str(smooth_level)] = nb_tfidf.predict(X_tfidf_outsample)
    
    return y_insample, y_outsample # return dataframe with preds appended

def knearestneighbors(X_insample, y_insample, X_outsample, y_outsample):
    X_cv_insample, X_cv_outsample = count_vectorizer(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = tfidf(X_insample, X_outsample)
    neighbor_counts = [3,5,10,25,75]
    # loop through neighbor counts
    for neighbor_count in neighbor_counts:
        # create knn models
        knn_cv = KNeighborsClassifier(n_neighbors=neighbor_count)\
            .fit(X_cv_insample, y_insample.in_actuals)
        knn_tfidf = KNeighborsClassifier(n_neighbors=neighbor_count)\
            .fit(X_tfidf_insample, y_insample.in_actuals)
        # make predictions in new column
        y_insample['cv_knn_n' + str(neighbor_count)] = knn_cv.predict(X_cv_insample)
        y_outsample['cv_knn_n' + str(neighbor_count)] = knn_cv.predict(X_cv_outsample)
        y_insample['tfidf_knn_n' + str(neighbor_count)] = knn_tfidf.predict(X_tfidf_insample)
        y_outsample['tfidf_knn_n' + str(neighbor_count)] = knn_tfidf.predict(X_tfidf_outsample)
    
    return y_insample, y_outsample # return dataframe with preds appended

# -------------------------- the vectorizers -------------------------- #

def count_vectorizer(X_insample, X_outsample):
    cv = CountVectorizer()
    X_cv_insample = cv.fit_transform(X_insample)
    X_cv_outsample = cv.transform(X_outsample)
    return X_cv_insample, X_cv_outsample

def tfidf(X_insample, X_outsample):
    tfidf = TfidfVectorizer()
    X_tfidf_insample = tfidf.fit_transform(X_insample)
    X_tfidf_outsample = tfidf.transform(X_outsample)
    return X_tfidf_insample, X_tfidf_outsample

def dense_cv(X_insample, X_outsample):
    cv = CountVectorizer()
    X_cv_insample = cv.fit_transform(X_insample).todense()
    X_cv_outsample = cv.transform(X_outsample).todense()
    return X_cv_insample, X_cv_outsample
def dense_tfidf(X_insample, X_outsample):
    tfidf = TfidfVectorizer()
    X_tfidf_insample = tfidf.fit_transform(X_insample).todense()
    X_tfidf_outsample = tfidf.transform(X_outsample).todense()
    return X_tfidf_insample, X_tfidf_outsample