import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# ------------------------- print results ------------------------- #

def print_nlp_results(y_insample, y_outsample):
    """ Takes in a dataframe with column-separated model predictions, 
        Calculates accuracy of each model (loops columns) for in- and out-sample data,
        Appends all results to running dataframe, 
        Returns dataframe. """
    # create empty results dataframe
    running_df = pd.DataFrame(columns=['Model','InSample_Accuracy','OutSample_Accuracy'])
    # loop through each model
    for model in y_insample.columns[1:]:
        # calculate model accuracy
        in_accuracy = (y_insample[model] == y_insample.in_actuals).mean()
        out_accuracy = (y_outsample[model] == y_outsample.out_actuals).mean()
        # add results to new row in dataframe
        running_df = running_df.append({'Model':model,
                                        'InSample_Accuracy':in_accuracy, 
                                        'OutSample_Accuracy':out_accuracy},
                                         ignore_index=True)

    return running_df # return results dataframe

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
    # Baseline - add predictions to df
    y_insample, y_outsample = nlp_bl(y_insample, y_outsample)
    # Decision Tree classifier - add predictions to df
    y_insample, y_outsample = decisiontree(X_insample, y_insample, X_outsample, y_outsample)
    # Random Forest classifier - add predictions to df
    y_insample, y_outsample = randomforest(X_insample, y_insample, X_outsample, y_outsample)
    # Logistic Regression classifier - add predictions to df
    y_insample, y_outsample = logisticregression(X_insample, y_insample, X_outsample, y_outsample)
    # Naive Bayes classifier - add predictions to df
    y_insample, y_outsample = naivebayes(X_insample, y_insample, X_outsample, y_outsample)
    # K-Nearest Neighbors classifier - add predictions to df
    y_insample, y_outsample = knearestneighbors(X_insample, y_insample, X_outsample, y_outsample)
    
    return y_insample, y_outsample # return dataframes of predictions

# -------------------------- the models -------------------------- #

def nlp_bl(y_insample, y_outsample):
    """ Calculate baseline using mode class for model comparison """
    # find baseline
    mode = y_insample.in_actuals.mode().item()
    # set baseline as prediction
    y_insample['baseline'] = mode
    y_outsample['baseline'] = mode

    return y_insample, y_outsample # return df with baseline predictions column

def decisiontree(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates decision trees with max_depth 1,2,3,5,10 and random_state=123 """
    # transform data into count-vectorized and TFIDF-vectorized data
    X_cv_insample, X_cv_outsample = count_vectorizer(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = tfidf(X_insample, X_outsample)
    # set loop list
    max_depths = [1,2,3,5,10]
    # loop through max depths
    for depth in max_depths:
        # create trees for count-vectorized and TFIDF-vectorized documents
        tree_cv = DecisionTreeClassifier(max_depth=depth, random_state=123)\
            .fit(X_cv_insample, y_insample.in_actuals)
        tree_tfidf = DecisionTreeClassifier(max_depth=depth, random_state=123)\
            .fit(X_tfidf_insample, y_insample.in_actuals)
        # make predictions in new columns
        y_insample['cv_tree_maxdepth' + str(depth)] = tree_cv.predict(X_cv_insample)
        y_outsample['cv_tree_maxdepth' + str(depth)] = tree_cv.predict(X_cv_outsample)
        y_insample['tfidf_tree_maxdepth' + str(depth)] = tree_tfidf.predict(X_tfidf_insample)
        y_outsample['tfidf_tree_maxdepth' + str(depth)] = tree_tfidf.predict(X_tfidf_outsample)

    return y_insample, y_outsample # return dataframe with predictions appended

def randomforest(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates random forests with max_depth 1,2,3,5,10 and random_state=123 """
    # transform data into count-vectorized and TFIDF-vectorized data
    X_cv_insample, X_cv_outsample = count_vectorizer(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = tfidf(X_insample, X_outsample)
    # set loop list
    max_depths = [1,2,3,5,10]
    # loop through max depths
    for depth in max_depths:
        # create forests for count-vectorized and TFIDF-vectorized documents
        rf_cv = RandomForestClassifier(max_depth=depth, random_state=123)\
            .fit(X_cv_insample, y_insample.in_actuals)
        rf_tfidf = RandomForestClassifier(max_depth=depth, random_state=123)\
            .fit(X_tfidf_insample, y_insample.in_actuals)
        # make predictions in new columns
        y_insample['cv_rf_depth' + str(depth)] = rf_cv.predict(X_cv_insample)
        y_outsample['cv_rf_depth' + str(depth)] = rf_cv.predict(X_cv_outsample)
        y_insample['tfidf_rf_depth' + str(depth)] = rf_tfidf.predict(X_tfidf_insample)
        y_outsample['tfidf_rf_depth' + str(depth)] = rf_tfidf.predict(X_tfidf_outsample)
    
    return y_insample, y_outsample # return dataframe with predictions appended

def logisticregression(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates logistic regressions with random_state=123 """
    # transform data into count-vectorized and TFIDF-vectorized data
    X_cv_insample, X_cv_outsample = count_vectorizer(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = tfidf(X_insample, X_outsample)
    # create logistic regressions for count-vectorized and TFIDF-vectorized documents
    logit_cv = LogisticRegression(random_state=123)\
        .fit(X_cv_insample, y_insample.in_actuals)
    logit_tfidf = LogisticRegression(random_state=123)\
        .fit(X_tfidf_insample, y_insample.in_actuals)
    # add columns for predictions
    y_insample['cv_logit'] = logit_cv.predict(X_cv_insample)
    y_outsample['cv_logit'] = logit_cv.predict(X_cv_outsample)
    y_insample['tfidf_logit'] = logit_tfidf.predict(X_tfidf_insample)
    y_outsample['tfidf_logit'] = logit_tfidf.predict(X_tfidf_outsample)
    
    return y_insample, y_outsample # return dataframe with predictions appended

def naivebayes(X_insample, y_insample, X_outsample, y_outsample):
    """ Creates Naive-Bayes with var_smoothing of .001, .01, 10, 100 """
    # transform data into count-vectorized and TFIDF-vectorized data
    X_cv_insample, X_cv_outsample = dense_cv(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = dense_tfidf(X_insample, X_outsample)
    # set loop list
    smooth_levels = [.001, .01, 10, 100]
    # loop through smoothing levels
    for smooth_level in smooth_levels:
        # create naive bayes for count-vectorized and TFIDF-vectorized documents
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
    """ Create KNNs with neighbor counts of 3, 5, 10, 25, 75 """
    # transform data into count-vectorized and TFIDF-vectorized data
    X_cv_insample, X_cv_outsample = count_vectorizer(X_insample, X_outsample)
    X_tfidf_insample, X_tfidf_outsample = tfidf(X_insample, X_outsample)
    # set loop list
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
    """ Return count-vectorized Pandas Series """
    # build vectorizer
    cv = CountVectorizer()
    # transform data
    X_cv_insample = cv.fit_transform(X_insample)
    X_cv_outsample = cv.transform(X_outsample)

    return X_cv_insample, X_cv_outsample # return transformed data

def tfidf(X_insample, X_outsample):
    """ Return TF-IDF Pandas Series """
    # build vectorizer
    tfidf = TfidfVectorizer()
    # transform data
    X_tfidf_insample = tfidf.fit_transform(X_insample)
    X_tfidf_outsample = tfidf.transform(X_outsample)

    return X_tfidf_insample, X_tfidf_outsample # return transformed data

def dense_cv(X_insample, X_outsample):
    """ Return *dense* count-vectorized Pandas Series """
    # build vectorizer
    cv = CountVectorizer()
    # transform data
    X_cv_insample = cv.fit_transform(X_insample).todense()
    X_cv_outsample = cv.transform(X_outsample).todense()

    return X_cv_insample, X_cv_outsample # return transformed data

def dense_tfidf(X_insample, X_outsample):
    """ Return *dense* TF-IDF Pandas Series """
    # build vectorizer
    tfidf = TfidfVectorizer()
    # transform data
    X_tfidf_insample = tfidf.fit_transform(X_insample).todense()
    X_tfidf_outsample = tfidf.transform(X_outsample).todense()

    return X_tfidf_insample, X_tfidf_outsample # return transformed data