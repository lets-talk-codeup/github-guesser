# NLP - Guessing Github
Analyze Github READMEs to try to guess programming language using Natural Language Processing techniques.

# Team Members
- Jason Tellez
- Jeff Akins
- Veronica Reyes
- Jacob Paxton

# Data Links
- 1,500 Original READMEs from Microsoft's GitHub org, scraped on 10/28/2021 (JSON): 
    * https://drive.google.com/file/d/1tu9A0pWc-At6tvAGwQp4Ww8WwqDHbjxD/view
- Cleaned, tokenized, lemmatized, stopwords-removed READMEs file (JSON):
    * https://drive.google.com/file/d/1aec5UqivmWouJ0DqFM-3Nn3yE1Bd-E7f/view
    
    
    
## Table of contents

- [Table of Contents](#table-of-contents)
- [Project Summary](#project-summary)
- [Executive Summary](#executive-summary)
- [Data Dictionary](#dictionary)
- [Pipeline](#pipeline)
- [Conclusions](#conclusions)   
- [Recreate These Results](#recreate-these-results)

---

## Project Summary
[(Back to top)](#table-of-contents)

#### Goal
The goal of this project was to build a model that can predict the primary programming language for a GitHub repository, given the text of the README file. To achieve this, we first had to decide which and how many repos we wanted to acquire. Microsoft has a large number of repos on its GitHub site with a wide variety of coding languages, so we determined that we could pull the README and coding language from their repos. This required the use of a variety of web scraping and Natural Language Processing tools as well as use of GitHub's API. In the end we acquired 1500 READMEs along with their associated primary coding language from Microsoft's GitHub page. The results are contained in this notebook as well as in a presentation slide deck.

#### Deliverables
1. A final notebook. This notebook will be used as a walkthough to present results and allow users to recreate the findings.
2. Slides presenting our project and findings
3. A video created by our team talking through our slides

**Additional Deliverables**

4. Python modules that automate the data pipeline process. These modules will be imported and used in the final notebook.
5. This README that describes what the project is and pipeline process as well as steps on how to recreate our results.
6. A Trello board that details the process of creating this project.

---

## Executive Summary 
[(Back to top)](#table-of-contents)

After acquiring and exploring the READMEs collected, we determined that the most common coding language in Microsoft's repos was TypeScript. Therefore, we decided to use classification modeling to attempt to predict whether the repos used TypeScript or not based on features from the READMEs. We were able to predict with an 84% accuracy whether a repo used TypeScript or not based on types of words and word length of a README.

---

## Data Dictionary
[(Back to top)](#table-of-contents)

key|datatype|description
|:------------------|:------------------------|:-------------|                   
repo                  |object              |Name of the repo  |
language              |object              |The main language of the repo  |
clean                 |object              |Normalized README content of repo  |
lemma_no_stopwords    |object              |Lemmatized and normalized README content with stopwords removed  |
clean_word_count      |int64               |Word count of the normalized README  |
readme_char_count     |int64               |Character count of the normalized README  |
is_TypeScript         |bool                |Boolean value of the repo's language being TypeScript  |

---
 
## Pipeline
[(Back to top)](#table-of-contents)

### Plan
- Create list of repos to be scraped by acquire.py
- Create a README that will hold project information
- Create a [Trello board](https://trello.com/b/j2IoVnUG/nlp-project) that defines project goals
- Prepare a slideshow that will hold presentaion of results

### Wrangle (Acquire/Prepare)
- Created a function that scrapes repos from [Microsft's github page](https://github.com/microsoft) and input list of repo's through acquire.py to create .json file
- Read .json file as dataframe with columns:
    - `repo`
    - `language`
    - `readme_contents`
- By running `readme_contents` values through BeautifulSoup's 'html.parser' and normalizing functions, we returned cleaned README files containing non-html language from each repo. We called this new column `clean`.
- From the `clean` column, we lemmatized each README and removed stopwords to get a column of more generalized README
README
- We created additional columns that counted the total number of words in each README (`clean_word_count`) ,as well as the total number of character in each README (`readme_char_count`)
- Created `is_TypeScript` column to act as our target variable.
- We removed rows that had null values for the `language` columns and rows that had empty, non-null values in the `lemma_no_stopwords` column.
    - The empty rows were not considered to have nulls because pandas identified these columns as non-nulls despite having no characters within the READMEs
- Initial dataframe: 1500 rows x 3 columns
- Final dataframe: 1370 rows x 7 columns

### Explore
- TypeScript is the most common type of programming language used making up 25% of the the repos acquired. 
- We will make TypeScript the target programming language to predict.
- The most common words in READMEs for ALL repos are:
    - project      
    - microsoft    
    - run          
    - 9            
    - azure 

- The most common words in READMEs for TypeScript repos are:
    - 9            
    - project      
    - run          
    - microsoft    
    - extension  
    
- The most common words in READMEs for Non-TypeScript repos are:
    - project      
    - microsoft    
    - azure        
    - run          
    - data  
    
- README lengths do seem to vary by programming language.
   - MATLAB repos are the largest READMEs and Dafny repos are the smallest.
- IDFs range from 1.19 for the word 'microsoft' to 7.08 for several words
- The mean is  generally lower for TypeScript message lengths compared to Non-TypeScript message lengths.
- The mean message length is the not same all programming languages.
- The word 'data' does occur more often in non-TypeScript documents than in TypeScript.
- Most repositories do not contain the word 'Data' or it appears less than 20 times

### Model
The goal of the modeling step is to create several classification models with different hyperparameter combinations to best-predict whether a README belongs to a TypeScript repository or not. The model creation, fit, and prediction is tied to one function in model.py, and the accuracy scores are tied to another function.

Results:

    1. The baseline model had an accuracy of **74%** on all splits.
    2. Our best-performing model, TFIDF-Vectorized DecisionTreeClassifier with max_depth=3 and random_state=123, had the following accuracies:
        - In-Sample Split: **86%**
        - Out-of-Sample Split #1: **86%**
        - Out-Of-Sample Split #2: **81%**

    Taking the average of the two out-of-sample splits, we assess that **our model can predict repository language using README contents with 84% accuracy.** Our model significantly outperforms the baseline for prediction accuracy.

---

## Conclusions
[(Back to top)](#table-of-contents)

After acquiring and exploring the READMEs collected, we determined that the most common coding language in Microsoft's repos was TypeScript. Therefore, we decided to use classification modeling to attempt to predict whether the repos used TypeScript or not based on features from the READMEs. We were able to predict with an 84% accuracy whether a repo used TypeScript or not based on types of words and word length of a README.

### Next Steps

Given more time, we would like to:

- 

## Recreate These Results
[(Back to top)](#table-of-contents)

There are two methods that you can use to recreate this project.
1. **Quick Method:** Utilize our final .json file with the cleaned README files. This is the simplest method and will produce the same results that we were able to achieve.
2. **Long Method:** Start from scratch using the same functions that we used. This will pull in the most recent repos from Microsoft's GitHub page and therefore will produce slightly different results from what we achieved. This method will also take longer e.g. it took us nearly 30 minutes to download the data from the 1500 repos.
