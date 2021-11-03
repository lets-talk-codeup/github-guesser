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

-

---

## Data Dictionary
[(Back to top)](#table-of-contents)

key|datatype|description
|:------------------|:------------------------|:-------------|                   
repo                  |object              |Name of the repo  |
language              |object              |The main language of the repo  |
clean                 |object              |Normalized readme content of repo  |
lemma_no_stopwords    |object              |Lemmatized and normalized readme content with stopwords removed  |
clean_word_count      |int64               |Word count of the normalized readme  |
readme_char_count     |int64               |Character count of the normalized readme  |
is_TypeScript         |bool                |Boolean value of the repo's language being TypeScript  |

---
 
## Pipeline
[(Back to top)](#table-of-contents)

### Plan
- 

### Acquire
- 

### Prepare
- 

### Explore
- 

### Model
- 

---

## Conclusions
[(Back to top)](#table-of-contents)
    
- 

### Next Steps

Given more time, we would like to:

- 

## Recreate These Results
[(Back to top)](#table-of-contents)
