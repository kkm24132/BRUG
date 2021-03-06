# BRUG meetup (Bangalore R User Group)

**This repository captures content of my presentations, code notebooks / information shared in Bangalore R User Group meetup events.**

## Bagging and Boosting in R
- Please refer to the deck [Bagging_Boosting_in_R.pdf](/Bagging_Boosting_in_R.pdf) for presentation content.
- The related source file depicting comparision between algorithms using a sample dataset is demonstrated [here](https://github.com/kkm24132/BRUG/blob/master/src/Ensembling.R)

## Interpretable Machine Learning / ML Explainability / XAI
- Please refer to the deck [ML_Interpretability.pdf](/ML_Interpretability.pdf) for presentation content.
- The related Jupyter notebook depicting sample dataset demonstration of a LIME package usage is captured [here](https://github.com/kkm24132/BRUG/blob/master/src/MLExplainability_using_LIME.ipynb) The [nbviewer version is here](https://nbviewer.jupyter.org/github/kkm24132/BRUG/blob/8c5e0d15b1250f86a47d288a6ea9912279fe2b8a/src/MLExplainability_using_LIME.ipynb) which will display the charts for some LIME explanations appropriately.
- Following categorization can be looked at from Model Explainability and Interpretability standpoint:
  - Transparent models (Linear / Logistic Regression, Decision trees, k-Nearest Neighbours, Rule based learners, Bayesian models, Generative additive models)
  - Opaque models (Random forests, SVM, Multi layer Neural Net)
  - Both of these Transparent and Opaque models have Model agnostic and Model specific explainability categories
    - Model Agnostic
      - Explanation by simplification
        - Rule based learner
        - Decision tree
      - Feature relevance explanation
        - Influence functions
        - Sensitivity
        - Game theory inspired (SHAP)
        - Interaction based
      - Local explanations
        - Rule based learner (Anchors)
        - Linear approximation (LIME)
        - Counterfactuals
      - Visual explanations
        - Sensitivity
        - Dependency plots (ICE, PDP etc.)
    - Model Specific
      - Explanation by simplification
        - Rule based learners (InTrees)
        - Decision trees (InTrees)
        - Distilation
      - Feature relevance explanation
        - Feature importance
  

## 10+10 use cases in Retail in 2020

- Please refer to [10plus10_in_Retail_in_2020.pdf](/10plus10_in_Retail_in_2020.pdf) for the presentation.
- The purpose is to present predictions around key business use cases that will continue to dominate in Retail in Data Science and AI in 2020. Stakeholders should focus these use cases to get benefits and create impact and value for their business.

## Extreme Gradient Boosting in R

- The purpose is to understand concepts of the scalable tree boosting approach (XGBoost) in R, it's features etc. This solves many data science problems in relatively fast and accurate manner.
- The code snippet is under [here](/src)
- Please refer to [XGBoost_in_R.pdf](/XGBoost_in_R.pdf) for the presentation.
- DIsclaimer: Some of this content, approach are reused from various references. 

## Deep Learning using R

- Please refer to the [DeepLearning_using_R.pdf](/DeepLearning_using_R.pdf) for details.
- Objective is to provide a very high level view about Deep learning and some packages in R. Example of MNIST dataset can be leveraged to showcase the use of R using various libraries.


## Packages / Libraries in R

- Please refer to ``` Packages_and_OOP_in_R.pdf ``` deck for content on the presentation conducted as part of BRUG. This focuses on Packages and OOP in R (at a high level).

- The following list does not involve entire exhaustive list. However, the intent is to provide some key and important packages that are used and helpful in most CRISP-DM phases.

### R Packages at a glance by category

Package category|Package Name|Features
----------------------|---------------------|-------------------------
Data Manipulation|dplyr|Data wrangling, working with remote data frames
Data Manipulation|data.table|Data aggregation involving large datasets, file reader and parallel file writer
Data Manipulation|lubridate|Working with date and time formats, parsing of date-time data
Data Manipulation|jsonlite|Robust parsing of JSON objects in R


Package category|Package Name|Features
----------------------|---------------------|-------------------------
Graphic Display|ggplot2|Powerful implementation of the grammar of graphics visualization, Plot specifications
Graphic Display|corrplot|Abilities to visualize correlation matrices and confidence intervals
Graphic Display|lattice|Emphasis on multivariate data


Package category|Package Name|Features
----------------------|---------------------|-------------------------
HTML Widget|plotly|Rich features around charts, web based toolbox for building visualizations
HTML Widget|ggvis|Implementation of an interactive grammar of graphic
HTML Widget|DT(DataTables)|Displays R matrices and data frames as interactive HTML tables
HTML Widget|rCharts|Interactive JS charts from R


Package category|Package Name|Features
----------------------|---------------------|-------------------------
Reproducible Research|knitr|Easy dynamic report generation in R, enables integration of R code into LaTex, HTML, Markdown, AsciiDoc, reStructuredText documents
Reproducible Research|rMarkdown|Next generation implementation of R Markdown based on pandoc
Reproducible Research|slidify|Generated reproducible html5 slides from R markdown


Package category|Package Name|Features
----------------------|---------------------|-------------------------
Machine Learning|mlr|Extensible framework for classification, regression, survival analysis and clustering, easy extension mechanism through S3 inheritance
Machine Learning|xgboost|Implementation of Gradient Boosted Decision Trees algorithm
Machine Learning|caret|Multiple model comparision and usage for classification and regression
Machine Learning|gbm|Generalized Boosted Regression Models
Machine Learning|prophet|Forecast for time series data, manages data with multiple seasonality with linear or non-linear growth
Machine Learning|randomforest|Implements Breiman's random forest algorithm for classification
Machine Learning|Arules|Mining Association Rules and Frequent itemsets
Machine Learning|Boruta|Wrapper algorithm for all relevant feature selection
Machine Learning|Forecast|Timeseries forecasting using ARIMA, ETS, STLM, TBATS, and neural network models
Machine Learning|Anomalize|Tidy Anomaly Detection using Twitter’s AnomalyDetection method
Machine Learning|AnomalyDetection|AnomalyDetection R package from Twitter
Machine Learning|e1071|Misc Functions of the Department of Statistics (e1071)
Machine Learning|MXNet|MXNet brings flexible and efficient GPU computing and state-of-art deep learning to R


Package category|Package Name|Features
----------------------|---------------------|-------------------------
Web Search|Rcurl|general network (HTTP/FTP…) client interface for R
Web Search|Curl|flexible web client for R
Web Search|Httr|user friendly Rcurl wrapper
Web Search|shiny|simple interactive web applications with R
Web Search|Plumber|A library to expose existing R code as web API
Web Search|Rfacebook|access to facebook API via R


Package category|Package Name|Features
----------------------|---------------------|-------------------------
Database Management|RODBC|ODBC database access for R
Database Management|DBI|common interface between R and DBMS
Database Management|Elastic|wrapper for elastic search HTTP API
Database Management|ROracle|OCI based Oracle database interface for R
Database Management|RPostgreSQL|R interface to PostgreSQL database system
Database Management|RSQLite|SQLite interface for R


Package category|Package Name|Features
----------------------|---------------------|-------------------------
NLP Specific|text2vec|Fast Text Mining Framework for Vectorization and Word Embeddings
NLP Specific|tm|A comprehensive text mining framework for R
NLP Specific|OpenNLP|Apache OpenNLP Tools Interface
NLP Specific|koRpus|An R Package for Text Analysis
NLP Specific|LDAvis|Interactive visualization of topic models
NLP Specific|SnowballC|Snowball stemmers based on the C libstemmer UTF-8 library
NLP Specific|Tidytext|Implementing tidy principles of Hadley Wickham to text mining


Package category|Package Name|Features
----------------------|---------------------|-------------------------
Optimization|lpSolve|Interface to Lp_solve to Solve Linear/Integer Programs
Optimization|Minqa|Derivative-free optimization algorithms by quadratic approximation
Optimization|Nloptr|NLopt is a free/open-source library for nonlinear optimization
Optimization|Rglpk|R/GNU Linear Programming Kit Interface


Package category|Package Name|Features
----------------------|---------------------|-------------------------
Computer vision|magick|importing / converting to/from all formats / basic image manipulation
Computer vision|imageR|image processing library based on “CImg” (interpolation, resizing, filtering, fourier transformations, denoising, gradients, blurring)
Computer vision|OpenImageR|an image processing toolkit (hashing, edge detection, manipulation)


*Disclaimer:* 
- *The contents of this document are to best of my knowledge and based on my own experiences only. Some data and names MAY BE tweaked/masked to take care of data privacy, sensitivity and business sensitivity aspects if applicable. The information provided is purely to highlight experience gathered with clear business impact created and NO WAY RELATES TO ANY ORGANIZATION or ORGANIZATION's OPINIONS, VIEWS.*
- *Intent is for knowledge sharing and continuous learning as much as possible.*
- *Focus is also to share from the quorum and leverage from lessons learnt, continuous learning.*

