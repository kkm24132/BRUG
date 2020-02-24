# Demonstrate various techniques and comparision of outcomes 
# Ensembling in R (Bagging + Boosting)

####### Load required libraries
library(caret) # For leveraging caret package to use multiple algorithms provided by it
library(randomForest) # For random forest
library(gbm) # For gradient boosting
library(tree) # For simple decision trees
library(ISLR)

####### About sample dataset that will be used
## Considering a sample dataset related "car seats" to compare various approaches
## A simulated data set containing sales of child car seats at 400 different stores.

# A data frame with 400 observations on the following 11 variables.
# 
# Sales - Unit sales (in thousands) at each location
# CompPrice - Price charged by competitor at each location
# Income - Community income level (in thousands of dollars)
# Advertising - Local advertising budget for company at each location (in thousands of dollars)
# Population - Population size in region (in thousands)
# Price - Price company charges for car seats at each site
# ShelveLoc - A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
# Age - Average age of the local population
# Education - Education level at each location
# Urban - A factor with levels No and Yes to indicate whether the store is in an urban or rural location
# US - A factor with levels No and Yes to indicate whether the store is in the US or not

################################################################################
## 1. Leverage individual libraries such as randomforest, gbm etc
################################################################################

## Regression problem - predict sales 
attach(Carseats)
Carseats = ISLR::Carseats

## Simple Decision Tree
set.seed(124)
data.Carseats <- sample(1:nrow(Carseats), nrow(Carseats) / 2)
train <- Carseats[data.Carseats, ]
test <- Carseats[-data.Carseats, ]
model_decisiontree = tree(Sales ~ .,data = train)
predict_decisiontree = predict(model_decisiontree,test)
mean((predict_decisiontree-test$Sales)^2)
plot(predict_decisiontree,test$Sales)
abline(0,1)
summary(model_decisiontree)
plot(model_decisiontree)
text(predict_decisiontree, pretty=0)

## Random Forest
set.seed(124)
train = sample(1:nrow(Carseats), 200)
test = Carseats[-train,]
model_rf = randomForest(Sales ~ ., data = Carseats,subset = train,importance=TRUE)
predict_rf = predict(model_rf, test)
mean((predict_rf-test$Sales)^2)
plot(predict_rf,test$Sales)
abline(0,1)
summary(model_rf)
importance(model_rf)
varImpPlot(model_rf)

## Random Forest with different variables consideration
set.seed(124)
model_rf = randomForest(Sales ~ ., data = Carseats,subset = train,mtry=6,importance=TRUE)
predict_rf = predict(model_rf, test)
mean((predict_rf-test$Sales)^2)
plot(predict_rf,test$Sales)
abline(0,1)
summary(model_rf)
importance(model_rf)
varImpPlot(model_rf)


## Gradient Boosting (GBM)
set.seed(124)
model_gbm = gbm(Sales ~ ., data = Carseats, distribution = "gaussian",
                n.trees = 5000, interaction.depth = 4)
predict_gb = predict(model_gbm, newdata = test,n.trees = 5000)
mean((predict_gb-test$Sales)^2)
plot(predict_gb,test$Sales)
abline(0,1)
summary(model_gbm)


################################################################################
## 2. Leverage caret library and leverage algorithms within it
################################################################################

# This also can be used similarly like above 
# Outcomes can be demonstrated using a similar sample dataset with similar seed settings to compare

################################################################################
## 3. Leverage H2O framework and leverage algorithms within it
################################################################################

# This also can be used similarly like above 
# Outcomes can be demonstrated using a similar sample dataset with similar seed settings to compare
