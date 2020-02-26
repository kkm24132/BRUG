# Demonstrate various techniques and comparision of outcomes 
# Ensembling in R (Bagging + Boosting)

####### Load required libraries
library(caret) # For leveraging caret package to use multiple algorithms provided by it
library(randomForest) # For random forest
library(gbm) # For gradient boosting
library(tree) # For simple decision trees
library(ISLR)
library(randomGLM) # For GLM to be used in caret package

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

set.seed(124)

inTraining <- createDataPartition(Carseats$Sales, p=0.75, list = FALSE)
train <- Carseats[inTraining,]
test <- Carseats[-inTraining,]

fitControl <- trainControl(method = "repeatedcv",
                           number = 6,
                           repeats = 5)

# Random Forest
set.seed(124)
# Grid Search strategy - We also have defined here a grid of algorithm to tunning model. Each axis of grid
#                        is an algorithm parameter and point in grid are specific combinations of parameter. 
model_rf <- train(Sales ~ ., 
                  data = train,
                  #tuneGrid = expand.grid(mtry = c(6,11),
                  #  splitrule = c("gini", "extratrees"),
                  #  min.node.size = c(1, 2, 4, 6, 8, 10)),
                  method="rf",
                  metric = "RMSE",
                  # preprocess definitions:
                    # zv -  identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations
                    # center - subtracts the mean of the predictor's data (again from the data in x) from the predictor values
                    # scale - divides by the standard deviation
                  preProcess = c("center","scale","zv"),
                  trControl = fitControl,
                  verbose = TRUE
                  )
model_rf
predict_rf = predict(model_rf, test)
mean((predict_rf-test$Sales)^2)
plot(predict_rf,test$Sales)
abline(0,1)

# GLM (Generalized Linear Model) - using method as randomGLM 
set.seed(124)
model_GLM <- train(Sales ~ ., 
                   data = train,
                   method = "randomGLM",
                   trControl = fitControl,
                   verbose = TRUE)
model_GLM
predict_GLM = predict(model_GLM, test)
mean((predict_GLM - test$Sales)^2)
plot(predict_GLM,test$Sales)
abline(0,1)

# GLM (Generalized Linear Model) - using method as glmnet 
set.seed(124)
model_GLMnet <- train(Sales ~ ., 
                   data = train,
                   method = "glmnet",
                   trControl = fitControl,
                   verbose = TRUE)
model_GLMnet
predict_GLMnet = predict(model_GLMnet, test)
mean((predict_GLMnet - test$Sales)^2)
plot(predict_GLMnet,test$Sales)
abline(0,1)


# GBM experiment 1
set.seed(124)
model_gbm <- train(Sales ~ ., 
                   data = train,
                   method = "gbm",
                   trControl = fitControl,
                   verbose = TRUE)
model_gbm
predict_gbm = predict(model_gbm, test)
mean((predict_gbm - test$Sales)^2)
plot(predict_gbm,test$Sales)
abline(0,1)

# GBM experiment 2 - with tunegrid parameter tunings

set.seed(124)
model_gbm2 <- train(Sales ~ ., 
                   data = train,
                   method = "gbm",
                   trControl = fitControl,
                   verbose = TRUE,
                   tuneGrid = expand.grid(interaction.depth = c(2),
                                          n.trees = (1:5)*50,
                                          shrinkage = 0.1,
                                          n.minobsinnode = 10)
                   )
model_gbm2
predict_gbm2 = predict(model_gbm2, test)
mean((predict_gbm2 - test$Sales)^2)
plot(predict_gbm2,test$Sales)
abline(0,1)

trellis.par.set(caretTheme())
plot(model_gbm)
ggplot(model_gbm)
densityplot(model_gbm, pch = "|")


# XGBoost - using method as xgbTree 
# set up the cross-validated hyper-parameter search
tuneGrid1 = expand.grid(
  nrounds          = c(25, 50, 100, 200, 350),
  eta              = c(0.001, 0.01, 0.1, 0.25, 0.4),        
  max_depth        = c(1, 2, 4, 6, 8),
  gamma            = 0,
  colsample_bytree = seq(0.4, 1.0, by = 0.2),
  min_child_weight = 1,
  subsample        = seq(0.4, 1.0, by = 0.2)            
)
# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 3,  
  allowParallel = TRUE
)
set.seed(124)
# model_xgbTree <- train(Sales ~ ., 
#                        data     = as.data.frame(train)[,2:11],
#                        method = "xgbTree",
#                        trControl = xgb_trcontrol_1,
#                        tuneGrid = tuneGrid1,
#                        verbose = TRUE)
# model_xgbTree
# predict_xgbTree = predict(model_xgbTree, test)
# mean((predict_xgbTree - test$Sales)^2)
# plot(predict_xgbTree,test$Sales)
# abline(0,1)


#### Compare models

resamps <- resamples(list(GBM = model_gbm, RF = model_rf, GLMnet = model_GLMnet, GLM = model_GLM))
resamps
summary(resamps)
dotplot(resamps, metric = "RMSE")
#xyplot(resamps, what = "BlandAltman")


################################################################################
## 3. Leverage H2O framework and leverage algorithms within it
################################################################################

# This also can be used similarly like above 
# Outcomes can be demonstrated using a similar sample dataset with similar seed settings to compare
