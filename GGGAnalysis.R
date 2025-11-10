library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

## Read in Training Data

trainData <- vroom('train.csv') %>%
  mutate(type = case_when(
    type == "Ghost" ~ 0,
    type == "Ghoul" ~ 1,
    type == "Goblin" ~ 2
  )) %>%
  mutate(type = factor(type))

## Read in Test Data

testData <- vroom('test.csv')

## EDA
#####

# library(ggplot2)
# library(ggmosaic)
# library(forcats)
# 
# ggplot(data=trainData, aes(x=type, y=bone_length)) +
#   geom_boxplot()

#####

## Recipe
#####

# ggg_recipe <- recipe(type ~ ., data = trainData) %>%
#   step_rm(id) %>%
#   step_mutate_at(color, fn = factor) %>%
#   step_dummy(all_factor_predictors())
# 
# ggg_prep <- prep(ggg_recipe)
# bake(ggg_prep, new_data = trainData)

#####

## Random Forest - Score: 0.72022
#####

# library(rpart)
# 
# tree_mod <- rand_forest(mtry=tune(),
#                         min_n=tune(),
#                         trees=tune()) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# tree_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(tree_mod)
# 
# ### Grid of values to tune Over
# 
# tuning_grid <- grid_regular(mtry(range=c(1,9)),
#                             min_n(),
#                             trees(range=c(100,1000)),
#                             levels=5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tree_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=(metric_set(roc_auc)))
# 
# ### Find best tuning parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ### Finalize workflow
# 
# final_wf <-
#   tree_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# tree_predictions <- final_wf %>%
#   predict(new_data = testData, type="class")
# 
# ### Kaggle
# 
# tree_kaggle_submission <- tree_predictions %>%
#   bind_cols(., testData) %>%
#   select(id, .pred_class) %>%
#   rename(type=.pred_class) %>%
#   mutate(type = case_when(
#     type == 0 ~ "Ghost",
#     type == 1 ~ "Ghoul",
#     type == 2 ~ "Goblin"
#   ))
# 
# vroom_write(x=tree_kaggle_submission, file="./RandForPreds.csv", delim=',')

#####

## Naive Bayes - Score: 0.72778
#####

# library(discrim)
# 
# nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# nb_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(nb_model)
# 
# ## Grid of parameters
# 
# tuning_grid <- grid_regular(Laplace(),
#                             smoothness(),
#                             levels = 5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tune_grid(
#   nb_workflow,
#   resamples = folds,
#   grid = tuning_grid,
#   metrics = metric_set(roc_auc))
# 
# ### Find best parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ### Finalize Workflow
# 
# final_wf <-
#   nb_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# nb_predictions <- final_wf %>%
#   predict(new_data = testData, type="class")
# 
# ### Kaggle
# 
# nb_kaggle_submission <- nb_predictions %>%
#   bind_cols(., testData) %>%
#   select(id, .pred_class) %>%
#   rename(type=.pred_class) %>%
#   mutate(type = case_when(
#     type == 0 ~ "Ghost",
#     type == 1 ~ "Ghoul",
#     type == 2 ~ "Goblin"
#   ))
# 
# vroom_write(x=nb_kaggle_submission, file="./NaiveBayesPreds.csv", delim=',')

#####

## KNN - Score: 0.70132
#####

# library(kknn)
# 
# knn_model <- nearest_neighbor(neighbors = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# knn_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(knn_model)
# 
# ### Tuning Parameters
# 
# tuning_grid <- grid_regular(neighbors(),
#                             levels = 5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tune_grid(
#   knn_workflow,
#   resamples = folds,
#   grid = tuning_grid,
#   metrics = metric_set(roc_auc))
# 
# ### Find Best K
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ### Finalize Workflow
# 
# final_wf <-
#   knn_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# knn_predictions <- final_wf %>%
#   predict(new_data = testData, type="class")
# 
# ### Kaggle
# 
# knn_kaggle_submission <- knn_predictions %>%
#   bind_cols(., testData) %>%
#   select(id, .pred_class) %>%
#   rename(type=.pred_class) %>%
#   mutate(type = case_when(
#     type == 0 ~ "Ghost",
#     type == 1 ~ "Ghoul",
#     type == 2 ~ "Goblin"
#   ))
# 
# vroom_write(x=knn_kaggle_submission, file="./KNNPreds.csv", delim=',')

#####

## Radial SVM - Score: 0.72778
#####

# library(kernlab)
# 
# svm_model <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svm_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(svm_model)
# 
# ### Tuning Parameters
# 
# tuning_grid <- grid_regular(rbf_sigma(),
#                             cost(),
#                             levels = 5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tune_grid(
#   svm_workflow,
#   resamples = folds,
#   grid = tuning_grid,
#   metrics = metric_set(roc_auc))
# 
# ### Find best parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ### Finalize Workflow
# 
# final_wf <-
#   svm_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# svm_predictions <- final_wf %>%
#   predict(new_data = testData, type="class")
# 
# ### Kaggle
# 
# svm_kaggle_submission <- svm_predictions %>%
#   bind_cols(., testData) %>%
#   select(id, .pred_class) %>%
#   rename(type=.pred_class) %>%
#   mutate(type = case_when(
#     type == 0 ~ "Ghost",
#     type == 1 ~ "Ghoul",
#     type == 2 ~ "Goblin"
#   ))
# 
# vroom_write(x=svm_kaggle_submission, file="./SVMPreds.csv", delim=',')

#####

## Linear SVM - Score: 0.73724, With Adjustments: 0.74102, More Adjust: 0.73913
#####

library(kernlab)

svm_lin_model <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_lin_workflow <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(svm_lin_model)

### Tuning Parameters

# tuning_grid <- grid_regular(cost(),
#                             levels = 10)

tuning_grid <- grid_regular(cost(),
                            levels = 20)

#### Considering cost from best model so far: 0.4017056

# tuning_grid <- grid_regular(cost(range=c(0.4,0.41)),
#                             levels = 20)

### CV

# folds <- vfold_cv(trainData, v = 5, repeats = 1)

folds <- vfold_cv(trainData, v = 10, repeats = 5)

# folds <- vfold_cv(trainData, v = 15, repeats = 1)

CV_results <- tune_grid(
  svm_lin_workflow,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc, accuracy))

### Find best parameters

# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")

bestTune <- CV_results %>%
  select_best(metric="accuracy")

### Finalize Workflow

final_wf <-
  svm_lin_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

### Predict

svm_lin_predictions <- final_wf %>%
  predict(new_data = testData, type="class")

### Kaggle

svm_lin_kaggle_submission <- svm_lin_predictions %>%
  bind_cols(., testData) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class) %>%
  mutate(type = case_when(
    type == 0 ~ "Ghost",
    type == 1 ~ "Ghoul",
    type == 2 ~ "Goblin"
  ))

# vroom_write(x=svm_lin_kaggle_submission, file="./SVMLinPreds.csv", delim=',')

vroom_write(x=svm_lin_kaggle_submission, file="./SVMLinPredsAdj.csv", delim=',')

# vroom_write(x=svm_lin_kaggle_submission,
#             file="./SVMLinPredsMoreAdj.csv", delim=',')

#####

## Polynomial SVM - Score: 0.71833
#####

# library(kernlab)
# 
# svm_poly_model <- svm_poly(degree = tune(), cost = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svm_poly_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(svm_poly_model)
# 
# ### Tuning Parameters
# 
# tuning_grid <- grid_regular(degree(),
#                             cost(),
#                             levels = 5)
# 
# ### CV
# 
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# 
# CV_results <- tune_grid(
#   svm_poly_workflow,
#   resamples = folds,
#   grid = tuning_grid,
#   metrics = metric_set(roc_auc))
# 
# ### Find best parameters
# 
# bestTune <- CV_results %>%
#   select_best(metric="roc_auc")
# 
# ### Finalize Workflow
# 
# final_wf <-
#   svm_poly_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ### Predict
# 
# svm_poly_predictions <- final_wf %>%
#   predict(new_data = testData, type="class")
# 
# ### Kaggle
# 
# svm_poly_kaggle_submission <- svm_poly_predictions %>%
#   bind_cols(., testData) %>%
#   select(id, .pred_class) %>%
#   rename(type=.pred_class) %>%
#   mutate(type = case_when(
#     type == 0 ~ "Ghost",
#     type == 1 ~ "Ghoul",
#     type == 2 ~ "Goblin"
#   ))
# 
# vroom_write(x=svm_poly_kaggle_submission, file="./SVMPolyPreds.csv", delim=',')

#####