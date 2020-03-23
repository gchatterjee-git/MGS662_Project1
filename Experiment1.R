#Importing library glmnet
library(glmnet)
blogTrain = read.csv("blogData_train.csv", header = FALSE)
blogTest = read.csv("blogData_test-2012.03.03.00_00.csv", header = FALSE)
blogTest_2 = read.csv("blogData_test-2012.03.01.00_00.csv", header = FALSE)
blogTest_3 = read.csv("blogData_test-2012.02.02.00_00.csv", header = FALSE)
blogTest_4 = read.csv("blogData_test-2012.02.07.00_00.csv", header = FALSE)

#dividing the training data into basic features and target concept
features = blogTrain[,51:60]
basic_features = model.matrix(~., features)[,-1]


target_concept = blogTrain[,281]


#dividing the test data into basic features and target concept
features_test = blogTest[,51:60]
basic_test_features = model.matrix(~., features_test)[,-1]

features_test_2 = blogTest_2[,51:60]
basic_test_features_2 = model.matrix(~., features_test_2)[,-1]

features_test_3 = blogTest_3[,51:60]
basic_test_features_3 = model.matrix(~., features_test_3)[,-1]

features_test_4 = blogTest_4[,51:60]
basic_test_features_4 = model.matrix(~., features_test_4)[,-1]


target_concept_test = blogTest[,281]
target_concept_test_2 = blogTest_2[,281]
target_concept_test_3 = blogTest_3[,281]
target_concept_test_4 = blogTest_4[,281]

#defining a range for lambda
lambda = 10^seq(10,-2,length = 100)

#Ridge regression model
ridge_model = glmnet(basic_features,target_concept,alpha = 0 ,lambda = lambda)


#Cross-validating to find the best lambda value
cross_val = cv.glmnet(basic_features,target_concept, alpha = 0)

#Selecting lambda that minimizes training MSE
best_lambda = cross_val$lambda.min


#Predicting test data using best lambda
ridge_pred = predict(ridge_model, s = best_lambda, newx = basic_test_features)
ridge_pred_2 = predict(ridge_model, s = best_lambda, newx = basic_test_features_2)
ridge_pred_3 = predict(ridge_model, s = best_lambda, newx = basic_test_features_3)
ridge_pred_4 = predict(ridge_model, s = best_lambda, newx = basic_test_features_4)

#MSE
MSE_ridge = mean((ridge_pred - target_concept_test)^2)
print(paste0("MSE-Ridge for test set 1: ",MSE_ridge))

MSE_ridge_2 = mean((ridge_pred_2 - target_concept_test_2)^2)
print(paste0("MSE-Ridge for test set 2: ",MSE_ridge_2))

MSE_ridge_3 = mean((ridge_pred_3 - target_concept_test_3)^2)
print(paste0("MSE-Ridge for test set 3: ",MSE_ridge_3))

MSE_ridge_4 = mean((ridge_pred_4 - target_concept_test_4)^2)
print(paste0("MSE-Ridge for test set 4: ",MSE_ridge_4))
  
#Lasso regression model

lasso_model = glmnet(basic_features,target_concept,alpha = 1 ,lambda = lambda)

#cross validating for best lambda
lasso_model_cv = cv.glmnet(basic_features,target_concept, alpha = 1)

#selecting lambda that minimizes training MSE
best_lambda_lasso = lasso_model_cv$lambda.min

#Use best lambda to predict test data
lasso_pred = predict(lasso_model, s=best_lambda_lasso, newx = basic_test_features)
lasso_pred_2 = predict(lasso_model, s=best_lambda_lasso, newx = basic_test_features_2)
lasso_pred_3 = predict(lasso_model, s=best_lambda_lasso, newx = basic_test_features_3)
lasso_pred_4 = predict(lasso_model, s=best_lambda_lasso, newx = basic_test_features_4)

#MSE value for test data
MSE_lasso = mean((lasso_pred - target_concept_test)^2)
print(paste0("MSE-Lasso for test set 1: ",MSE_lasso))

MSE_lasso_2 = mean((lasso_pred_2 - target_concept_test_2)^2)
print(paste0("MSE-Lasso for test set 2: ",MSE_lasso_2))

MSE_lasso_3 = mean((lasso_pred_3 - target_concept_test_3)^2)
print(paste0("MSE-Lasso for test set 3: ",MSE_lasso_3))

MSE_lasso_4 = mean((lasso_pred_4 - target_concept_test_4)^2)
print(paste0("MSE-Lasso for test set 4: ",MSE_lasso_4))