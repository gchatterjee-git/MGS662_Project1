#Importing library glmnet
library(glmnet)
blogTrain = read.csv("blogData_train.csv", header = FALSE)
blogTest = read.csv("blogData_test-2012.03.03.00_00.csv", header = FALSE)
blogTest_2 = read.csv("blogData_test-2012.03.01.00_00.csv", header = FALSE)
blogTest_3 = read.csv("blogData_test-2012.02.04.00_00.csv", header = FALSE)
blogTest_4 = read.csv("blogData_test-2012.02.07.00_00.csv", header = FALSE)


#dividing the training data into textual features and target concept
textual=blogTrain[,63:262]
textual_features = model.matrix(~., textual)[,-1]

target_concept = blogTrain[,281]

#dividing the test data into basic features and target concept
textual_features_t = blogTest[,63:262]
textual_features_test = as.matrix(textual_features_t)
target_concept_t_test = blogTest[,281]

textual_features_t_2 = blogTest_2[,63:262]
textual_features_test_2 = as.matrix(textual_features_t_2)
target_concept_t_test_2 = blogTest_2[,281]

textual_features_t_3 = blogTest_3[,63:262]
textual_features_test_3 = as.matrix(textual_features_t_3)
target_concept_t_test_3 = blogTest_3[,281]

textual_features_t_4 = blogTest_4[,63:262]
textual_features_test_4 = as.matrix(textual_features_t_4)
target_concept_t_test_4 = blogTest_4[,281]

#defining a range for lambda
lambda = 10^seq(10,-2,length = 100)

#Ridge regression model
ridge_model_textual = glmnet(textual_features,target_concept,alpha = 0 ,lambda = lambda)


#Cross-validating to find the best lambda value
cross_val = cv.glmnet(textual_features,target_concept, alpha = 0)

#Selecting lambda that minimizes training MSE
best_lambda = cross_val$lambda.min

#Predicting test data using best lambda
ridge_pred_textual = predict(ridge_model_textual, s = best_lambda, newx = textual_features_test)
ridge_pred_textual_2 = predict(ridge_model_textual, s = best_lambda, newx = textual_features_test_2)
ridge_pred_textual_3 = predict(ridge_model_textual, s = best_lambda, newx = textual_features_test_3)
ridge_pred_textual_4 = predict(ridge_model_textual, s = best_lambda, newx = textual_features_test_4)


#MSE for Ridge
MSE_ridge_textual = mean((ridge_pred_textual - target_concept_t_test)^2)
print (paste0("MSE-Ridge for test set 1: ",MSE_ridge_textual))

MSE_ridge_textual_2 = mean((ridge_pred_textual_2 - target_concept_t_test_2)^2)
print (paste0("MSE-Ridge for test set 2: ",MSE_ridge_textual_2))

MSE_ridge_textual_3 = mean((ridge_pred_textual_3 - target_concept_t_test_3)^2)
print (paste0("MSE-Ridge for test set 3: ",MSE_ridge_textual_3))

MSE_ridge_textual_4 = mean((ridge_pred_textual_4 - target_concept_t_test_4)^2)
print (paste0("MSE-Ridge for test set 4: ",MSE_ridge_textual_4))


#Lasso regression model

lasso_model_textual = glmnet(textual_features,target_concept,alpha = 1 ,lambda = lambda)

#cross validating for best lambda
lasso_model_cv = cv.glmnet(textual_features,target_concept, alpha = 1)

#selecting lambda that minimizes training MSE
best_lambda_lasso = lasso_model_cv$lambda.min

#Use best lambda to predict test data
lasso_pred_textual = predict(lasso_model_textual, s=best_lambda_lasso, newx = textual_features_test)
lasso_pred_textual_2 = predict(lasso_model_textual, s=best_lambda_lasso, newx = textual_features_test_2)
lasso_pred_textual_3 = predict(lasso_model_textual, s=best_lambda_lasso, newx = textual_features_test_3)
lasso_pred_textual_4 = predict(lasso_model_textual, s=best_lambda_lasso, newx = textual_features_test_4)

#MSE value for test data
MSE_lasso_textual = mean((lasso_pred_textual - target_concept_t_test)^2)
print (paste0("MSE-Lasso for test set 1: ",MSE_lasso_textual))

MSE_lasso_textual_2 = mean((lasso_pred_textual_2 - target_concept_t_test_2)^2)
print (paste0("MSE-Lasso for test set 2: ",MSE_lasso_textual_2))

MSE_lasso_textual_3 = mean((lasso_pred_textual_3 - target_concept_t_test_3)^2)
print (paste0("MSE-Lasso for test set 3: ",MSE_lasso_textual_3))

MSE_lasso_textual_4 = mean((lasso_pred_textual_4 - target_concept_t_test_4)^2)
print (paste0("MSE-Lasso for test set 4: ",MSE_lasso_textual_4))
