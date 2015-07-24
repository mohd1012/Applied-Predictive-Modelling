library(ggplot2)
library(caret)
library(elasticnet)
library(party)

data_set <- diamonds[1:1000, c(1, 5, 6, 7, 8, 9, 10)]
formula <- price ~ carat + depth + table + x + y + z

set.seed(100)
enet_model <- train(formula,
                    importance = TRUE,
                    data = data_set,
                    method = "enet",
                    trControl = trainControl(method = "cv"),
                    preProc = c("center", "scale"))

set.seed(100)
ctree_model <- train(formula, 
                     data = data_set,
                     method = "ctree",
                     trControl = trainControl(method = "cv"))

set.seed(Set_seed_seed)
knn_model <- train(formula,
                   importance = TRUE,
                   data = data_set,
                   method = "knn",
                   preProc = c("center", "scale"),
                   tuneGrid = data.frame(k = 1:20),
                   trControl = training_control)

varImp(enet_model)
varImp(ctree_model)
varImp(knn_model)

# Though three different models, all give same varImp.
# In fact vNNet, ctree, enet, knn, M5, pcr, ridge, svmRadial all give the same variable contributions
# Some will take importance = TRUE as input: vNNet, enet, knn, pcr, ridge, svmRadial, rf all do.
# Others generated an error with importance = TRUE: ctree, M5, mars, glm, rpart, gbm
# (The error is "Something is wrong; all the RMSE metric values are missing:")
# 
# My question is why do different models give the same variable importance?
# This seems wrong, but I can't see what I've done wrong.

# Those methods don't have importance scores implemented so you get model-free measures. I can add one for enet based on the coefficients but knn and ctree have no obvious methods. 



