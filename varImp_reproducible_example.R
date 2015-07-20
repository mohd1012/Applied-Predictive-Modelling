library(ggplot2)
library(caret)
library(elasticnet)
library(party)

data_set <- diamonds[1:1000, c(1, 5, 6, 7, 8, 9, 10)]
formula <- price ~ carat + depth + table + x + y + z

set.seed(100)
enet_model <- train(formula, 
                    data = data_set,
                    method = "enet",
                    trControl = trainControl(method = "cv"),
                    preProc = c("center", "scale"))

set.seed(100)
ctree_model <- train(formula, 
                     data = data_set,
                     method = "ctree",
                     trControl = trainControl(method = "cv"))

varImp(enet_model)
varImp(ctree_model)

# though different models, both give same varImp
# vNNet, ctree, enet, knn, M5, pcr, ridge, svmRadial all give the same variable contributions

