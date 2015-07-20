library(ggplot2)
library(caret)
library(elasticnet)
library(party)

data_set <- diamonds[1:1000, c(1, 5, 6, 7, 8, 9, 10)]
formula <- price ~ carat + depth + table + x + y + z

enetGrid <- expand.grid(lambda = c(0, 0.01, .1), 
                        fraction = seq(.05, 1, length = 20))

set.seed(100)
enet_model <- train(formula, 
                    data = data_set,
                    method = "enet",
                    tuneGrid = enetGrid,
                    trControl = trainControl(method = "cv"),
                    preProc = c("center", "scale"))

cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))

set.seed(100)
ctree_model <- train(formula, 
                     data = data_set,
                     method = "ctree",
                     tuneGrid = cGrid,
                     trControl = trainControl(method = "cv"))

varImp(enet_model)
varImp(ctree_model)

# though different models, both give same varImp
# vNNet, ctree, enet, knn, M5, pcr, ridge, svmRadial all give the same variable contributions

