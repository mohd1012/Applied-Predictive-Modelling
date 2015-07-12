library(AppliedPredictiveModeling)
library(ggplot2)
library(lattice)
library(caret)
library(plotrix)
library(TeachingDemos)
library(plotmo)
library(e1071)
library(earth)
library(kernlab)
library(corrplot)
library(pls)
library(lars)
library(elasticnet)
library(nnet)
library(rpart)
library(grid)
library(partykit)
library(RWeka)
library(plyr)
library(ipred)
library(randomForest)
library(survival)
library(splines)
library(parallel)
library(gbm)
library(Cubist)
library(party)

# create dataset and base model
# scriptLocation()
# C:\Program Files\R\R-3.2.1\library\AppliedPredictiveModeling\chapters
data_set <- diamonds[, c(1, 5, 6, 7, 8, 9, 10)]
data_set <- data_set[1:1000,]
formula <- price ~ carat + depth + table + x + y + z

# Examine data
apply(data_set, 2, skewness)
apply(data_set, 2, min)
nearZeroVar(data_set)
featurePlot(x = data_set[,-1], y = data_set[,1])
corrplot(cor(data_set))


# Create test and training sets
set.seed(100)
inTrain <- createDataPartition(data_set$price, p = .8)[[1]]
data_set_train <- data_set[ inTrain, ]
data_set_test  <- data_set[-inTrain, ]

# Fit linear model
set.seed(1)
linear_model <- train(formula, 
                data = data_set_train,
                method = "lm", 
                trControl = trainControl(method = "cv"))
linear_model

testResults <- data.frame(obs = data_set_train[,1],
                          Linear_Regression = predict(linear_model))
plot(testResults)

# Fit second order linear model
formula_poly <- price ~ poly(carat, depth, table, x, y, z, degree = 2)

polynomial_model <- train(formula_poly, 
                data = data_set_train,
                method = "lm", 
                trControl = trainControl(method = "cv"))
polynomial_model

# Fit MARS model (Multivariate Adaptive Regression)
set.seed(1)
mars_model <- train(formula, 
                 data = data_set_train,
                 method = "earth",
                 tuneGrid = expand.grid(degree = 1, nprune = 2:38),
                 trControl = trainControl(method = "cv"))
mars_model

# Set up SVM tunning grid
set.seed(231)
sigDist <- sigest(price ~ ., data = data_set_train, frac = 1)
svmTuneGrid <- data.frame(sigma = as.vector(sigDist)[1], C = 2 ^ (-2:7))

# Fit SVM model
# classProbs = TRUE was added since the text was written
set.seed(1056)
svm_model <- train(price ~ .,
                data = data_set_train,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneGrid = svmTuneGrid,
                trControl = trainControl(method = "cv"))
svm_model

# A line plot of the average performance. The 'scales' argument is actually an 
# argument to xyplot that converts the x-axis to log-2 units.
plot(svm_model, scales = list(x = list(log = 2)))

# Fit glm
set.seed(1056)
glm_model <- train(price ~ .,
                    data = data_set_train,
                    method = "glm",
                    trControl = trainControl(method = "cv"))
glm_model

# Fit partial least squares
set.seed(100)
tune_grid <- expand.grid(ncomp = 1:(ncol(data_set_train) - 2))
pls_model <- train(x = data_set[, -1], y = data_set[, 1],
                 method = "pls",
                 tuneGrid = tune_grid,
                 trControl = trainControl(method = "cv"))
pls_model

# Fit principle components regression
tune_grid <- expand.grid(ncomp = 1:min(ncol(data_set_train) - 2, 35))
pcr_model <- train(formula, 
                   data = data_set_train,
                   method = "pcr",
                   tuneGrid = tune_grid,
                   trControl = trainControl(method = "cv"))
pcr_model

# Fit ridge regression
ridgeGrid <- expand.grid(lambda = seq(0, .1, length = 15))
ridge_model <- train(formula, 
                    data = data_set_train,
                    method = "ridge",
                    tuneGrid = ridgeGrid,
                    trControl = trainControl(method = "cv"),
                    preProc = c("center", "scale"))
ridge_model

# Fit enet model
enetGrid <- expand.grid(lambda = c(0, 0.01, .1), 
                        fraction = seq(.05, 1, length = 20))
enet_model <- train(formula, 
                    data = data_set_train,
                    method = "enet",
                    tuneGrid = enetGrid,
                    trControl = trainControl(method = "cv"),
                    preProc = c("center", "scale"))
enet_model

# Fit neural net
nnetGrid <- expand.grid(decay = c(0, 0.01, .1), 
                        size = c(1, 3, 5, 7, 9, 11, 13), 
                        bag = FALSE)

set.seed(100)

nnet_model <- train(formula, 
                  data = data_set_train,
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = trainControl(method = "cv"),
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 13 * (ncol(data_set_train[,-1]) + 1) + 13 + 1,
                  maxit = 1000,
                  allowParallel = FALSE)
nnet_model

plot(nnet_model)

# knnet
knn_model <- train(formula, 
                  data = data_set_train,
                  method = "knn",
                  preProc = c("center", "scale"),
                  tuneGrid = data.frame(k = 1:20),
                  trControl = trainControl(method = "cv"))

knn_model

plot(knn_model)

# make rpart model
rpart_model <- train(formula, 
                    data = data_set_train,
                    method = "rpart",
                    tuneLength = 25,
                    trControl = trainControl(method = "cv"))

plot(rpart_model, scales = list(x = list(log = 10)))

rpart_imp <- varImp(rpart_model, scale = FALSE, competes = FALSE)
rpart_imp

rpart_tree <- as.party(rpart_model$finalModel)
plot(rpart_tree)

# Make conditional inference tree
cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))
ctree_model <- train(formula, 
                    data = data_set_train,
                    method = "ctree",
                    tuneGrid = cGrid,
                    trControl = trainControl(method = "cv"))
ctree_model
plot(ctree_model$finalModel)

# Make m5 model
m5_model <- train(formula, 
                  data = data_set_train,
                  method = "M5",
                  trControl = trainControl(method = "cv"),
                  control = Weka_control(M = 10))
m5_model

plot(m5_model)

rule_fit <- M5Rules(formula, data = data_set_train, control = Weka_control(M = 10))
rule_fit

# Make bagged tree model
treebag_model <- train(formula, 
                      data = data_set_train,
                      method = "treebag",
                      nbagg = 50,
                      trControl = trainControl(method = "cv"))

treebag_model

# Make random forests model
mtryGrid <- data.frame(mtry = floor(seq(10, ncol(data_set_train) - 1, length = 10)))

set.seed(100)
rf_model <- train(formula, 
                  data = data_set_train,
                  method = "rf",
                  tuneGrid = mtryGrid,
                  ntree = 1000,
                  importance = TRUE,
                  trControl = trainControl(method = "cv"))
rf_model
plot(rf_model)

rfImp <- varImp(rf_model, scale = FALSE)
rfImp


# Boosted Trees
# Added n.minobsinnode = 10 to expand grid. Seems this is now required, and 10 is a good default
# but didn't find any reasonable limits so fixed at 10.

mtryGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                        n.trees = seq(100, 1000, by = 50),
                        n.minobsinnode = 10,
                        shrinkage = c(0.01, 0.1))

set.seed(100)
gbm_model <- train(formula, 
                   data = data_set_train,
                   method = "gbm",
                   tuneGrid = mtryGrid,
                   verbose = FALSE,
                   trControl = trainControl(method = "cv"))
gbm_model

# Cubist
cbGrid <- expand.grid(committees = c(1:10, 20, 50, 75, 100), 
                      neighbors = c(0, 1, 5, 9))

set.seed(100)
cubist_model <- train(formula, 
                    data = data_set_train,
                    method = "cubist",
                    tuneGrid = cbGrid,
                    trControl = trainControl(method = "cv"))
cubist_model

plot(cubist_model, auto.key = list(columns = 4, lines = TRUE))

cbImp <- varImp(cubist_model, scale = FALSE)
cbImp


# make model comparisons

model_list <- list(linear_model = linear_model, polynomial_model = polynomial_model,
                   mars_model = mars_model, svm_model = svm_model, glm_model = glm_model,
                   pls_model = pls_model, pcr_model = pcr_model, ridge_model = ridge_model, 
                   enet_model = enet_model, nnet_model = nnet_model, knn_model = knn_model,
                   rpart_model = rpart_model, ctree_model = ctree_model, m5_model = m5_model,
                   treebag_model = treebag_model, rf_model = rf_model, gbm_model = gbm_model, cubist_model = cubist_model)

resamp <- resamples(model_list)
summary(resamp)
parallelplot(resamp, metric = "Rsquared")

# list of models setup. Couldn't get gbm working
linear_model
polynomial_model
mars_model
svm_model
glm_model
pls_model
pcr_model
ridge_model
enet_model
nnet_model
knn_model
rpart_model
ctree_model
m5_model
treebag_model
gbm_model
rf_model
cubist_model




model_results <- data.frame(obs = data_set[, 4],
                            Linear_Regression = predict(linear_model, data_set[,-4]))
plot(model_results)
model_prediction <- predict(gbm_model, data_set[,-4])

plot(rf_model)

varImp(rf_model)
importance_list <- lapply(model_list, varImp)
           
