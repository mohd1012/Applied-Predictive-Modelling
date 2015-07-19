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
library(tidyr)

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

# Setup common caret parameters
training_control <- trainControl(method = "cv")


# Fit linear model
set.seed(100)
linear_model <- train(formula, 
                      data = data_set_train,
                      method = "lm", 
                      trControl = training_control)

# Fit second order linear model
formula_poly <- price ~ poly(carat, depth, table, x, y, z, degree = 2)

set.seed(100)
polynomial_model <- train(formula_poly, 
                          data = data_set_train,
                          method = "lm", 
                          trControl = training_control)

# Fit MARS model (Multivariate Adaptive Regression)
set.seed(100)
mars_model <- train(formula, 
                    data = data_set_train,
                    method = "earth",
                    tuneGrid = expand.grid(degree = 1, nprune = 2:38),
                    trControl = training_control)

# Set up SVM tunning grid
set.seed(100)
sigDist <- sigest(price ~ ., data = data_set_train, frac = 1)
svmTuneGrid <- data.frame(sigma = as.vector(sigDist)[1], C = 2 ^ (-2:7))

# Fit SVM model
# classProbs = TRUE was added since the text was written
set.seed(100)
svm_model <- train(price ~ .,
                   data = data_set_train,
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   tuneGrid = svmTuneGrid,
                   trControl = training_control)

# Fit glm
set.seed(100)
glm_model <- train(price ~ .,
                   data = data_set_train,
                   method = "glm",
                   trControl = training_control)

# Fit partial least squares
set.seed(100)
tune_grid <- expand.grid(ncomp = 1:(ncol(data_set_train) - 2))
pls_model <- train(x = data_set[, -1], y = data_set[, 1],
                   method = "pls",
                   tuneGrid = tune_grid,
                   trControl = training_control)

# Fit principle components regression
tune_grid <- expand.grid(ncomp = 1:min(ncol(data_set_train) - 2, 35))
pcr_model <- train(formula, 
                   data = data_set_train,
                   method = "pcr",
                   tuneGrid = tune_grid,
                   trControl = training_control)

# Fit ridge regression
ridgeGrid <- expand.grid(lambda = seq(0, .1, length = 15))
set.seed(100)
ridge_model <- train(formula, 
                     data = data_set_train,
                     method = "ridge",
                     tuneGrid = ridgeGrid,
                     trControl = training_control,
                     preProc = c("center", "scale"))

# Fit enet model
enetGrid <- expand.grid(lambda = c(0, 0.01, .1), 
                        fraction = seq(.05, 1, length = 20))
set.seed(100)
enet_model <- train(formula, 
                    data = data_set_train,
                    method = "enet",
                    tuneGrid = enetGrid,
                    trControl = training_control,
                    preProc = c("center", "scale"))

# Fit neural net
nnetGrid <- expand.grid(decay = c(0, 0.01, .1), 
                        size = c(1, 3, 5, 7, 9, 11, 13), 
                        bag = FALSE)

set.seed(100)
nnet_model <- train(formula, 
                    data = data_set_train,
                    method = "avNNet",
                    tuneGrid = nnetGrid,
                    trControl = training_control,
                    preProc = c("center", "scale"),
                    linout = TRUE,
                    trace = FALSE,
                    MaxNWts = 13 * (ncol(data_set_train[,-1]) + 1) + 13 + 1,
                    maxit = 1000,
                    allowParallel = FALSE)

# knnet
set.seed(100)
knn_model <- train(formula, 
                   data = data_set_train,
                   method = "knn",
                   preProc = c("center", "scale"),
                   tuneGrid = data.frame(k = 1:20),
                   trControl = training_control)

# make rpart model
set.seed(100)
rpart_model <- train(formula, 
                     data = data_set_train,
                     method = "rpart",
                     tuneLength = 25,
                     trControl = training_control)


# Make conditional inference tree
cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))
set.seed(100)
ctree_model <- train(formula, 
                     data = data_set_train,
                     method = "ctree",
                     tuneGrid = cGrid,
                     trControl = training_control)

# Make m5 model
set.seed(100)
m5_model <- train(formula, 
                  data = data_set_train,
                  method = "M5",
                  trControl = training_control,
                  control = Weka_control(M = 10))


# Make bagged tree model
set.seed(100)
treebag_model <- train(formula, 
                       data = data_set_train,
                       method = "treebag",
                       nbagg = 50,
                       trControl = training_control)

# Make random forests model
mtryGrid <- data.frame(mtry = floor(seq(10, ncol(data_set_train) - 1, length = 10)))

set.seed(100)
rf_model <- train(formula, 
                  data = data_set_train,
                  method = "rf",
                  tuneGrid = mtryGrid,
                  ntree = 1000,
                  importance = TRUE,
                  trControl = training_control)

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
                   trControl = training_control)

# Cubist
cbGrid <- expand.grid(committees = c(1:10, 20, 50, 75, 100), 
                      neighbors = c(0, 1, 5, 9))

set.seed(100)
cubist_model <- train(formula, 
                      data = data_set_train,
                      method = "cubist",
                      tuneGrid = cbGrid,
                      trControl = training_control)


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

# list of models setup
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

# Some customised model plots:
plot(rpart_model, scales = list(x = list(log = 10)))
# A line plot of the average performance. The 'scales' argument is actually an 
# argument to xyplot that converts the x-axis to log-2 units.
plot(svm_model, scales = list(x = list(log = 2)))
rpart_tree <- as.party(rpart_model$finalModel)
plot(rpart_tree)
plot(ctree_model$finalModel)
plot(cubist_model, auto.key = list(columns = 4, lines = TRUE))

# Additional code for M5
rule_fit <- M5Rules(formula, data = data_set_train, control = Weka_control(M = 10))
rule_fit

# Examining fits
model_results <- data.frame(obs = data_set[, 4],
                            Linear_Regression = predict(linear_model, data_set[,-4]))
plot(model_results)
model_prediction <- predict(gbm_model, data_set[,-4])

# Examine predictor contributions accross all models.
# Make a long table for all the models and the contributions of the predictors
y <- NULL
var_imp_table <- NULL
for (model in model_list) {
  y <- varImp(model)$importance
  y[,2] <- y[,1]
  y[,1] <- row.names(y)
  y[,3] <- model$method
  colnames(y) <- c("predictor", "ranking", "model")
  row.names(y) <- NULL
  var_imp_table <- rbind(var_imp_table, y)
}
rm(y)
# For polynomial, because formula is different than the other models, will need to pull out.
var_imp_table_long <- var_imp_table[-grep("poly", var_imp_table$predictor), ]
var_imp_table_long <- var_imp_table_long[var_imp_table_long$predictor != "price",]
var_imp_table_wide <- spread(data = var_imp_table_long, key = predictor, value = ranking)
# Some models had term not in other model, so spread fills wide table entries with NA
var_imp_table_wide[is.na(var_imp_table_wide)] <- 0.0

p <- ggplot(var_imp_table_long, aes(predictor, model))
p <- p + geom_tile(aes(fill = ranking), colour = "white")
p <- p + scale_fill_gradient(low = "white", high = "steelblue")
p
