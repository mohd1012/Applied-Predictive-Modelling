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
library(psych)
# Lots of code from Kuhn, M., & Johnson, K. (2013).
# Applied Predicive Modelling. New York, USA, USA: Springer.

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
pairs.panels(data_set)
corrplot(cor(data_set))

Set_seed_seed <- 100
# Create test and training sets
set.seed(Set_seed_seed)
inTrain <- createDataPartition(data_set$price, p = .8)[[1]]
data_set_train <- data_set[ inTrain, ]
data_set_test  <- data_set[-inTrain, ]

# Setup common caret parameters
training_control <- trainControl(method = "cv")


# Fit linear model
set.seed(Set_seed_seed)
linear_model <- train(formula, 
                      data = data_set_train,
                      method = "lm", 
                      trControl = training_control)

# Fit second order linear model
formula_poly <- price ~ poly(carat, depth, table, x, y, z, degree = 2)

set.seed(Set_seed_seed)
polynomial_model <- train(formula_poly, 
                          data = data_set_train,
                          method = "lm", 
                          trControl = training_control)

# Fit MARS model (Multivariate Adaptive Regression)
set.seed(Set_seed_seed)
mars_model <- train(formula, 
                    data = data_set_train,
                    method = "earth",
                    tuneGrid = expand.grid(degree = 1, nprune = 2:38),
                    trControl = training_control)

# Set up SVM tunning grid
sigDist <- sigest(price ~ ., data = data_set_train, frac = 1)
svmTuneGrid <- data.frame(sigma = as.vector(sigDist)[1], C = 2 ^ (-2:7))

# Fit SVM model
# classProbs = TRUE was added since the text was written
set.seed(Set_seed_seed)
svm_model <- train(formula,
                   data = data_set_train,
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   tuneGrid = svmTuneGrid,
                   trControl = training_control)

# Fit glm
set.seed(Set_seed_seed)
glm_model <- train(formula,
                   data = data_set_train,
                   method = "glm",
                   trControl = training_control)

# Fit partial least squares
set.seed(Set_seed_seed)
tune_grid <- expand.grid(ncomp = 1:(ncol(data_set_train) - 2))
pls_model <- train(x = data_set_train[, -1], y = data_set_train[, 1],
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
set.seed(Set_seed_seed)
ridge_model <- train(formula, 
                     data = data_set_train,
                     method = "ridge",
                     tuneGrid = ridgeGrid,
                     trControl = training_control,
                     preProc = c("center", "scale"))

# Fit enet model
enetGrid <- expand.grid(lambda = c(0, 0.01, .1), 
                        fraction = seq(.05, 1, length = 20))
set.seed(Set_seed_seed)
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

set.seed(Set_seed_seed)
nnet_model <- train(formula, 
                    data = data_set_train,
                    method = "avNNet",
                    tuneGrid = nnetGrid,
                    trControl = training_control,
                    preProc = c("center", "scale"),
                    linout = TRUE,
                    trace = FALSE,
                    MaxNWts = 13 * (ncol(data_set_train[,-1]) + 1) + 13 + 1,
                    maxit = Set_seed_seed0,
                    allowParallel = FALSE)

# knnet
set.seed(Set_seed_seed)
knn_model <- train(formula, 
                   data = data_set_train,
                   method = "knn",
                   preProc = c("center", "scale"),
                   tuneGrid = data.frame(k = 1:20),
                   trControl = training_control)

# make rpart model
set.seed(Set_seed_seed)
rpart_model <- train(formula, 
                     data = data_set_train,
                     method = "rpart",
                     tuneLength = 25,
                     trControl = training_control)


# Make conditional inference tree
cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))
set.seed(Set_seed_seed)
ctree_model <- train(formula, 
                     data = data_set_train,
                     method = "ctree",
                     tuneGrid = cGrid,
                     trControl = training_control)

# Make m5 model
set.seed(Set_seed_seed)
m5_model <- train(formula, 
                  data = data_set_train,
                  method = "M5",
                  trControl = training_control,
                  control = Weka_control(M = 10))


# Make bagged tree model
set.seed(Set_seed_seed)
treebag_model <- train(formula, 
                       data = data_set_train,
                       method = "treebag",
                       nbagg = 50,
                       trControl = training_control)

# Make random forests model
mtryGrid <- data.frame(mtry = floor(seq(10, ncol(data_set_train) - 1, length = 10)))

set.seed(Set_seed_seed)
rf_model <- train(formula, 
                  data = data_set_train,
                  method = "rf",
                  tuneGrid = mtryGrid,
                  ntree = Set_seed_seed0,
                  importance = TRUE,
                  trControl = training_control)

# Boosted Trees
# Added n.minobsinnode = 10 to expand grid. Seems this is now required, and 10 is a good default
# but didn't find any reasonable limits so fixed at 10.

mtryGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                        n.trees = seq(100, 1000, by = 50),
                        n.minobsinnode = 10,
                        shrinkage = c(0.01, 0.1))

set.seed(Set_seed_seed)
gbm_model <- train(formula, 
                   data = data_set_train,
                   method = "gbm",
                   tuneGrid = mtryGrid,
                   verbose = FALSE,
                   trControl = training_control)

# Cubist
cbGrid <- expand.grid(committees = c(1:10, 20, 50, 75, 100), 
                      neighbors = c(0, 1, 5, 9))

set.seed(Set_seed_seed)
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

# Examine predictor contributions across all models.
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

x <- var_imp_table_wide
rownames(x) <- var_imp_table_wide$model
x <- x[, -1]
heat_map_matrix <- as.matrix(x)
heatmap(heat_map_matrix)

# Find the number of clusters. Modification of a stackoverflow posting replacing loop with sapply
# Ben. (2013, March 13). Cluster analysis in R: determine the optimal number of clusters.
# Retrieved July 20, 2015, from stackoverflow:
# https://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters

which_cluster <- function(mydata, i) {
  sum(kmeans(mydata, centers = i)$withinss)
}
mydata <- x
max_clusters <- 9
wss <- (nrow(mydata) - 1)*sum(apply(mydata,2,var))
wss[2:max_clusters] <- sapply(2:max_clusters,  FUN = which_cluster, mydata = mydata)

plot(1:max_clusters, wss, type = "b", xlab = "Number of Clusters",
     ylab = "Within groups sum of squares")
y <- kmeans(x, 4)
y <- as.data.frame(as.factor(y$cluster))
colnames(y) <- "cluster"

# Find the principal components
trans <- preProcess(x, 
                   method = "pca")
PC <- predict(trans, x)
# Add to matrix
x <- cbind(x, PC[,1:2])
x <- cbind(x, y)

p <- ggplot(var_imp_table_long, aes(predictor, model))
p <- p + geom_tile(aes(fill = ranking), colour = "white")
p <- p + scale_fill_gradient(low = "white", high = "steelblue")
p <- p + theme(panel.background = element_rect(fill = 'white'),
               panel.grid.major = element_blank(),
               panel.border = element_blank())
p

p <- ggplot(data = x, aes(x = PC1, y = PC2, colour = cluster))
p <- p + geom_point(size = 5)
p <- p + scale_colour_brewer(palette = "Set1")
p <- p + ggtitle("Feature contribution by kmeans clustering")
# p <- p + geom_text(label = rownames(x))
p
# 
# avNNet, ctree, enet, knn, M5, pcr, ridge, svmRadial all have the same predictor contributions, PC1, PC2, cluster
# varImp(enet_model) of all the models gives duplicate results.
# Something up with the caret call?


