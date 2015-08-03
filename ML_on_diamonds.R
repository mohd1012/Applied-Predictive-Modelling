library(AppliedPredictiveModeling)
library(ggplot2)
library(lattice)
library(nlme)
library(mgcv)
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
library(gam)

# To do -
# Replace wides with longs
# in train control add "allowParallel = FALSE"
# replace all plots with ggplot2

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
                      importance = TRUE,
                      data = data_set_train,
                      method = "lm", 
                      trControl = training_control)

# Fit second order linear model
formula_poly <- price ~ poly(carat, depth, table, x, y, z, degree = 2)

set.seed(Set_seed_seed)
polynomial_model <- train(formula_poly,
                          importance = TRUE,
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
                   importance = TRUE,
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
                   importance = TRUE,
                   method = "pls",
                   tuneGrid = tune_grid,
                   trControl = training_control)

# Fit principle components regression
tune_grid <- expand.grid(ncomp = 1:min(ncol(data_set_train) - 2, 35))
pcr_model <- train(formula,
                   importance = TRUE,
                   data = data_set_train,
                   method = "pcr",
                   tuneGrid = tune_grid,
                   trControl = training_control)

# Fit ridge regression
tune_grid <- expand.grid(lambda = seq(0, .1, length = 15))
set.seed(Set_seed_seed)
ridge_model <- train(formula,
                     importance = TRUE,
                     data = data_set_train,
                     method = "ridge",
                     tuneGrid = tune_grid,
                     trControl = training_control,
                     preProc = c("center", "scale"))

# Fit enet model
tune_grid <- expand.grid(lambda = c(0, 0.01, .1), 
                        fraction = seq(.05, 1, length = 20))
set.seed(Set_seed_seed)
enet_model <- train(formula,
                    importance = TRUE,
                    data = data_set_train,
                    method = "enet",
                    tuneGrid = tune_grid,
                    trControl = training_control,
                    preProc = c("center", "scale"))

# Fit neural net
tune_grid <- expand.grid(decay = c(0, 0.01, .1), 
                        size = c(1, 3, 5, 7, 9, 11, 13), 
                        bag = FALSE)

set.seed(Set_seed_seed)
nnet_model <- train(formula,
                    importance = TRUE,
                    data = data_set_train,
                    method = "avNNet",
                    tuneGrid = tune_grid,
                    trControl = training_control,
                    preProc = c("center", "scale"),
                    linout = TRUE,
                    trace = FALSE,
                    MaxNWts = 13 * (ncol(data_set_train[,-1]) + 1) + 13 + 1,
                    maxit = 100,
                    allowParallel = FALSE)

# knnet
set.seed(Set_seed_seed)
knn_model <- train(formula,
                   importance = TRUE,
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
tune_grid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))
set.seed(Set_seed_seed)
ctree_model <- train(formula,
                     data = data_set_train,
                     method = "ctree",
                     tuneGrid = tune_grid,
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
                       importance = TRUE,
                       data = data_set_train,
                       method = "treebag",
                       nbagg = 50,
                       trControl = training_control)

# Make random forests model
tune_grid <- data.frame(mtry = floor(seq(10, ncol(data_set_train) - 1, length = 10)))

set.seed(Set_seed_seed)
rf_model <- train(formula,
#                 importance = TRUE,
                  data = data_set_train,
                  method = "rf",
                  tuneGrid = tune_grid,
                  ntree = Set_seed_seed,
                  importance = TRUE,
                  trControl = training_control)

# Boosted Trees
# Added n.minobsinnode = 10 to expand grid. Seems this is now required, and 10 is a good default
# but didn't find any reasonable limits so fixed at 10.

tune_grid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                        n.trees = seq(100, 1000, by = 50),
                        n.minobsinnode = 10,
                        shrinkage = c(0.01, 0.1))

set.seed(Set_seed_seed)
gbm_model <- train(formula,
                   data = data_set_train,
                   method = "gbm",
                   tuneGrid = tune_grid,
                   verbose = FALSE,
                   trControl = training_control)

# Cubist
tune_grid <- expand.grid(committees = c(1:10, 20, 50, 75, 100), 
                      neighbors = c(0, 1, 5, 9))

set.seed(Set_seed_seed)
cubist_model <- train(formula, 
                      data = data_set_train,
                      method = "cubist",
                      tuneGrid = tune_grid,
                      trControl = training_control)

# largely Matt's code from here on.
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
bwplot(resamp, metric = "Rsquared")
dotplot(resamp, metric = "Rsquared")

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
new_row <- NULL
var_imp_table <- NULL
for (model in model_list) {
  new_row <- varImp(model)$importance
  new_row[,2] <- new_row[,1]
  new_row[,1] <- row.names(new_row)
  new_row[,3] <- model$method
  colnames(new_row) <- c("predictor", "ranking", "model")
  row.names(new_row) <- NULL
  var_imp_table <- rbind(var_imp_table, new_row)
}
rm(new_row)
# For polynomial, because formula is different than the other models, will need to pull out.
var_imp_table_long <- var_imp_table[-grep("poly", var_imp_table$predictor), ]
var_imp_table_long <- var_imp_table_long[var_imp_table_long$predictor != "price",]
var_imp_table_long$model <- as.factor(var_imp_table_long$model)
var_imp_table_wide <- spread(data = var_imp_table_long, key = predictor, value = ranking)
# Some models had terms not in other models, so spread fills wide table entries with NA
var_imp_table_wide[is.na(var_imp_table_wide)] <- 0.0

rownames(var_imp_table_wide) <- var_imp_table_wide$model
heat_map_matrix <- as.matrix(var_imp_table_wide[, -1])
heatmap(heat_map_matrix, main = "ctree, avNNet, enet, knn, M5, pcr, ridge, svmRadial have model free impacts")

# Note that ctree, avNNet, enet, knn, M5, pcr, ridge, svmRadial do not have
# variable impacts defined, so use a model free method and will all report
# the same impacts. rpart, pls, lm, glm, earth, gbm, treebag, cubist and 
# rf have defined methods.

# Find the number of clusters.
# Modification of a stackoverflow posting replacing loop with sapply
# and converting to ggplot2
# Ben. (2013, March 13). Cluster analysis in R: determine the optimal number of clusters.
# Retrieved July 20, 2015, from stackoverflow:
# https://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters

find_number_of_clusters <- function(df) {
  which_cluster <- function(mydata, i) {
    sum(kmeans(mydata, centers = i)$withinss)
  }
  mydata <- var_imp_table_wide[,-1]
  max_clusters <- 9
  wss <- (nrow(mydata) - 1)*sum(apply(mydata,2,var))
  wss[2:max_clusters] <- sapply(2:max_clusters,  FUN = which_cluster, mydata = mydata)
  wss <- cbind(1:max_clusters, wss)
  colnames(wss) <- c("Number of clusters", "Within group SS")
  wss <- as.data.frame(wss)
  wss <<- wss
  p <- ggplot(data = wss, aes(x = wss$`Number of clusters`, y = wss$`Within group SS`))
  p <- p + geom_point()
  return(p)
}
df <- var_imp_table_wide[,-1]
plot(find_number_of_clusters(df = df))
clusters <- kmeans(var_imp_table_wide[,-1], 4)

# barplot(t(clusters$centers)), beside = TRUE, xlab = "cluster", ylab = "value")
# build code to do ggplot2 version of barplot. Not checked yet.

clusters <- as.data.frame(as.factor(clusters$cluster))
colnames(clusters) <- "cluster"

# Find the principal components
trans <- preProcess(var_imp_table_wide[,-1], method = "pca")
PC <- predict(trans, var_imp_table_wide[,-1])
# Add to matrix
var_imp_table_wide <- cbind(var_imp_table_wide, PC[,1:2])
var_imp_table_wide <- cbind(var_imp_table_wide, clusters)
var_imp_table_wide <- var_imp_table_wide[order(var_imp_table_wide$cluster),]
# Join wide table clusters to long table
# look up model in wide, get cluster, add to long
var_imp_table_long <- merge(var_imp_table_wide[,c("model", "cluster")], var_imp_table_long, by = "model")
var_imp_table_long <- var_imp_table_long[order(var_imp_table_long$cluster),]
var_imp_table_long <- var_imp_table_long[order(var_imp_table_long$predictor),]
p <- ggplot(data = var_imp_table_long, aes(x = predictor, y = ranking))
p <- p + geom_bar(stat = "identity", position = "dodge") + facet_wrap(~cluster)
p <- p + ggtitle("feature contributions to clusters")
p

with(var_imp_table_long,
     model <- factor(model, levels = model[order(cluster,model)], ordered = TRUE))

# maybe use with for above line to shorten it
p <- ggplot(var_imp_table_long, aes(x = predictor, y = model, cluster))
p <- p + geom_tile(aes(fill = ranking), colour = "white")
p <- p + scale_fill_gradient(low = "white", high = "steelblue")
p <- p + theme(panel.background = element_rect(fill = 'white'),
               panel.grid.major = element_blank(),
               panel.border = element_blank())
p <- p + ggtitle("ctree, avNNet, enet, knn, M5, pcr, ridge, svmRadial use a model free method")
p

p <- ggplot(data = var_imp_table_wide, aes(x = PC1, y = PC2, colour = cluster, label = rownames(var_imp_table_wide)))
p <- p + geom_point(size = 5)
p <- p + scale_colour_brewer(palette = "Set1")
p <- p + ggtitle("Feature contribution by kmeans clustering")
p <- p + geom_text(position = position_jitter(height = 0.5, width = 0.5), hjust = 0, vjust = -1)
p

# code for feature selection. Methods supported are below:
  # caretFuncs - exist for
  # lmFuncs
  # rfFuncs
  # treebagFuncs
  # ldaFuncs
  # nbFuncs
  # gamFuncs
  # lrFuncs

feature_selection_control <- rfeControl(functions = rfFuncs)
ctrl <- rfeControl(functions = lmFuncs,
                   method = training_control,
                   repeats = 5,
                   verbose = FALSE)

rf_features <- rfe(formula, data = data_set_train,
                 rfeControl = feature_selection_control)




