pkgs <- list( 'AppliedPredictiveModeling', 'ggplot2', 'lattice', 'nlme', 'mgcv', 
             'caret', 'plotrix', 'TeachingDemos', 'plotmo', 'e1071',  'earth',
             'kernlab',  'corrplot', 'pls', 'lars', 'elasticnet', 'nnet', 'rpart',
             'grid', 'partykit', 'RWeka', 'plyr', 'ipred', 'randomForest',  'survival',
             'splines',  'parallel',  'gbm',  'Cubist', 'party', 'tidyr', 'psych',
             'gam', 'GGally', 'reshape2', 'MASS', 'VIM')
invisible(lapply(pkgs, library, character.only = T))
# To do -
# pls call needs to be formula or have better column selection
# replace ncol with numbers of variables in formula
# variable contributions doesn't seem to be working for GAMlowess
# check through all generated data frames to see if I can combine any of them

# Following segment is where you customise for a particular dataset
# Assumes in some models, such as PLS and PCR, that data_set only contains
# features and response. Otherwise, ncol calcs in several grid.expand calls will
# be wrong. PCR doesn't yet take a formula call, so need to customise

data_set <- diamonds[, c(1, 5, 6, 7, 8, 9, 10)]
data_set <- data_set[1:1000,]
response <- "price"
response_col <- grep(pattern = response, x = colnames(data_set))
formula <- price ~ carat + depth + table + x + y + z
formula_poly <- price ~ poly(carat, depth, table, x, y, z, degree = 2)

# Examine data

# check atleast min and skew
plot_summary <- function(data_set, func, title) {
  summary_stat <- adply(.data = data_set, .margins = 2, .fun = func)
  colnames(summary_stat) <- c("feature", "summary_stat")
  p <- ggplot(data = summary_stat, aes(x = feature, y = summary_stat))
  p <- p + geom_bar(stat = "identity")
  p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
  p <- p + ggtitle(title)
  p
}

p_min <- plot_summary(data_set, min, "min")
p_skew <- plot_summary(data_set, skew, "skew")
p_na <- plot_summary(data_set, function(x) sum(is.na(x)/length(x)), "% missing")

p_min;p_skew;p_na

p_missing <- aggr(data_set, col = c('navyblue','red'), numbers = TRUE, sortVars = TRUE, labels = names(data), cex.axis = .7, gap = 3, ylab = c("Histogram of missing data","Pattern"))

colnames(data_set)[nearZeroVar(data_set)]

feature_plot <- function(data_set, response) {
  mtmelt <<- melt(data_set, id = response)
  p <- ggplot(mtmelt, aes(x = value, y = mtmelt[, 1])) +
    facet_wrap(~variable, scales = "free") +
    geom_point() +
    labs(y = response)
  p
}
p_feature <- feature_plot(data_set, response)
p_feature
rm(mtmelt)
# alternative: featurePlot(x = data_set[,-response_col], y = data_set[,response_col])

Set_seed_seed <- 100
# Create test and training sets

set.seed(Set_seed_seed)
inTrain <- createDataPartition(data_set[, response_col], p = .8)[[1]]
data_set_train <- data_set[ inTrain, ]
data_set_test  <- data_set[-inTrain, ]

linear_model <- NULL
polynomial_model <- NULL
mars_model <- NULL
svm_model <- NULL
glm_model <- NULL
pls_model <- NULL
pcr_model <- NULL
ridge_model <- NULL
enet_model <- NULL
nnet_model <- NULL
knn_model <- NULL
rpart_model <- NULL
ctree_model <- NULL
m5_model <- NULL
treebag_model <- NULL
rf_model <- NULL
gbm_model <- NULL
cubist_model <- NULL
GAM_model <- NULL

model_list <- list(linear_model = linear_model, polynomial_model = polynomial_model,
                   mars_model = mars_model, svm_model = svm_model, glm_model = glm_model,
                   pls_model = pls_model, pcr_model = pcr_model, ridge_model = ridge_model, 
                   enet_model = enet_model, nnet_model = nnet_model, knn_model = knn_model,
                   rpart_model = rpart_model, ctree_model = ctree_model, m5_model = m5_model,
                   treebag_model = treebag_model, rf_model = rf_model, gbm_model = gbm_model,
                   cubist_model = cubist_model, GAM_model = GAM_model)

rm(linear_model, polynomial_model, mars_model, svm_model, glm_model, pls_model, pcr_model, ridge_model, enet_model, nnet_model, knn_model, rpart_model, ctree_model, m5_model, treebag_model, rf_model, gbm_model, cubist_model, GAM_model)

# Setup common caret parameters
training_control <- trainControl(method = "cv")

# Fit linear model
set.seed(Set_seed_seed)
model_list$linear_model <- train(formula,
                      # importance = TRUE,
                      data = data_set_train,
                      method = "lm", 
                      trControl = training_control)

# Fit second order linear model
set.seed(Set_seed_seed)
model_list$polynomial_model <- train(formula_poly,
                          # importance = TRUE,
                          data = data_set_train,
                          method = "lm", 
                          trControl = training_control)

# Fit MARS model (Multivariate Adaptive Regression)
tune_grid <- expand.grid(degree = 1, nprune = 2:38)
set.seed(Set_seed_seed)
model_list$mars_model <- train(formula,
                    data = data_set_train,
                    method = "earth",
                    tuneGrid = tune_grid,
                    trControl = training_control)

# Set up SVM tunning grid
sigDist <- sigest(x = formula, data = data_set_train, frac = 1)
tune_grid <- data.frame(sigma = as.vector(sigDist)[1], C = 2 ^ (-2:7))
rm(sigDist)
# Fit SVM model
# classProbs = TRUE was added since the text was written
set.seed(Set_seed_seed)
model_list$svm_model <- train(formula,
                   importance = TRUE,
                   data = data_set_train,
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   tuneGrid = tune_grid,
                   trControl = training_control)

# Fit glm
set.seed(Set_seed_seed)
model_list$glm_model <- train(formula,
                   data = data_set_train,
                   method = "glm",
                   trControl = training_control)

# Fit partial least squares
set.seed(Set_seed_seed)
tune_grid <- expand.grid(ncomp = 1:(ncol(data_set_train) - 2))
model_list$pls_model <- train(x = data_set_train[, -response_col], y = unlist(data_set_train[, response_col]),
                   importance = TRUE,
                   method = "pls",
                   tuneGrid = tune_grid,
                   trControl = training_control)

# Fit principle components regression
tune_grid <- expand.grid(ncomp = 1:min(ncol(data_set_train) - 2, 35))
model_list$pcr_model <- train(formula,
                   importance = TRUE,
                   data = data_set_train,
                   method = "pcr",
                   tuneGrid = tune_grid,
                   trControl = training_control)

# Fit ridge regression
tune_grid <- expand.grid(lambda = seq(0, .1, length = 15))
set.seed(Set_seed_seed)
model_list$ridge_model <- train(formula,
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
model_list$enet_model <- train(formula,
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
model_list$nnet_model <- train(formula,
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
tune_grid <- data.frame(k = 1:20)
set.seed(Set_seed_seed)
model_list$knn_model <- train(formula,
                   importance = TRUE,
                   data = data_set_train,
                   method = "knn",
                   preProc = c("center", "scale"),
                   tuneGrid = tune_grid,
                   trControl = training_control)

# make rpart model
set.seed(Set_seed_seed)
model_list$rpart_model <- train(formula,
                     data = data_set_train,
                     method = "rpart",
                     tuneLength = 25,
                     trControl = training_control)


# Make conditional inference tree
tune_grid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 2))))
set.seed(Set_seed_seed)
model_list$ctree_model <- train(formula,
                     data = data_set_train,
                     method = "ctree",
                     tuneGrid = tune_grid,
                     trControl = training_control)

# Make m5 model
set.seed(Set_seed_seed)
model_list$m5_model <- train(formula,
                  data = data_set_train,
                  method = "M5",
                  trControl = training_control,
                  control = Weka_control(M = 10))


# Make bagged tree model
set.seed(Set_seed_seed)
model_list$treebag_model <- train(formula,
                       importance = TRUE,
                       data = data_set_train,
                       method = "treebag",
                       nbagg = 50,
                       trControl = training_control)

# Make random forests model
tune_grid <- data.frame(mtry = floor(seq(10, ncol(data_set_train) - 1, length = 10)))

set.seed(Set_seed_seed)
model_list$rf_model <- train(formula,
#                 importance = TRUE,
                  data = data_set_train,
                  method = "rf",
                  tuneGrid = tune_grid,
                  ntree = Set_seed_seed,
                  importance = TRUE,
                  trControl = training_control)

# Boosted Trees
# Added n.minobsinnode = 10 to expand grid. Seems this is now required, and 10 is a good default
# but didn't find any reasonable limits so fixed at 10. However, with small data sets, 10 will
# to small, so set 10 or 1/6 of data set, whichever is smallest

tune_grid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                        n.trees = seq(100, 1000, by = 50),
                        n.minobsinnode = min(floor(dim(data_set_train)[1]/5), 10),
                        shrinkage = c(0.01, 0.1))

set.seed(Set_seed_seed)
model_list$gbm_model <- train(formula,
                   data = data_set_train,
                   method = "gbm",
                   tuneGrid = tune_grid,
                   verbose = FALSE,
                   trControl = training_control)

tune_grid <- expand.grid(committees = c(1:10, 20, 50, 75, 100), 
                       neighbors = c(0, 1, 5, 9))

set.seed(Set_seed_seed)
model_list$cubist_model <- train(formula, 
                      data = data_set_train,
                      method = "cubist",
                      tuneGrid = tune_grid,
                      trControl = training_control)

tune_grid <- expand.grid(span = seq(0.1, 0.9, length = 9), degree = 1)

set.seed(Set_seed_seed)
model_list$GAM_model <- train(formula,
                   data = data_set,
                   method = "gamLoess", 
                   tuneGrid = tune_grid,
                   trControl = training_control
)

# Plots to compare the models
resamp <- resamples(model_list)
summary(resamp)

resamp_plot <- function (resamp) {
  resamp_df <- resamp$values
  resamp_df <- resamp_df[, grep("Rsquared", colnames(resamp_df))]
  resamp_df <- gather(resamp_df)
  colnames(resamp_df) <- c("method", "R2")
  resamp_df$method <- unlist(strsplit(split = "~", as.character(resamp_df$method)))[seq(from =  1, to = 2*length(resamp_df$method), by = 2)]
  resamp_df$method <- as.factor(resamp_df$method)
  summary <- ddply(.data = resamp_df, .variables = c("method"), summarise, median(R2))
  colnames(summary) <- c("method", "median")
  resamp_df <- merge(resamp_df, summary, by = "method")
  resamp_df$method <- factor(resamp_df$method,levels(resamp_df$method)[order(summary$median)])
  
  p <- ggplot(data = resamp_df, aes(x = method, y = R2))
  p <- p + geom_boxplot()
  p <- p + geom_point()
  p <- p + theme(axis.text.x = element_text(angle = -90))
  p <- p + coord_cartesian(ylim = c(min(resamp_df$R2), 1.0)) 
  p <- p + ggtitle("")
  p
}
p_resamp <- resamp_plot(resamp)
p_resamp

# parallelplot(resamp, metric = "Rsquared")
# bwplot(resamp, metric = "Rsquared")
# dotplot(resamp, metric = "Rsquared")

# Code for R2 vs model size
model_size <- ldply(model_list, object.size)
size_and_fit <- resamp$values[seq(3, length(resamp$values), 2)]
size_and_fit <- apply(size_and_fit, 2, mean)
size_and_fit <- cbind(size_and_fit, model_size)
colnames(size_and_fit) <- c("R2", "model", "size")
rownames(size_and_fit) <- size_and_fit$model

p <- ggplot(data = size_and_fit, aes(x = R2, y = size, label = model))
p <- p + geom_point()
p <- p + geom_text()
p <- p + coord_trans(y = "log2")
p <- p + ggtitle("Which models are efficient at giving good predictions?")
p_model_size <- p
p_model_size

#########################################################################################
# Examining fits for rf_model



model_results <- data.frame(obs = data_set[, response_col],
                            prediction = predict(model_list$rf_model, data_set), order = 1:dim(data_set)[1])

# Prediction vs observation order
p <- ggplot(data = model_results, aes(x = order, y = prediction))
p <- p + geom_point()
p <- p + ggtitle("Prediction vs observation order") + labs(x = "order", y = "prediction")
p_pred_vs_order <- p
p_pred_vs_order

# Residuals vs order
p <- ggplot(data = model_results, aes(x = order, y = model_results$prediction - data_set[, response_col]))
p <- p + geom_point()
p <- p + ggtitle("Residuals vs order") + labs(x = "order", y = "residual")
p_resid_vs_order <- p
p_resid_vs_order

# Predicted vs fitted values
p <- ggplot(data = model_results, aes(x = obs, y = prediction))
p <- p + geom_point()
p <- p + ggtitle("Predicted vs fitted values") + labs(x = "obs", y = "prediction")
p_pred_vs_fitted <- p
p_pred_vs_fitted

# Correlation of predicted vs feature
model_cors <- as.data.frame(t(cor(model_results$prediction, data_set[, -response_col])))
model_cors$feature <- rownames(model_cors)
colnames(model_cors) <- c("cor", "feature")
p <- ggplot(data = model_cors, aes(x = feature, y = cor))
p <- p + geom_bar(stat = "identity")
p <- p + ggtitle("Correlation of predictions with feature") + labs(x = "feature", y = "correlation")
p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
p_pred_vs_feature <- p
p_pred_vs_feature 

# Correlation of residuals vs features
model_cors$resids <- t(cor(model_results$prediction - model_results$obs, data_set[, -response_col]))
p <- ggplot(data = model_cors, aes(x = feature, y = resids))
p <- p + geom_bar(stat = "identity")
p <- p + ggtitle("Correlation of residuals with feature") + labs(x = "feature", y = "correlation of residual")
p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
p_resid_corr_features <- p
p_resid_corr_features

# Residuals vs features
data_set_residuals <- data_set
data_set_residuals$residual <- data_set_residuals$price - predict(model_list$rf_model, data_set_residuals)

mtmelt <<- melt(data_set_residuals, id = "residual")
p <- ggplot(mtmelt, aes(x = value, y = residual)) +
  facet_wrap(~variable, scales = "free") +
  geom_point() +
  labs(y = "residual")
p_residuals_vs_features <- p
p_residuals_vs_features

# Residual distribution
p <- ggplot(data = model_results, aes(x = prediction - obs))
p <- p + geom_histogram()
p <- p + ggtitle("Histogram of residuals") + labs(x = "residuals", y = "count")
p_residual_dist <- p
p_residual_dist

# qq plot of model residuals
model_results$residuals <- model_results$obs - model_results$prediction
p <- ggplot(data = model_results, aes(sample = residuals))
p <- p + stat_qq()
p <- p + ggtitle("Model residuals")
p_residuals_qq <- p
p_residuals_qq

# Scale location (sqrt of standardize residuals vs fitted values)
model_results$scale_location <- sqrt(abs(scale(model_results$prediction - model_results$obs)))
p <- ggplot(data = model_results, aes(x = prediction, y = scale_location))
p <- p + geom_point()
p <- p + ggtitle("scale-location of residuals") + labs(x = "prediction", y = "sqrt(abs err)")
p_scale_location <- p
p_scale_location

# Still hacking hat and D code
mat_x <- as.matrix(data_set[, -response_col])
hat_x <- mat_x %*% ginv( t(mat_x) %*% mat_x) %*% t(mat_x)
diag_hat <- diag(hat_x)
D <- (model_results$prediction - model_results$obs) * diag_hat/(1 - (diag_hat %*% diag_hat))
D <- as.data.frame(D)
D$order <- 1:dim(D)[1]

p <- ggplot(data = D, aes(x = order, y = D))
p <- p + geom_point()
p <- p + ggtitle("D - Impact") + labs(x = "order", y = "D")
p

####################################################################################################
# Examine predictor contributions across all models.
# Make a long table for all the models and the contributions of the predictors

make_new_row <- function(model) {
  new_row <- varImp(model)$importance
  new_row[,2] <- new_row[,1]
  new_row[,1] <- row.names(new_row)
  new_row[,3] <- model$method
  colnames(new_row) <- c("predictor", "ranking", "model")
  row.names(new_row) <- NULL
  return(new_row)
}

make_var_imp_table_long <- function(model_list) {
  new_row <- NULL
  var_imp_table_long <- NULL
  var_imp_table_long <- ldply(model_list, make_new_row)
  var_imp_table_long <- var_imp_table_long[, 2:4]
  var_imp_table_long$model <- as.factor(var_imp_table_long$model)
# For polynomial, because formula is different than the other models, will need to pull out.
  var_imp_table_long <- var_imp_table_long[-grep("poly", var_imp_table_long$predictor), ]
  var_imp_table_long <- var_imp_table_long[var_imp_table_long$predictor != response,]
  var_imp_table_long$model <- as.factor(var_imp_table_long$model)
  return(var_imp_table_long)
}
var_imp_table_long <- make_var_imp_table_long(model_list)


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
plot(find_number_of_clusters(df = var_imp_table_wide[,-1]))
clusters <- kmeans(var_imp_table_wide[,-1], 4)
clusters <- as.data.frame(as.factor(clusters$cluster))
colnames(clusters) <- "cluster"

# Find the principal components

drop_col <- nearZeroVar(var_imp_table_wide)
trans <- preProcess(var_imp_table_wide[,-c(1, drop_col)], method = "pca")
PC <- predict(trans, var_imp_table_wide[,-c(1, drop_col)])
# Add to matrix
var_imp_table_wide <- cbind(var_imp_table_wide, PC[,1:2])
var_imp_table_wide <- cbind(var_imp_table_wide, clusters)
var_imp_table_wide <- var_imp_table_wide[order(var_imp_table_wide$cluster),]
# Join wide table clusters to long table
# look up model in wide, get cluster, add to long
var_imp_table_long <- merge(var_imp_table_wide[,c("model", "cluster")], var_imp_table_long, by = "model")
var_imp_table_long <- var_imp_table_long[order(var_imp_table_long$cluster),]
var_imp_table_long <- var_imp_table_long[order(var_imp_table_long$predictor),]
with(var_imp_table_long,
     model <- factor(model, levels = model[order(cluster,model)], ordered = TRUE))

p <- ggplot(data = var_imp_table_long, aes(x = predictor, y = ranking))
p <- p + geom_bar(stat = "identity", position = "dodge") + facet_wrap(~cluster)
p <- p + ggtitle("feature contributions to clusters")
p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
p_feature_contributions_to_clusters <- p
p_feature_contributions_to_clusters

p <- ggplot(var_imp_table_long, aes(x = predictor, y = model, cluster))
p <- p + geom_tile(aes(fill = ranking), colour = "white")
p <- p + scale_fill_gradient(low = "white", high = "steelblue")
p <- p + theme(panel.background = element_rect(fill = 'white'),
               panel.grid.major = element_blank(),
               panel.border = element_blank())
p <- p + ggtitle("ctree, avNNet, enet, knn, M5, pcr, ridge, svmRadial use a model free method")
p_var_contributions <- p
p_var_contributions

p <- ggplot(data = var_imp_table_wide, aes(x = PC1, y = PC2, colour = cluster, label = rownames(var_imp_table_wide)))
p <- p + geom_point(size = 5)
p <- p + scale_colour_brewer(palette = "Set1")
p <- p + ggtitle("Feature contribution by kmeans clustering")
p <- p + geom_text(position = position_jitter(height = 0.5, width = 0.5), hjust = 0, vjust = -1)
p_feature_contributions_kmeans <- p
p_feature_contributions_kmeans



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

# Plot all plots in one go
plot_names <- ls()[ldply(.data = ls(), .fun = function(x) substr(x, 1, 2)) == "p_"]
ldply(.data = plot_names, .fun = function(x) plot(get(x)))
