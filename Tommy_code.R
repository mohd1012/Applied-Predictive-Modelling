library(lattice)
library(ggplot2)
library(caret)
library(nnet)

# Mixture of code from Kuhn's Applied Predictive Modelling and a yammer posting by Tommy O'Dell

formula <- mpg ~ cyl + disp + hp + wt + gear

# Creat tuning grid
nnetGrid <- expand.grid(decay = c(0, 0.01, .1), 
                        size = c(1, 3, 5, 7, 9, 11, 13), 
                        bag = FALSE)
# Train with no preprocessing but with tuning
set.seed(100)
nnet_model_tune_nopreProc <- train(formula, 
                            data = mtcars,
                            method = "avNNet",
                            tuneGrid = nnetGrid,
                            trControl = trainControl(method = "cv"),
#                           preProc = c("center", "scale"),
                            linout = TRUE,
                            trace = FALSE,
                            MaxNWts = 13 * (ncol(mtcars[,-1]) + 1) + 13 + 1,
                            maxit = 1000,
                            allowParallel = FALSE)
nnet_model_tune_nopreProc

# Train with preprocessing and with tuning
set.seed(100)
nnet_model_preProc_tune <- train(formula, 
                    data = mtcars,
                    method = "avNNet",
                    tuneGrid = nnetGrid,
                    trControl = trainControl(method = "cv"),
                    preProc = c("center", "scale"),
                    linout = TRUE,
                    trace = FALSE,
                    MaxNWts = 13 * (ncol(mtcars[,-1]) + 1) + 13 + 1,
                    maxit = 1000,
                    allowParallel = FALSE)
nnet_model_preProc_tune

# Train without preprocessing or tuning
set.seed(100)
nnet_model_basic <- train(formula, 
                            data = mtcars,
                            method = "avNNet",
#                           tuneGrid = nnetGrid,
                            trControl = trainControl(method = "cv"),
#                           preProc = c("center", "scale"),
                            linout = TRUE,
                            trace = FALSE
#                           MaxNWts = 13 * (ncol(mtcars[,-1]) + 1) + 13 + 1,
#                           maxit = 1000,
#                           allowParallel = FALSE
                            )
nnet_model_basic

# Train with preprocessing but no tuning
set.seed(100)
nnet_model_preProc_notune <- train(formula, 
                          data = mtcars,
                          method = "avNNet",
#                         tuneGrid = nnetGrid,
                          trControl = trainControl(method = "cv"),
                          preProc = c("center", "scale"),
                          linout = TRUE,
                          trace = FALSE
#                         MaxNWts = 13 * (ncol(mtcars[,-1]) + 1) + 13 + 1,
#                         maxit = 1000,
#                         allowParallel = FALSE
)
nnet_model_preProc_notune

# Compare models
model_list <- list(nnet_model_basic = nnet_model_basic, 
                   nnet_model_tune_nopreProc = nnet_model_tune_nopreProc, 
                   nnet_model_preProc_notune = nnet_model_preProc_notune, 
                   nnet_model_preProc_tune = nnet_model_preProc_tune)
set.seed(100)
resamp <- resamples(model_list)
summary(resamp)

# Fit linear model
set.seed(100)
linear_model <- train(formula, 
                                 data = mtcars,
                                 method = "lm",
                                 trControl = trainControl(method = "cv"),
                                 preProc = c("center", "scale"),
                                 allowParallel = FALSE)
linear_model

# Make random forests model
mtryGrid <- data.frame(mtry = floor(seq(10, ncol(data_set_train) - 1, length = 10)))

set.seed(100)
rf_model <- train(formula, 
                  data = mtcars,
                  method = "rf",
                  tuneGrid = mtryGrid,
                  ntree = 1000,
                  importance = TRUE,
                  trControl = trainControl(method = "cv"))
rf_model
plot(rf_model)

# Compare models
model_list <- list(nnet_model_basic = nnet_model_basic, 
                   nnet_model_tune_nopreProc = nnet_model_tune_nopreProc, 
                   nnet_model_preProc_notune = nnet_model_preProc_notune, 
                   nnet_model_preProc_tune = nnet_model_preProc_tune,
                   linear_model = linear_model,
                   rf_model = rf_model)
set.seed(100)
resamp <- resamples(model_list)
summary(resamp)
parallelplot(resamp, metric = "Rsquared")
