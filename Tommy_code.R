library(lattice)
library(ggplo2)
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


