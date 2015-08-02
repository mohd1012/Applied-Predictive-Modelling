# Fit lowess
library(splines)
library(foreach)
library(gam)
library(lattice)
library(ggplot2)
library(caret)

# crashes when I uncomment the tuneGrid = tuneGrid line. Maybe impossible values?
Set_seed_seed <- 100
data_set <- diamonds[, c(1, 5, 6, 7, 8, 9, 10)]
data_set <- data_set[1:1000,]
formula <- price ~ carat + depth + table + x + y + z
training_control <- trainControl(method = "cv", allowParallel = FALSE)
tune_grid <- expand.grid(span = seq(0.1, 0.9, length = 9), degree = seq(1, 2, length = 2))
set.seed(Set_seed_seed)
GAM_model <- train(formula,
                      data = data_set,
                      method = "gamLoess", 
                      # tuneGrid = tune_grid
                      trControl = training_control
                   )

# http://topepo.github.io/caret/Generalized_Additive_Model.html
# Tuning Parameters: span (Span), degree (Degree)

# Regression diagnostics
n <- predict(GAM_model, data_set)
# Tuning Parameters: span (Span), degree (Degree)
# tuneGrid <- expand.grid(ncomp = 1:(ncol(data_set_train) - 2))

# Residuals vs order
plot(n - data_set$price)
# Predicted vs fitted values
plot(n, data_set$price)
# Residuals vs features
cor(n - data_set$price, data_set[, -grep("price", colnames(data_set))])
# Predicted vs feature
cor(n, data_set[, -grep("price", colnames(data_set))])
# Residual distribution
hist(n - data_set$price)
qqnorm(n)
# Scale location (sqrt of standardize residuals vs fitted values)
plot(x = n, y = sqrt(abs(scale(n - data_set$price))))

mat_x <- as.matrix(data_set[, -grep("price", colnames(data_set))])
hat_x <- mat_x %*% ginv( t(mat_x) %*% mat_x) %*% t(mat_x)
diag_hat <- diag(hat_x)
D <- (n - data_set$price) * diag_hat/(1 - (diag_hat %*% diag_hat))
plot(D)
hist(D)
qqnorm(D)
plot(n - data_set$price)
