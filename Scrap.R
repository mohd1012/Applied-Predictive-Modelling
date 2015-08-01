# Fit lowess
library(splines)
library(foreach)
library(gam)
library(lattice)
library(ggplot2)
library(caret)
# crashes when I uncomment the tuneGrid = tuneGrid line. Maybe impossible values?

training_control <- trainControl(method = "cv", allowParallel = FALSE)
tune_grid <- expand.grid(span = seq(0.1, 0.9, length = 9), degree = seq(1, 2, length = 2))
set.seed(Set_seed_seed)
GAM_model <- train(formula,
                      importance = TRUE,
                      data = data_set_train,
                      method = "gamLoess", 
                      tuneGrid = tune_grid,
                      trControl = training_control)


# http://topepo.github.io/caret/Generalized_Additive_Model.html
# Tuning Parameters: span (Span), degree (Degree)

z <- loess(formula = formula, data = data_set_train, span = 20)
plot(z)
z <- gam(formula = formula, data = data_set_train)

# Regression diagnostics
x <- predict(GAM_model, data_set_train)
# Tuning Parameters: span (Span), degree (Degree)
# tuneGrid <- expand.grid(ncomp = 1:(ncol(data_set_train) - 2))
plot(x - data_set_train$price)

# Residuals vs fitted values
plot(x, data_set_train$price)
# Residuals vs features
cor(x, data_set_train[, -grep("price", colnames(data_set_train))])
# Residual distribution
hist(x - data_set_train$price)
qqnorm(x)
# Scale location (sqrt of standardize residuals vs fitted values)
plot(x = x, y = sqrt(abs(scale(x - data_set_train$price))))

