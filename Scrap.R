# Fit lowess
library(gam)

tuneGrid <- expand.grid(span = seq(1, 5, length = 15), degree = seq(1, 5, length = 10))
set.seed(Set_seed_seed)
GAM_model <- train(formula,
                      importance = TRUE,
                      data = data_set_train,
                      method = "gamLoess", 
                      tuneGrid = tuneGrid,
                      trControl = training_control)


# http://topepo.github.io/caret/Generalized_Additive_Model.html
# Tuning Parameters: span (Span), degree (Degree)



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

