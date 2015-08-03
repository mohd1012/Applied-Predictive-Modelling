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

########################################################
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


###################################################
# Blog postings
# Charpentier, A. (2015, July 21). Choosing a classifier.
# Retrieved July 31, 2015
# from Freakonometrics: http://freakonometrics.hypotheses.org/20002
###################################################
library(ggplot2)
n <- 500
set.seed(1)
X <- rnorm(n)
ma <- 10 - (X + 1.5) ^ 2*2
mb <- -10 + (X - 1.5) ^ 2*2
M <- cbind(ma,mb)
set.seed(1)
Z <- sample(1:2,size = n,replace = TRUE)
Y <- ma*(Z == 1) + mb*(Z == 2) + rnorm(n)*5
df <- data.frame(Z = as.factor(Z),X,Y)
p <- ggplot(data = df, aes(x = X, y = Y, color = Z))
p <- p + geom_point()
p
df1 = training = df[1:300,]
df2 = testing  = df[301:500,]
library(rpart)
fit <- rpart(Z ~ X + Y, data = df1)
pred <- function(x,y) predict(fit,newdata = data.frame(X = x,Y = y))[,1]
vx <- seq(-3,3,length = 101)
vy <- seq(-25,25,length = 101)

z <- NULL
for (i in 1:length(vx)) {
  for (j in 1:length(vy)) {
    z_pred <- pred(vx[i],vy[j])
    new_row <- cbind(vx[i],vy[j], z_pred)
    z <- rbind(z, new_row)
  }
}
z <- as.data.frame(z)
colnames(z) <- c("vx", "vy", "pred_z")

p <- ggplot()
p <- p + geom_tile(data = z, aes(x = vx, y = vy, color = pred_z))
p <- p + geom_point(data = df, aes(x = X, y = Y, fill = Z, shape = Z),
                    pch = 21, size = 5, colour = NA)
p

