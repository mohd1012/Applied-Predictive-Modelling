# Fit lowess
library(splines)
library(foreach)
library(gam)
library(lattice)
library(ggplot2)
library(caret)

# check atleast min and skew
summary_stat <- adply(.data = data_set, .margins = 2, .fun = min)
colnames(summary_stat) <- c("feature", "min")
p <- ggplot(data = summary_stat, aes(x = feature, y = min))
p <- p + geom_bar(stat = "identity")
p

summary_stat <- adply(.data = data_set, .margins = 2, .fun = skew)
colnames(summary_stat) <- c("feature", "skew")
p <- ggplot(data = summary_stat, aes(x = feature, y = skew))
p <- p + geom_bar(stat = "identity")
p



# crashes when I uncomment the tuneGrid = tuneGrid line.
# http://stackoverflow.com/questions/32043010/r-crashes-when-training-using-caret-and-method-gamloess
# Bug in GAM package. Can't have degree  = 2.

Set_seed_seed <- 100
data_set <- diamonds[, c(1, 5, 6, 7, 8, 9, 10)]
data_set <- data_set[1:1000,]
formula <- price ~ carat + depth + table + x + y + z
training_control <- trainControl(method = "cv", allowParallel = FALSE)
tune_grid <- expand.grid(span = seq(0.1, 0.9, length = 9), degree = 1)
set.seed(Set_seed_seed)
GAM_model <- train(formula,
                      data = data_set,
                      method = "gamLoess", 
                      tuneGrid = tune_grid,
                      trControl = training_control
                   )


########################################################
# Regression diagnostics
n <- predict(model_list$rf_model, data_set)
# Tuning Parameters: span (Span), degree (Degree)
# tuneGrid <- expand.grid(ncomp = 1:(ncol(data_set_train) - 2))

# Residuals vs order
plot(n - data_set[, response_col])
# Predicted vs fitted values
plot(n, unlist(data_set[, response_col]))
# Residuals vs features
cor(n - data_set[, response_col], data_set[, -response_col])
# Predicted vs feature
cor(n, data_set[, -response_col])
# Residual distribution
hist(n - unlist(data_set[, response_col]))
qqnorm(n)
# Scale location (sqrt of standardize residuals vs fitted values)
plot(x = n, y = sqrt(abs(scale(n - data_set[, response_col]))))

mat_x <- as.matrix(data_set[, -response_col])
hat_x <- mat_x %*% ginv( t(mat_x) %*% mat_x) %*% t(mat_x)
diag_hat <- diag(hat_x)
plot(diag_hat)
plot(diag_hat/(1 - (diag_hat %*% diag_hat)))

D <- (n - data_set[, response_col]) * diag_hat/(1 - (diag_hat %*% diag_hat))
plot(D[,1])
hist(D[,1])
qqnorm(D[,1])

###################################################
# Blog postings
# Charpentier, A. (2015, July 21). Choosing a classifier.
# Retrieved July 31, 2015
# from Freakonometrics: http://freakonometrics.hypotheses.org/20002
###################################################
library(ggplot2)
# Make dataset
n <- 500
set.seed(10)
X <- rnorm(n)
ma <- 10 - (X + 1.5) ^ 2*2
mb <- -10 + (X - 1.5) ^ 2*2
M <- cbind(ma,mb)
set.seed(10)
Z <- sample(1:2,size = n,replace = TRUE)
Y <- ma*(Z == 1) + mb*(Z == 2) + rnorm(n)*5
df <- data.frame(Z = as.factor(Z),X,Y)

df1 = training = df[1:300,]
df2 = testing  = df[301:500,]
#
# library(rpart) # code for Rpart
# fit <- rpart(Z ~ X + Y, data = df1)
# pred <- function(x,y) predict(fit,newdata = data.frame(X = x,Y = y))[,1]
# 
# fit = glm(Z ~ X + Y,data = df1, family = binomial)
# pred = function(x,y) {
#   predict(fit,newdata = data.frame(X = x,Y = y), type = "response")
# }
# library(MASS)
# fit = qda(Z ~ X+Y,data = df1, family = binomial)
# pred = function(x,y) {
#   predict(fit, newdata = data.frame(X = x,Y = y))$posterior[,2]
# }
#
# library(mgcv)
# fit = gam(Z ~ s(X,Y),data = df1,family = binomial)
# pred = function(x,y) {
#   predict(fit, newdata = data.frame(X = x, Y = y), type = "response")
# }
#
# library(caret)
# fit = knn3(Z ~ X + Y,data = df1, k = 9)
# pred = function(x,y) {
#   predict(fit,newdata = data.frame(X = x,Y = y))[,2]
# }

# Define prediction function
library(randomForest)
fit <- randomForest(Z ~ X + Y, data = df1)
pred <- function(x,y) predict(fit,newdata = data.frame(X = x,Y = y), type = "prob")[,2]

# Make predictions
vx <- seq(-3,3,length = 101)
vy <- seq(-25,25,length = 101)
z <- expand.grid(vx = seq(-3, 3, length = 101), vy = seq(-25, 25, length = 101), stringsAsFactors = FALSE)
z$pred_z <- pred(z$vx, z$vy)

# Plot data set
p <- ggplot(data = df, aes(x = X, y = Y, color = Z))
p <- p + geom_point()
p <- p + ggtitle("Random Forests")
p <- p + xlab("variable 1") + ylab("variable 2")
p

# Plot fit
p <- ggplot()
p <- p + geom_tile(data = z, aes(x = vx, y = vy, fill = pred_z))
p <- p + geom_contour(data = z, aes(x = vx, y = vy, z = pred_z), colour = "slategrey") 
p <- p + geom_point(data = df, aes(x = X, y = Y, fill = as.integer(Z)),
                    pch = 21, size = 5, colour = NA)
p <- p + theme_bw() + ggtitle("Random Forests")
p <- p + xlab("variable 1") + ylab("variable 2")
p

p <- ggplot()
p <- p + geom_point(data = z, aes(x = vx, y = vy, colour = as.integer(2*pred_z)), pch = 15, size = 2)
p <- p + geom_point(data = df, aes(x = X, y = Y, fill = as.integer(Z)),
                    pch = 21, size = 5, colour = NA)
p <- p + theme_bw() + ggtitle("Random Forests")
p <- p + xlab("variable 1") + ylab("variable 2")
p
################################################
