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
# Some customised model plots:
plot(model_list$rpart_model, scales = list(x = list(log = 10)))
# A line plot of the average performance. The 'scales' argument is actually an 
# argument to xyplot that converts the x-axis to log-2 units.
plot(model_list$svm_model, scales = list(x = list(log = 2)))
rpart_tree <- as.party(model_list$rpart_model$finalModel)
plot(rpart_tree)
plot(model_list$ctree_model$finalModel)
plot(model_list$cubist_model, auto.key = list(columns = 4, lines = TRUE))

# Additional code for M5
rule_fit <- M5Rules(formula, data = data_set_train, control = Weka_control(M = 10))
rule_fit
