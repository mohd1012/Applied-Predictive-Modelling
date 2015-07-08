

### Get the test set results across several models

nnetPred <- predict(nnetFit, testing)
gbmPred <- predict(gbmFit, testing)
cbPred <- predict(cbFit, testing)

testResults <- rbind(postResample(nnetPred, testing$CompressiveStrength),
                     postResample(gbmPred, testing$CompressiveStrength),
                     postResample(cbPred, testing$CompressiveStrength))
testResults <- as.data.frame(testResults)
testResults$Model <- c("Neural Networks", "Boosted Tree", "Cubist")
testResults <- testResults[order(testResults$RMSE),]

################################################################################
### Section 10.3 Optimizing Compressive Strength

library(proxy)

### Create a function to maximize compressive strength* while keeping
### the predictor values as mixtures. Water (in x[7]) is used as the 
### 'slack variable'. 

### * We are actually minimizing the negative compressive strength

modelPrediction <- function(x, mod, limit = 2500)
{
  if (x[1] < 0 | x[1] > 1) return(10 ^ 38)
  if (x[2] < 0 | x[2] > 1) return(10 ^ 38)
  if (x[3] < 0 | x[3] > 1) return(10 ^ 38)
  if (x[4] < 0 | x[4] > 1) return(10 ^ 38)
  if (x[5] < 0 | x[5] > 1) return(10 ^ 38)
  if (x[6] < 0 | x[6] > 1) return(10 ^ 38)
  
  x <- c(x, 1 - sum(x))
  
  if (x[7] < 0.05) return(10 ^ 38)
  
  tmp <- as.data.frame(t(x))
  names(tmp) <- c('Cement','BlastFurnaceSlag','FlyAsh',
                  'Superplasticizer','CoarseAggregate',
                  'FineAggregate', 'Water')
  tmp$Age <- 28
  -predict(mod, tmp)
}

### Get mixtures at 28 days 
subTrain <- subset(training, Age == 28)

### Center and scale the data to use dissimilarity sampling
pp1 <- preProcess(subTrain[, -(8:9)], c("center", "scale"))
scaledTrain <- predict(pp1, subTrain[, 1:7])

### Randomly select a few mixtures as a starting pool

set.seed(91)
startMixture <- sample(1:nrow(subTrain), 1)
starters <- scaledTrain[startMixture, 1:7]
pool <- scaledTrain
index <- maxDissim(starters, pool, 14)
startPoints <- c(startMixture, index)

starters <- subTrain[startPoints,1:7]
startingValues <- starters[, -4]

### For each starting mixture, optimize the Cubist model using
### a simplex search routine

cbResults <- startingValues
cbResults$Water <- NA
cbResults$Prediction <- NA

for (i in 1:nrow(cbResults))
{
  results <- optim(unlist(cbResults[i,1:6]),
                   modelPrediction,
                   method = "Nelder-Mead",
                   control = list(maxit = 5000),
                   mod = cbFit)
  cbResults$Prediction[i] <- -results$value
  cbResults[i,1:6] <- results$par
}
cbResults$Water <- 1 - apply(cbResults[,1:6], 1, sum)
cbResults <- subset(cbResults, Prediction > 0 & Water > .02)
cbResults <- cbResults[order(-cbResults$Prediction),][1:3,]
cbResults$Model <- "Cubist"

### Do the same for the neural network model

nnetResults <- startingValues
nnetResults$Water <- NA
nnetResults$Prediction <- NA

for (i in 1:nrow(nnetResults))
{
  results <- optim(unlist(nnetResults[i, 1:6,]),
                   modelPrediction,
                   method = "Nelder-Mead",
                   control = list(maxit = 5000),
                   mod = nnetFit)
  nnetResults$Prediction[i] <- -results$value
  nnetResults[i,1:6] <- results$par
}
nnetResults$Water <- 1 - apply(nnetResults[,1:6], 1, sum)
nnetResults <- subset(nnetResults, Prediction > 0 & Water > .02)
nnetResults <- nnetResults[order(-nnetResults$Prediction),][1:3,]
nnetResults$Model <- "NNet"

### Convert the predicted mixtures to PCA space and plot

pp2 <- preProcess(subTrain[, 1:7], "pca")
pca1 <- predict(pp2, subTrain[, 1:7])
pca1$Data <- "Training Set"
pca1$Data[startPoints] <- "Starting Values"
pca3 <- predict(pp2, cbResults[, names(subTrain[, 1:7])])
pca3$Data <- "Cubist"
pca4 <- predict(pp2, nnetResults[, names(subTrain[, 1:7])])
pca4$Data <- "Neural Network"

pcaData <- rbind(pca1, pca3, pca4)
pcaData$Data <- factor(pcaData$Data,
                       levels = c("Training Set","Starting Values",
                                  "Cubist","Neural Network"))

lim <- extendrange(pcaData[, 1:2])

xyplot(PC2 ~ PC1, 
       data = pcaData, 
       groups = Data,
       auto.key = list(columns = 2),
       xlim = lim, 
       ylim = lim,
       type = c("g", "p"))


