#install.packages('ROCR')

require(ROCR)
require(caret)
require(ggplot2)

Evaluate_GBM_AUC <- function(dfEvaluate, CV=5, trees=3, depth=2, shrink=0.1) {
  require(caret)
  require(Metrics)
  CVs <- CV
  cvDivider <- floor(nrow(dfEvaluate) / (CVs+1))
  indexCount <- 1
  outcomeName <- c('cluster')
  predictors <- names(dfEvaluate)[!names(dfEvaluate) %in% outcomeName]
  lsErr <- c()
  lsAUC <- c()
  for (cv in seq(1:CVs)) {
    print(paste('cv',cv))
    
    dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
    dataTest <- dfEvaluate[dataTestIndex,]
    dataTrain <- dfEvaluate[-dataTestIndex,]
    
    
    dataTrain[,outcomeName] <- ifelse(dataTrain[,outcomeName]==1,'yes','nope')
    
    # create caret trainControl object to control the number of cross-validations performed
    objControl <- trainControl(method='cv', number=2, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)
    
    # run model
    bst <- train(dataTrain[,predictors],  as.factor(dataTrain[,outcomeName]), 
                 method='gbm', 
                 trControl=objControl,
                 metric = "ROC",
                 tuneGrid = expand.grid(n.trees = trees, interaction.depth = depth, shrinkage = shrink)
    )
    
    predictions <- predict(object=bst, dataTest[,predictors], type='prob')
    auc <- auc(ifelse(dataTest[,outcomeName]==1,1,0),predictions[[2]])
    err <- rmse(ifelse(dataTest[,outcomeName]==1,1,0),predictions[[2]])
    
    lsErr <- c(lsErr, err)
    lsAUC <- c(lsAUC, auc)
    gc()
  }
  print(paste('Mean Error:',mean(lsErr)))
  print(paste('Mean AUC:',mean(lsAUC)))
}

# https://archive.ics.uci.edu/ml/datasets/Gisette
# http://stat.ethz.ch/R-manual/R-devel/library/stats/html/princomp.html
# word of warning, this is 20mb - slow
#library(RCurl) # download https data
#urlfile <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data'
#x <- getURL(urlfile, ssl.verifypeer = FALSE)
#gisetteRaw <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)
#urlfile <- "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels"
#x <- getURL(urlfile, ssl.verifypeer = FALSE)
#g_labels <- read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)

data <- read.csv("D:\\Users\\spatel\\Dropbox\\anaconda\\santandar\\train.csv", header = TRUE)
dim(data)
head(data)

#myvars1 <- names(data) %in% c("target")
#myvars2 <- names(data) %in% c("ID", "target") 
myvars2 <- names(data) %in% c("target")
myvars1 <- names(data) %in% c("ID", "target") 


gisetteRaw <- data[!myvars1]
g_labels <- data[myvars2]

head(indata)
head(outdata)


# Remove zero and close to zero variance

nzv <- nearZeroVar(gisetteRaw, saveMetrics = TRUE)
range(nzv$percentUnique)

# how many have no variation at all
print(length(nzv[nzv$zeroVar==T,]))

print(paste('Column count before cutoff:',ncol(gisetteRaw)))

# how many have less than 0.1 percent variance
dim(nzv[nzv$percentUnique > 0.1,])

# remove zero & near-zero variance from original data set
gisette_nzv <- gisetteRaw[c(rownames(nzv[nzv$percentUnique > 0.1,])) ]
print(paste('Column count after cutoff:',ncol(gisette_nzv)))

# Run model on original data set

dfEvaluateOrig <- cbind(as.data.frame(sapply(gisette_nzv, as.numeric)),
                        cluster=g_labels$target)

Evaluate_GBM_AUC(dfEvaluateOrig, CV=5, trees=10, depth=2, shrink=1) 

# Run prcomp on the data set

pmatrix <- scale(gisette_nzv)
princ <- prcomp(pmatrix)

# change nComp to try different numbers of component variables
nComp <- 20  
dfComponents <- predict(princ, newdata=pmatrix)[,1:nComp]
dfEvaluatePCA <- cbind(as.data.frame(dfComponents),
                       cluster=g_labels$target)
Evaluate_GBM_AUC(dfEvaluatePCA,CV=5, trees=10, depth=2, shrink=1) 