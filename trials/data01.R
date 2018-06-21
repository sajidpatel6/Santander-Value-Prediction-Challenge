?stats
?prcomp


data <- read.csv("D:\\Users\\spatel\\Dropbox\\anaconda\\santandar\\train.csv", header = TRUE)
dim(data)
head(data)

#myvars1 <- names(data) %in% c("target")
#myvars2 <- names(data) %in% c("ID", "target") 
myvars2 <- names(data) %in% c("target")
myvars1 <- names(data) %in% c("ID", "target") 


indata <- data[!myvars1]
outdata <- data[myvars2]

head(indata)
head(outdata)

#install.packages('caret')
library(caret)
nzv <- nearZeroVar(indata, saveMetrics = TRUE)
print(paste('Range:',range(nzv$percentUnique)))

print(head(nzv))

dim(nzv[nzv$percentUnique > 0.1,])

gisette_nzv <- indata[c(rownames(nzv[nzv$percentUnique > 0.1,])) ]
print(paste('Column count after cutoff:',ncol(gisette_nzv)))

EvaluateAUC <- function(dfEvaluate) {
  require(xgboost)
  require(Metrics)
  CVs <- 5
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
    
    bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                   label = dataTrain[,outcomeName],
                   max.depth=6, eta = 1, verbose=0,
                   nround=5, nthread=4, 
                   objective = "reg:linear")
    
    predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
    err <- rmse(dataTest[,outcomeName], predictions)
    auc <- auc(dataTest[,outcomeName],predictions)
    
    lsErr <- c(lsErr, err)
    lsAUC <- c(lsAUC, auc)
    gc()
  }
  print(paste('Mean Error:',mean(lsErr, na.rm=T)))
  print(paste('Mean AUC:',mean(lsAUC, na.rm=T)))
}


dfEvaluate <- cbind(as.data.frame(sapply(gisette_nzv, as.numeric)), cluster=outdata$target)

#install.packages('xgboost')
library(xgboost)

#install.packages('Metrics')
library(Metrics)

EvaluateAUC(dfEvaluate)

pmatrix <- scale(gisette_nzv)
princ <- prcomp(pmatrix)

head(princ)

nComp <- 1  
dfComponents <- predict(princ, newdata=pmatrix)[,1:nComp]

dfEvaluate <- cbind(as.data.frame(dfComponents),
                    cluster=outdata$target)

EvaluateAUC(dfEvaluate)
head(dfEvaluate)


nComp <- 5 
dfComponents <- predict(princ, newdata=pmatrix)[,1:nComp]

dfEvaluate <- cbind(as.data.frame(dfComponents),
                    cluster=outdata$target)

EvaluateAUC(dfEvaluate)
head(dfEvaluate)


nComp <- 10
dfComponents <- predict(princ, newdata=pmatrix)[,1:nComp]

dfEvaluate <- cbind(as.data.frame(dfComponents),
                    cluster=outdata$target)

EvaluateAUC(dfEvaluate)
head(dfEvaluate)



nComp <- 20
dfComponents <- predict(princ, newdata=pmatrix)[,1:nComp]

dfEvaluate <- cbind(as.data.frame(dfComponents),
                    cluster=outdata$target)

EvaluateAUC(dfEvaluate)
head(dfEvaluate)


nComp <- 40
dfComponents <- predict(princ, newdata=pmatrix)[,1:nComp]

dfEvaluate <- cbind(as.data.frame(dfComponents),
                    cluster=outdata$target)

EvaluateAUC(dfEvaluate)
head(dfEvaluate)

