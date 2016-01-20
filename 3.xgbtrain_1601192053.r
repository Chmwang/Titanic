rm(list=ls(all=TRUE))
gc()




##dir
{
# homedir <- "//matfic00/Utilisateurs-AXC/C-WANG/Personnel/Kaggle/Titanic"
homedir <- "/Users/chengmingwang/Google Drive/KaggleMaster/Titanic"

codedir <- file.path(homedir, "codes")
datadir <- file.path(homedir, "data")
outputdir <- file.path(homedir, "output")

source("/Users/chengmingwang/Google Drive/KaggleMaster/Toolbox160107.r")
}



###packages
{
##data
library(ggplot2)
library(data.table)
library(lubridate)
library(dummies)
library(stringr)
library(dplyr)
library(tidyr)
library(readr)
# library(tm)
# library(readxl)
# library(reshape2)
# library(RSQLite)
# library(compare)

##model
library(gbm)
library(xgboost)
library(randomForest)
library(glmnet)
library(leaps)
library(Ckmeans.1d.dp)
library(DiagrammeR)
library(pROC)
library(glmnet)
library(rpart)
library(mice)
library(rpart)
library(rpart.plot)
library(rattle)
library(party)
library(RColorBrewer)
}


###load pretreat data
{
load(file = file.path(datadir, 'pretreat/data2.Rda'))
# full[,Survived:=as.factor(Survived)]

}


###xgboost
{
tm <- GetSysTime()
mth <- "xgbtrain"
writedir <- file.path(outputdir, paste("model",paste(tm,mth,sep = "_"),sep="/"))
CreatDirIfNotExist(writedir)
setwd(writedir)
set.seed(7)


feat<-c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Pref2","FamilySize","FamilyID","Survived")
# feat<-c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Pref2","FamilySize","FamilyID2")

catevars <- c("Pclass","Sex","Embarked","Pref2","FamilyID")

full <- adddummy(full,catevars)

var_slct <- c()
for (f in feat){
  var_slct <- c(var_slct,names(full)[str_detect(names(full),f)])
}


train <- full[PassengerId %in% train.id,var_slct,with=F]
test <- full[PassengerId %in% test.id,var_slct,with=F]

xdata <- data.matrix(train[,-"Survived",with=F])
# ydata <- as.factor(data.matrix(train[,"Survived",with=F]))
ydata <- data.matrix(train[,"Survived",with=F])


set.seed(7)
dtrain <- xgb.DMatrix(xdata,label=ydata)
dtest <- dtrain



watchlist <- list(eval = dtest, train = dtrain)
c.booster = 'gbtree'
c.eta = 0.3
c.maxdepth = 10
c.min_child_weight=1
c.nrounds=1e4
c.subsample=0.6
c.colsample_bytree=0.8
xgb1 <- xgb.train(params = list(booster = c.booster, 
                                silent = 0,
                                eta = c.eta, 
                                maxdepth = c.maxdepth, 
                                # objective = 'reg:linear',
                                objective = 'binary:logistic',
                                # objective = 'rank',
                                min_child_weight=c.min_child_weight,
                                subsample=c.subsample,
                                colsample_bytree=c.colsample_bytree,
                                eval_metric="rmse",
                                # eval_metric="auc",
                                max_depth=c.maxdepth),
                  data = dtrain,
                  nrounds = c.nrounds,
                  verbose = 2,
                  watchlist = watchlist,
                  # nthread = n.cores,
                  early.stop.round=c.nrounds*0.4,
                  # maximize=TRUE
                  maximize=FALSE
)




Prediction <- predict(xgb1, data.matrix(test[,-"Survived",with=F]))
thrd<-0.5
Prediction[Prediction<thrd]<-0
Prediction[Prediction>=thrd]<-1

submission <- data.frame(PassengerId = test.id, Survived = Prediction)
write.csv(submission,file.path(writedir,paste0("sub",mth,tm,".csv")),row.names = F)

}
