rm(list=ls(all=TRUE))
gc()



# visualize <- FALSE

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
}


##dir
{
homedir <- "//matfic00/Utilisateurs-AXC/C-WANG/Personnel/Kaggle/Titanic"
# homedir <- "/Users/chengmingwang/Google Drive/Kaggle/Titanic"

codedir <- file.path(homedir, "codes")
datadir <- file.path(homedir, "data")
outputdir <- file.path(homedir, "output")

source("//matfic00/Utilisateurs-AXC/C-WANG/Personnel/20160108_DataScienceToolBox/Toolbox160107.r")
}




###load data
{
train <- as.data.table(read.csv(file.path(datadir, "/rawdata/train.csv"), stringsAsFactors = FALSE))
test <- as.data.table(read.csv(file.path(datadir, "/rawdata/test.csv"), stringsAsFactors = FALSE))
}




###check missing missing data in Age&Fare 
check.na.df(train)
check.na.df(test)
check.charempty.df(train)
check.charempty.df(test)



###get passage ids for train&test set
train.id <- train$PassengerId
test.id <- test$PassengerId

###combine train & test data
full <- rbindlist(list(train,test),use.names=TRUE, fill=TRUE)


###
full$Embarked[is.na(full$Embarked)]<-'S'


###add Pref1
# strsplit(full$Name[1], split='[,.]')
# strsplit(full$Name[1], split='[,.]')[[1]]
# strsplit(full$Name[1], split='[,.]')[[1]][2]
full$Pref1 <- sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
full$Pref1 <- sub(' ', '', full$Pref1)
table(full$Pref1)
full$Pref1[full$PassengerId == 797] <- 'Mrs' # female doctor
full$Pref1[full$Pref1 %in% c('Lady', 'the Countess', 'Mlle', 'Mee', 'Ms')] <- 'Miss'
full$Pref1[full$Pref1 %in% c('Capt', 'Don', 'Major', 'Sir', 'Col', 'Jonkheer', 'Rev', 'Dr', 'Master')] <- 'Mr'
full$Pref1[full$Pref1 %in% c('Dona','Mme')] <- 'Mrs'
table(full$Pref1)

###add Pref2
full$Pref2 <- sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
full$Pref2 <- sub(' ', '', full$Pref2)
table(full$Pref2)
full$Pref2[full$Pref2 %in% c('Mme', 'Mlle')] <- 'Mlle'
full$Pref2[full$Pref2 %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
full$Pref2[full$Pref2 %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
table(full$Pref2)


###Adding FamilyNum
full$FamilyNum <- full$SibSp + full$Parch + 1


###Perform Imputation to remove NAs
# set.seed(144)
# vars.for.imputation = setdiff(names(full), "Survived")
# imputed = complete(mice(full[,vars.for.imputation,with=F]))
# full[,vars.for.imputation,with=F] <- imputed



# replace NA Fare value
full$Fare[is.na(full$Fare)] <- median(full$Fare, na.rm=TRUE)



#Adding Mother
full$Mother<-0
full$Mother[full$Sex=='female' & full$Parch>0 & full$Age>18 & full$Pref1!='Miss']<-1
sum(full$Mother)

#Adding Child
full$Child<-0
full$Child[full$Parch>0 & full$Age<=18]<- 1


#FamilyId
full$Surname <- sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
full$FamilyID <- paste(as.character(full$FamilyNum), full$Surname, sep="")
full$FamilyID[full$FamilyNum <= 2] <- 'Small'
# table(full$FamilyID)
famIDs <- data.frame(table(full$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
full$FamilyID[full$FamilyID %in% famIDs$Var1] <- 'Small'
full$FamilyID <- factor(full$FamilyID)


#FamilyId2
full$FamilyID2 <- full$FamilyID
full$FamilyID2 <- as.character(full$FamilyID2)
full$FamilyID2[full$FamilyNum <= 3] <- 'Small'
full$FamilyID2 <- factor(full$FamilyID2)



###Exact Deck from Cabin number
full$Deck<-sapply(full$Cabin, function(x) strsplit(x,NULL)[[1]][1])


###Excat Position from Cabin number
full$CabinNum<-sapply(full$Cabin,function(x) strsplit(x,'[A-Z]')[[1]][2])
full$num<-as.numeric(full$CabinNum)
num<-full$num[!is.na(full$num)]
Pos<-kmeans(num,3)
full$CabinPos[!is.na(full$num)]<-Pos$cluster
full$CabinPos<-factor(full$CabinPos)
levels(full$CabinPos)<-c('Front','End','Middle')
full$num<-NULL

###age fit
summary(full$Age)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Pref2 + FamilyNum,
                data=full[!is.na(full$Age),], method="anova")
full$Age[is.na(full$Age)] <- predict(Agefit, full[is.na(full$Age),])
summary(full$Age)


###class
showclass(full)

table(full$Pclass)
full$Pclass <- as.factor(full$Pclass)
table(full$Pclass)

table(full$Sex)
full$Sex <- as.factor(full$Sex)
table(full$Sex)
full$FamilyID
full<-transform(full,
                Pclass=factor(Pclass),
                Sex=factor(Sex),
                Embarked=factor(Embarked),
                Pref1=factor(Pref2),
                Pref2=factor(Pref2),
                Mother=factor(Mother),
                Child=factor(Child),
                FamilyID=factor(FamilyID),
                FamilyID2=factor(FamilyID2),
                Deck=factor(Deck)
)

showclass(full)
# full$CabinNum
# table(full$CabinPos)

save(train.id, test.id,full, file = file.path(datadir, 'pretreat/data2.Rda'))




