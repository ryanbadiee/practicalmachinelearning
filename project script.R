#Download training/testing sets
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
               "./trainingset.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              "./testingset.csv")
trainingset <- read.csv("./trainingset.csv")
testing <- read.csv("./testingset.csv")


#Make training vs validation set
library(caret)
inTrain <- createDataPartition(y=trainingset$classe,p=0.75,list=F)
training <- trainingset[inTrain,]
validation <- trainingset[-inTrain,]

#First, start by removing covariates with nearly no variability for simplicity
nsv <- nearZeroVar(training, saveMetrics = T)
library(dplyr)
nsv <- subset(nsv, nsv$nzv==TRUE)
nsv
training <- training[ , !(names(training) %in% row.names(nsv))]
validation <- validation[,!(names(validation) %in% row.names(nsv))]

#Next, identify variables with many NAs and remove from data set
totalnas <- sapply(training, function(x) sum(is.na(x)))
propnas <- totalnas/nrow(training)
propnas <- propnas[which(propnas>0.9)] #If over 90% NAs, remove from data
toomanynas <- names(propnas)
training <- training[ , !(names(training) %in% toomanynas)]
validation <- validation[,!(names(validation) %in% toomanynas)]

#Finally, remove variables a priori if not relevant, i.e. timepoint
training <- training[,-(1:5)]
validation <- validation[,-(1:5)]

#Impute missing data to improve performance
Itraining <- preProcess(training[,-54],method="knnImpute")
Itraining <- data.frame(Itraining$data,training$classe)
Ivalidation <- preProcess(validation[,-54],method="knnImpute")
Ivalidation <- data.frame(Ivalidation$data,validation$classe)

#Generate boosted model:
boomod <- train(training.classe~.,data=Itraining,method="gbm")
checkboo <- predict(boomod,Ivalidation)
booacc <- confusionMatrix(Ivalidation$validation.classe, crossval)

#Generate random forest model:
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
rfmodControl <- trainControl(method = "cv",
                        number = 5,
                        allowParallel = TRUE)
rfmod <- train(training.classe~.,data=Itraining,method="rf", metric="Accuracy", maximize=TRUE)
stopCluster(cluster)
registerDoSEQ()
checkrf <- predict(rfmod,Ivalidation)
rfacc <- confusionMatrix(Ivalidation$validation.classe,checkrf)

#Generate linear discriminant analysis model:
ldamod <- train(training.classe~.,data=Itraining,method="lda")
checklda <- predict(ldamod,Ivalidation)
ldaacc <- confusionMatrix(Ivalidation$validation.classe,checklda)

#Stack models:
predDF <- data.frame(checkboo, checkrf, 
                     classe = Ivalidation$validation.classe)
combmod <- train(classe ~ ., data=predDF, method="gam")
checkcomb <- predict(combmod,predDF)
combacc <- confusionMatrix(Ivalidation$validation.classe, checkcomb)

#Predict testing data:
testing <- testing[ , !(names(testing) %in% row.names(nsv))]
testing <- testing[ , !(names(testing) %in% toomanynas)]
testing <- testing[,-(1:5)]
Itesting <- preProcess(testing[,-54],method="knnImpute")
Itesting <- data.frame(Itesting$data,testing$problem_id)
testrf <- predict(rfmod,Itesting)
