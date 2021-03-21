# Get genre of a unique music

# Import dataset
dataset <- read.csv("data.csv", header=TRUE)

# Import music values
musicValues <- read.csv("testSongs/discoFromYoutube.csv", header=TRUE)

# remove file name column
dataset <- dataset[,-1]
musicValues <- musicValues[,-1]

# Load library
library(dplyr)
library(na.tools)
library(caret)
library(naivebayes)
library(randomForest)

# String to Levels (factor)
dataset$label <- as.factor(dataset$label)

# knn 
knn.model <- knn3(label ~., data=dataset, k=50)
knn.preds <- predict(knn.model, musicValues,type="class")
knn.preds # result 

# naive bayes
nb.model <- naive_bayes(label ~., data=dataset)
nb.preds <- predict(nb.model, musicValues,type="class", laplace=1)
nb.preds # result 

# random forest
randomForestModel <- randomForest(label ~ ., data = dataset, ntree = 5000, mtry = 6, importance = TRUE)
predValid <- predict(randomForestModel, musicValues, type = "class")
predValid # result 