# First tests and model comparisons

# Import dataset
dataset <- read.csv("../data.csv", header=TRUE)
# Remove file name column
dataset <- dataset[,-1]

# Load library
library(dplyr)
library(na.tools)
library(caret)
library(naivebayes)
library(randomForest)

# Check if there is some NA values
n_na(dataset)

# Separate the data set into 70% / 30% for training and test set
idxs <- sample(1:nrow(dataset),as.integer(0.7*nrow(iris)))
train_dataset <- dataset[-idxs,]
test_dataset <- dataset[idxs,]
train_dataset <- as_tibble(train_dataset)
test_dataset <- as_tibble(test_dataset)

# Save test genre values
test_musical_genres <- test_dataset$label
test_musical_genres <- as_tibble(test_musical_genres)
test_musical_genres$value <- as.factor(test_musical_genres$value)

# Remove genre attribute from test dataset
test_dataset <- test_dataset %>% select(-label)

# String to Levels (factor)
train_dataset$label <- as.factor(train_dataset$label)

# KNN model
knn.model <- knn3(label ~., data=train_dataset, k=50)
knn.preds <- predict(knn.model, test_dataset,type="class")
knn.preds # Result of the test
knn.confM <- confusionMatrix(test_musical_genres$value, knn.preds)
knn.confM # Accuracy

# Naive Bayes model
nb.model <- naive_bayes(label ~., data=train_dataset)
nb.model
nb.preds <- predict(nb.model, test_dataset,type="class", laplace=1)
nb.preds # Result of the test
nb.confM <- confusionMatrix(test_musical_genres$value, nb.preds)
nb.confM # Accuracy

# Random Forest model
randomForestModel <- randomForest(label ~ ., data = train_dataset, ntree = 5000, mtry = 6, importance = TRUE)
predValid <- predict(randomForestModel, test_dataset, type = "class")
predValid # Result of the test
randomForestModel # Accuracy


