#Load libraries needed for graphing, statistical tests, and machine learning
library(ggplot2)
library(rpart)
library(dplyr)
library(caTools)
library(randomForest)
library(caret)
library(doSNOW)
library(rpart.plot)

#Read the file with our study results
Study <- read.csv("TSF Study.csv")
#Create vector with 4 categories of Quizlet cards
Category <- c(rep("Numbers", nrow(Study)),
            rep("Letters", nrow(Study)),
            rep("Colours", nrow(Study)),
            rep("Shapes", nrow(Study)))
Category <- factor(Category, levels = c("Colours", "Letters", "Numbers", "Shapes"))

#Create vector with all values from all categories, appended, in order
Score <- c(Study[,1], Study[,2], Study[,3], Study[,4])
Score <- as.factor(Score)
StudyData <- data.frame(Category, Score)

#Plot multiple bar graph of category vs frequency
ggplot(StudyData, aes(Category, fill = Score)) +
  geom_bar(position = "dodge")

#Prepare data for bar graph of overall results
Score <- c(Study[,1], Study[,2], Study[,3], Study[,4], Study[,5], Study[,6])
AllCategories <- c(rep("Numbers", nrow(Study)),
                 rep("Letters", nrow(Study)),
                 rep("Colours", nrow(Study)),
                 rep("Shapes", nrow(Study)),
                 rep("Control", nrow(Study)),
                 rep("Total", nrow(Study)))
AllCategories <- factor(AllCategories, levels = c("Colours", "Letters", "Numbers", "Shapes", "Control", "Total"))
#Create dataframe with AllCategories and Score as variables
StudyData1 <- data.frame(AllCategories, Score)
StudyData1 <- StudyData1 %>% group_by(AllCategories) %>% summarise(Mean = mean(Score), SD = sd(Score))

#Plot bar graph of category vs mean recognition level across all categories
ggplot(StudyData1, aes(AllCategories, Mean, fill = AllCategories)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = Mean - SD, ymax = Mean + SD), width = 0.2)

#Define the variable Success as whether a student recalled better than the average
Success <- c()
for(i in 1:nrow(Study)){
  x <- ifelse(Study$Total[i] > StudyData1[6,2], TRUE, FALSE)
  Success <- c(Success, x)
}
Study <- cbind(Study, Success)

#Two-way ANOVA test on flashcard types
Anova <- aov(data = Study, Success ~ Numbers + Letters + Colours + Shapes)
summary(Anova)
#Add interactions between variables
AnovaInteractions <- aov(data = Study, Success ~ Numbers * Letters * Colours * Shapes * Control)
summary(AnovaInteractions)

#Train a Random Forest with the parameters as Numbers & Letters
RfTrain1 <- Study[c("Numbers", "Letters")]
RfLabel <- as.factor(Study$Success)
set.seed(1)
#Generate the Random Forest
Rf1 <- randomForest(x = RfTrain1, y = RfLabel, importance = TRUE, ntree = 1000)
Rf1
#Result: OOB estimate oferror rate: 24.36%

#Train a Random Forest with the parameters as Numbers, Letters & Colours
RfTrain2 <- Study[c("Numbers", "Letters", "Colours")]
set.seed(1)
#Generate the Random Forest
Rf2 <- randomForest(x = RfTrain2, y = RfLabel, importance = TRUE, ntree = 1000)
Rf2
#Results: OOB estimate of error rate: 14.1%

#Train a Random Forest with the parameters as Numbers, Letters & Shapes
RfTrain3 <- Study[c("Numbers", "Letters", "Shapes")]
set.seed(1)
#Generate the Random Forest
Rf3 <- randomForest(x = RfTrain3, y = RfLabel, importance = TRUE, ntree = 1000)
Rf3
#Results: OOB estimate of error rate: 11.54%

#Train a Random Forest with the parameters as Letters, Colours & Shapes
RfTrain4 <- Study[c("Letters", "Colours", "Shapes")]
set.seed(1)
#Generate the Random Forest
Rf4 <- randomForest(x = RfTrain4, y = RfLabel, importance = TRUE, ntree = 1000)
Rf4
#Results: OOB estimate of error rate: 24.36%

#Train a Random Forest with the parameters as Letters, Shapes, & Control
RfTrain5 <- Study[c("Letters", "Shapes", "Control")]
set.seed(1)
#Generate the Random Forest
Rf5 <- randomForest(x = RfTrain5, y = RfLabel, importance = TRUE, ntree = 1000)
Rf5
#Results: OOB estimate of error rate: 12.82%

#Train a Random Forest with the parameters as Numbers, Letters, Colours & Control
RfTrain6 <- Study[c("Numbers", "Letters", "Colours", "Control")]
set.seed(1)
#Generate the Random Forest
Rf6 <- randomForest(x = RfTrain6, y = RfLabel, importance = TRUE, ntree = 1000)
Rf6
#Results: OOB estimate of error rate: 14.1%

#Visualize importance of variables in best-performing Random Forests
varImpPlot(Rf3)
varImpPlot(Rf5)
varImpPlot(Rf2)
varImpPlot(Rf6)

#Set up caret's trainControl object
Ctrl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

#Test out Rf3
Cl <- makeCluster(2, type = "SOCK")
registerDoSNOW(Cl)
set.seed(1)
Rf3Cv1 <- train(x = RfTrain3, y = RfLabel, method = "rf", tuneLength = 2,
                ntree = 100, trControl = Ctrl1)
stopCluster(Cl)
Rf3Cv1
#Results: mtry = 2, Accuracy = 0.9019841, Kappa = 0.7962724 <-- Our best result!

#Test out Rf5
Cl <- makeCluster(2, type = "SOCK")
registerDoSNOW(Cl)
set.seed(1)
Rf5Cv1 <- train(x = RfTrain5, y = RfLabel, method = "rf", tuneLength = 2,
                ntree = 100, trControl = Ctrl1)
stopCluster(Cl)
Rf5Cv1
#Results: mtry = 2, Accuracy = 0.8786508, Kappa = 0.7500297

#Test out Rf2
Cl <- makeCluster(2, type = "SOCK")
registerDoSNOW(Cl)
set.seed(1)
Rf2Cv1 <- train(x = RfTrain2, y = RfLabel, method = "rf", tuneLength = 2,
                ntree = 100, trControl = Ctrl1)
stopCluster(Cl)
Rf2Cv1
#Results: mtry = 3, Accuracy = 0.8599206, Kappa = 0.7099774

#Test out Rf6
Cl <- makeCluster(2, type = "SOCK")
registerDoSNOW(Cl)
set.seed(1)
Rf6Cv1 <- train(x = RfTrain6, y = RfLabel, method = "rf", tuneLength = 2,
                ntree = 100, trControl = Ctrl1)
stopCluster(Cl)
Rf6Cv1
#Results: mtry = 2, Accuracy = 0.8639881, Kappa = 0.7218974

#Create utility function
RpartCV <- function(Seed, Training, Labels, Ctrl){
  Cl <- makeCluster(6, type = "SOCK")
  registerDoSNOW(Cl)
  set.seed(Seed)
  RpartCV <- train(x = Training, y = Labels, method = "rpart", tuneLength = 30,
                   trControl = Ctrl)
  stopCluster(Cl)
  return(RpartCV)
}

set.seed(1)
CvFolds <- createMultiFolds(RfLabel, k = 10, times = 10)
Ctrl2 <- trainControl(method ="repeatedcv", number = 10, repeats = 10,
                      index = CvFolds)
#Grab features
Features <- c("Numbers", "Letters", "Shapes")
RpartTrain3 <- Study[1:78, Features]
Rpart3CV1 <- RpartCV(1, RpartTrain3, RfLabel, Ctrl2)
Rpart3CV1
#Plot
prp(Rpart3CV1$finalModel, type = 0, extra = 1, under = TRUE)
#Accuracy = 0.7735317 for k = 10, times = 10