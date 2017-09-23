# Objective: #It begins with data prep, EDA, relationship visualization and predictive modeling
# Script name: acadperf_EDA_PredModel.R
# Data source: https://www.kaggle.com/aljarah/xAPI-Edu-Data

# EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
# STEP 1: DATA PREPARATION
# clear the working space
rm(list=ls())
# Load the raw data set
acadperf.data<- read.csv("LMS-Kaggle/xAPI-Edu-Data.csv", header = TRUE, sep = ",",
                         stringsAsFactors = TRUE)

# Rename the column names
library(data.table) # for setnames()
setnames(acadperf.data, old = c("gender","Nationality","PlaceofBirth","StageID","GradeID","SectionID",
                                "Subject","Semester","Relation","raisedhands","VisitRsrc","AnncmtView",
                                "Discussion","ParAnsSrv","ParSchSat","StudAbsnDay","Class"),
         new = c("gender","nationality","birth_place","StageID","GradeID","SectionID",
                 "subject","semester","relation","raise_hand","visit_resrc","view_anoucmt",
                 "discusion","par_ans_survy","par_schl_satsfy","absnt_day","class")
)

# Reorder the columns such that factor and numeric are separated
acadperf.data<- acadperf.data[,c(10:13,1:9,14:17)]

# STEP 2: DATA SAMPLING- Balance the variable, `gender` and `relation` are unequal
# --------------------------------------------------------------------------------
# For this paper I have applied over and under sampling as it gives the best results, so only execute the code for over and under sampling given below 
# Required dataframe: acadperf.data
# load the required libraries
library(rpart)
library(ROSE) # for accuracy.meas(), ovun.sample()
library(caret) # for createDataPartition()
# OVER AND UNDER SAMPLING 
# I have used this approach in the paper as it gives maximum accuracy
set.seed(2017)
# Balancing the gender 
data_balanced<- ovun.sample(gender ~., data = acadperf.data, method = "both", N=600, seed = 2017)$data # where N=610 because the original data has majority girls=305, so oversample the minority class until it reaches  305
# Balancing the relation 
data_balanced<- ovun.sample(relation ~., data = data_balanced, method = "both", 
                            N=600, seed = 2017)$data # where N=610 because the original data has majority girls=305, so oversample the minority class until it reaches  305
table(data_balanced$gender) # male=292 female=308
table(data_balanced$relation) # father=306 mother=294

# both (over and under sampling)
# -----
# precision: 0.770
# recall: 0.837
# F: 0.401
# # Area under the curve (AUC): 0.846

# summarize the class distribution before balancing the data
class_distr_befor <- prop.table(table(acadperf.data$class))*100
class_distr_befor
# summarize the class distribution after balancing the data
class_distr_aftr <- prop.table(table(data_balanced$class))*100
class_distr_aftr

# STEP 3: DATA VISUALIZATION for determining relationships
# ---------------------------------------------------------
# Objective: To explore relationship between predictors. The dataset used is from over and under sampling
# Required dataframe: data_balanced
# load the required libraries
library(ggplot2)
library(gridExtra) # for plotting multiple plots on one page
# BOXPLOTS  with facet_grid
## Note: For boxplots, put the categorical on the x axis and the continuous on the y. Do not try to plot, continuous on x and y axis, because it will generate a warning message: position_dodge requires non-overlapping x intervals. See this SO answer by Brian Diggs (https://stackoverflow.com/questions/8522527/position-dodge-warning-with-ggplot-boxplot)  
# reset the graphics window
par(mfrow=c(1,1))

g1<-ggplot(data = data_balanced, aes(x = gender, y = raise_hand)) + geom_boxplot()+
  # girls have more hand raises
  facet_grid(relation~class)+
  ggtitle("raise hands")+ 
  theme(plot.title = element_text(lineheight=.8, hjust = 0.5))

g2<-ggplot(data = data_balanced, aes(x = gender, y = visit_resrc)) + geom_boxplot()+
  # girls visit more resources
  facet_grid(relation~class)+
  ggtitle("visit resources")+ 
  theme(plot.title = element_text(lineheight=.8, hjust = 0.5))

g3<-ggplot(data = data_balanced, aes(x = gender, y = view_anoucmt)) + geom_boxplot()+
  # the boys are a liitle behind than girls in viewing announcements
  facet_grid(relation~class)+
  ggtitle("view announcements")+ 
  theme(plot.title = element_text(lineheight=.8, hjust = 0.5))

g4<-ggplot(data = data_balanced, aes(x = gender, y = discusion)) + geom_boxplot()+
  # girls more active than boys in discusions
  facet_grid(relation~class)+
  ggtitle("participate in discussions")+ 
  theme(plot.title = element_text(lineheight=.8, hjust = 0.5))

grid.arrange(g1,g2,g3,g4,ncol=2, nrow=2)
# reset the graphics window
par(mfrow=c(1,1))

# STEP 4: FEATURE EXTRACTION
# -----------------------------
# Check for Near Zero Variance
nzv<- nearZeroVar(data_balanced[,1:4], saveMetrics = TRUE) # No zero variance in continuous predictors.
# Feature extraction on the balanced dataset based on class
# required libraries
library(caret)
library(class)
set.seed(2017)
# reset the graphics window
par(mfrow=c(1,1))
#prepare training scheme
control<- trainControl(method = "repeatedcv", number = 10, repeats = 3)
# train the model
model<- train(class~., data = data_balanced, method="lvq", preProcess="scale",
              trControl=control)
# estimate variable importance
importance<- varImp(model, scale = FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance, main="Important Features", xlab="Student Grade Level in Exam",
     ylab="Features")
# create a vector of relevant features
# retain only those features with importance greater than 65%
relvant_features<- c("raise_hand","visit_resrc","view_anoucmt","discusion",
                     "gender","relation","par_ans_survy","par_schl_satsfy",
                     "absnt_day","class")
# subset the academic performance data containing relevant features only
acadperf.bal.relvfeat<- data_balanced[,relvant_features]

# STEP 5: PREDICTIVE MODELING
# ------------------------------
# Step A: split the data into train and test sets by 10 fold cross-validation
set.seed(2017)
# Step B: Randomly shuffle the data
data_bal_rs<-acadperf.bal.relvfeat[sample(nrow(acadperf.bal.relvfeat)),]
# Step C:  Create 10 equally size folds
folds <- cut(seq(1,nrow(data_bal_rs)),breaks=10,labels=FALSE)
# Step D: Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- data_bal_rs[testIndexes, ]
  trainData <- data_bal_rs[-testIndexes, ]
  #Use the test and train data partitions however you desire...
}
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Step E: Build Models
# Load the required libraries
library(MASS) # for LDA
library(kernlab) # for SVM
library(randomForest) # for rf
# a) linear algorithms
# LDA
fit.lda <- train(class~., data=trainData, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
fit.cart <- train(class~., data=trainData, method="rpart", metric=metric, trControl=control)
# kNN
fit.knn <- train(class~., data=trainData, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
fit.svm <- train(class~., data=trainData, method="svmRadial", metric=metric, trControl=control)
# Random Forest
fit.rf <- train(class~., data=trainData, method="rf",  metric=metric, trControl=control)
# results for a single model can be shown by the print()
print(fit.lda)
print(fit.rf)

# Step F: Select the Best Model
# summarize accuracy of models
results <- resamples(list(LDA=fit.lda, CART=fit.cart, 
                          KNN=fit.knn, SVM=fit.svm, RF=fit.rf)
)
summary(results)
# compare accuracy of models
dotplot(results)

# Step G: Make predictions
predictions<- predict(fit.rf, testData)
confusionMatrix(predictions, testData$class) # Only for writeup. Not to use in scripting

# Step H: Compute ROC AUC and plot
library(pROC)
set.seed(2017)
mypredictions<- as.numeric(predict(fit.rf, testData)) # converting to numeric because multiclass ROC works only with numeric data
roc.multi <- multiclass.roc(testData$class, mypredictions)
rf.auc= auc(roc.multi) 
rf.auc # Multi-class area under the curve: 0.9725
rs <- roc.multi[['rocs']]
plot.roc(rs[[1]], main="ROC plot")
sapply(1:length(rs),function(i) lines.roc(rs[[i]],col=i))
text(0.6,0.7,paste("AUC = ",format(rf.auc, digits=5, scientific=FALSE)))

# Step I: Plotting Random Forest
set.seed(200)
library(reprtree)
model <- randomForest(class~., data = testData, 
                      importance=TRUE, ntree=500, mtry = 2, do.trace=100)

#reprtree:::plot.getTree(model)
reprtree:::plot.getTree(model, k=1, depth=9, main="Random Forest Model") # save the plot with dimension 900 x 648