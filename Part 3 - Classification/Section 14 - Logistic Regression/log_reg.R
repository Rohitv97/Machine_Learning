#importing dataset
df = read.csv('Social_Network_Ads.csv')
df = df[3:5]

#Splitting data into training and test data
library(caTools)
set.seed(1234)
split = sample.split(df$Purchased, SplitRatio = 0.75)
training = subset(df, split == TRUE)
test = subset(df, split == FALSE)

#Feature scaling
training[1:2] = scale(training[1:2])
test[1:2] =  scale(test[1:2])

#Fitting Model
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training)

#Predicting the results
prob_pred = predict(classifier, type = 'response', newdata = test[-3])
y_pred = ifelse(prob_pred > 0.5,
                1,
                0)

#Generating confusion matrix
cm = table(test[,3],y_pred)

#Visualisation of the result
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training
X1 = seq(min(set[,1]) - 1, max(set[,1]) + 1, 0.01)
X2 = seq(min(set[,2]) - 1, max(set[,1]) + 1, 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age','EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1 ,0)
plot(set[,-3],
     main = 'Logistic Regression(Training Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.',col = ifelse(y_grid == 1, 'springgreen','tomato'))
points(set[,-3], pch =21, bg = ifelse(set[,3] == 1, 'green4', 'red3'))

#Visualisation of the result (test set)
set = test
X1 = seq(min(test[,1]) - 1, max(test[,1] + 1), 0.01)
X2 = seq(min(test[,2]) - 1, max(test[,2] + 1), 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age','EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[,-3],
     main = 'Logistic Regression(Test Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)),add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3] == 1, 'green4','red4'))