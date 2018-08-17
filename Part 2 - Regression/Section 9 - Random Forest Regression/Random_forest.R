#Reading the data
df = read.csv('Position_Salaries.csv')
df = df[2:3]

#Fitting Random Forest Regression
# install.packages('randomForest')
set.seed(1234)
library(randomForest)
regressor = randomForest(x = df[1],
                         y = df$Salary,
                         ntree = 100)

#Predicting the value using Regression
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

#Visualisation of the Random Forest Regression
library(ggplot2)
X_grid = seq(min(df$Level),max(df$Level),0.01)
X_grid = data.frame(Level =X_grid)
ggplot()+
  geom_point(aes(x = df$Level, y = df$Salary),
             colour = 'red')+
  geom_line(aes(x = X_grid$Level, y = predict(regressor, newdata = X_grid)),
            colour = 'blue')+
  ggtitle('Level vs Salary (Random Forest Regression Model')+
  xlab('Level')+
  ylab('Salary')
