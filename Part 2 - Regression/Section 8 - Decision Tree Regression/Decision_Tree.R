#Reading the data
df = read.csv('Position_Salaries.csv')
df = df[2:3]

#Fitting Decision Tree Regressor to the data
#install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = df,
                  control = rpart.control(minsplit = 1))

#Predicting using Decision Tree
test = data.frame(Level = 6.5)
y_pred = predict(regressor, newdata = test)

#Visualising the Decision Tree Regression
library(ggplot2)
X_grid = seq(min(df$Level), max(df$Level), 0.01)
ggplot()+
  geom_point(aes(x = df$Level, y = df$Salary),
             colour = 'red')+
  geom_line(aes(x = X_grid, y = predict(regressor, 
                                        newdata = data.frame(Level = X_grid))),
            colour = 'blue')+
  ggtitle('Level vs Salary (Decision Tree Regression Model)')+
  xlab('Level')+
  ylab('Salary')
