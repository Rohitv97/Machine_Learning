# Reading data from the file
df = read.csv('Position_Salaries.csv')
df = df[2:3]

# Implementing svm model
#install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ .,
                data = df,
                type = 'eps-regression',
                kernel ='radial')

#Predicting the value
test = data.frame(Level = 6.5)
y_pred = predict(regressor , newdata = test)

#Visualisation of data
library(ggplot2)
ggplot()+
  geom_point(aes(x = df$Level, y = df$Salary),
             colour = 'red')+
  geom_line(aes(x = df$Level, y = predict(regressor, newdata = df[1])),
            colour ='blue')+
  ggtitle('Level vs Salary (SVR Model)')+
  xlab('Level')+
  ylab('Salary')

  