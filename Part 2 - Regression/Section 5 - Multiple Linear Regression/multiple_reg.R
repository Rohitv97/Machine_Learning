#Reading data
df = read.csv('50_Startups.csv')

#Encoding categorical data
df$State = factor(df$State,
                  levels = c('New York','California','Florida'),
                  labels = c(1,2,3)
                  )

#Splitting data into training and test data
library(caTools)
set.seed(123)
split = sample.split(df$Profit,SplitRatio = 0.8)
training = subset(df,split == TRUE)
testing = subset(df, split == FALSE)

#Fitting Regression Models
regressor = lm(formula = Profit  ~ .,
              data = training)
y_pred_test = predict(regressor,newdata = testing)
summary(regressor)

#Fittig Regression model with significant variables
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training)
y_pred_test2 = predict(regressor,newdata = testing)
