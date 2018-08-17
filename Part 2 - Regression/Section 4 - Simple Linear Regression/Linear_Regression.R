#Reading data 
df = read.csv('Salary_Data.csv')

#Splitting data into training and testing datasets
library(caTools)
set.seed(123)
split = sample.split(Y = df$Salary, SplitRatio = 2/3)
training = subset(df, split == TRUE)
testing = subset(df, split == FALSE)

#Creating the linear model
model = lm(formula = Salary ~ YearsExperience, data = training)
y_pred = predict(model, newdata = testing )
y_pred_train = predict(model, newdata = training)

#installing ggplot2 package
#install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = training$YearsExperience, y = training$Salary),
             colour = 'red')+
  geom_line(aes(x = training$YearsExperience, y = y_pred_train),
            colour = 'blue')+
  ggtitle('Salary vs Years of Experience(training data')+
  xlab('Years of Experience')+
  ylab('Salary')

options(scipen = 100)
ggplot()+
    geom_point(aes(x = testing$YearsExperience, y = testing$Salary),
               colour = 'red')+
    geom_line(aes(x = testing$YearsExperience, y = y_pred),
              colour = 'blue')+
    ggtitle('Salary vs Years of Experience(Testing data)')+
    ylab('Salary')+
    xlab('Years of Experience')
