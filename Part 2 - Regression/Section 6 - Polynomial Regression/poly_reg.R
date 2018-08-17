df = read.csv('Position_Salaries.csv')
df = df[2:3]

#Linear Model
lin_reg = lm(formula = Salary ~ .,
             data = df)

test = data.frame(Level = 6.5)
y_pred = predict(lin_reg, newdata = test)

#Polynomial Model
df$Level2 = df$Level ^ 2
df$Level3 = df$Level ^ 3
df$Level4 = df$Level ^ 4
poly_reg = lm(formula = Salary ~ .,
              data =df)
test$Level2 = test$Level ^ 2
test$Level3 = test$Level ^ 3
test$Level4 = test$Level ^ 4
predict(poly_reg, test)

#Visualising Linear Model
library(ggplot2)
ggplot()+
  geom_point(aes(x = df$Level, y = df$Salary),
             colour = 'red')+
  geom_line(aes(x = df$Level, y = predict(lin_reg, newdata =df)),
            colour = 'blue')+
  ggtitle('Level vs Salary(Linear Model)')+
  xlab('Level')+
  ylab('Salary')

#Visualising Polynomial Model
library(ggplot2)
ggplot()+
  geom_point(aes(x = df$Level, y = df$Salary),
             colour = 'red')+
  geom_line(aes(x = df$Level, y = predict(poly_reg, newdata = df)),
            colour = 'blue')+
  ggtitle('Level vs Salary(Polynomial Model)')+
  xlab('Level')+
  ylab('Salary')
