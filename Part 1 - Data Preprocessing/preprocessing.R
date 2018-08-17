#Data Processing

#Importing data
df = read.csv('Data.csv')

#Handling Missing Data
df$Age = ifelse(is.na(df$Age),
                ave(df$Age, FUN = function(x) mean(x,na.rm = TRUE)),
                df$Age)
df$Salary = ifelse(is.na(df$Salary),
                   yes = ave(df$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                   df$Salary)

#Dealing with Categorical Data
df$Country = factor(df$Country,
                    levels = c('France','Spain','Germany'),
                    labels = c(1,2,3)
                    )

df$Purchased = factor(df$Purchased,
                      levels = c('Yes','No'),
                      labels = c(1,0)
                      )

#Splitting the dataset into training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(df$Purchased,SplitRatio = 0.8)
train_set = subset(df,split == TRUE)
test_Set = subset(df,split == FALSE)

#Feature Scaling
train_set[,2:3] = scale(train_set[,2:3])
test_Set[,2:3] = scale(test_Set[,2:3])