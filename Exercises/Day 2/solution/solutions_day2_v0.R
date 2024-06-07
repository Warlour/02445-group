##### Kalkdata

kalk <- read.table("kalk.txt", header = T)

## 75% to training data, 25% to test data
train_ids <- 1:30 ## slightly more correct to pick train/test set at random

# Split data
kalk_train <- kalk[train_ids, ]
kalk_test <- kalk[- train_ids, ]

L <- lm(tit2 ~ tit1, data = kalk_train)

predictions <- predict(L, kalk_test)
mean((kalk_test$tit2 - predictions)^2) ## Mean squared prediction error


##
# Choose some training/test sizes. 
# We choose 2,4,8,12,16,20,24,28,32,36,37,38,39
train_sizes <- c(2,4,8,12,16,20,24,28,32,34,36,37,38,39)

train_error <- rep(NA, length(train_sizes))
test_error <- rep(NA, length(train_sizes))

for (i in 1:length(train_sizes)) {
  # Random each time
  train_ids <- sample(40, train_sizes[i])
  
  # Split data
  kalk_train <- kalk[train_ids, ]
  kalk_test <- kalk[- train_ids, ]
  
  L <- lm(tit2 ~ tit1, data = kalk_train)
  
  # Residuals are just data-predictions on the training set.
  # So mean(residuals^2) must be training error
  train_error[i] <- sqrt(mean(residuals(L)^2))
  
  predictions <- predict(L, kalk_test)
  test_error[i] <- sqrt(mean((kalk_test$tit2 - predictions)^2)) ## Mean squared prediction error
  
}

# smart plot
matplot(train_sizes, cbind(train_error, test_error), 
        col = c(2,3), pch = 16, type = "b", ylab = "sqrt(MSE)")
# Note that due to very few data points, the test errors gets 'quite random' in the end.




##
# Choose some training/test sizes. 
# We choose 2,4,8,12,16,20,24,28,32,36,37,38,39
train_sizes <- c(2,4,8,12,16,20,24,28,32,34,36,37,38,39)

train_error <- rep(NA, length(train_sizes))
test_error <- rep(NA, length(train_sizes))

for (i in 1:length(train_sizes)) {
  # Random each time
  train_ids <- sample(40, train_sizes[i])
  
  # Split data
  kalk_train <- kalk[train_ids, ]
  kalk_test <- kalk[- train_ids, ]
  
  L <- lm(tit2 ~ tit1, data = kalk_train)
  
  # Residuals are just data-predictions on the training set.
  # So mean(residuals^2) must be training error
  train_error[i] <- sqrt(mean(residuals(L)^2))
  
  predictions <- predict(L, kalk_test)
  test_error[i] <- sqrt(mean((kalk_test$tit2 - predictions)^2)) ## Mean squared prediction error
  
}

# smart plot
matplot(train_sizes, cbind(train_error, test_error), 
        col = c(2,3), pch = 16, type = "b", ylab = "sqrt(MSE)")
# Note that due to very few data points, the test errors gets 'quite random' in the end.



#### Sheet manufactoring

sheet.data <- read.table("sheets.txt", header = T)

## Convert day & machine to factors:
sheet.data$day <- as.factor(sheet.data$day)
sheet.data$machine <- as.factor(sheet.data$machine)

## Test for interactions:
L <- lm(log.permeability ~ day * machine, data = sheet.data)

drop1(L, test = "F") ## Interaction not significant


## Predict on new sheet (from same machines on same days):
# Loop through all data (leave-one-out CV)
pred.errors <- rep(NA, 81)
for (i in 1:81) {
  train_data <- sheet.data[-i, ] 
  test_data <- sheet.data[i, ]
  
  ## Which model to use? We haven't discussed that, but let's use the additive model
  L <- lm(log.permeability ~ day + machine, data = train_data)
  prediction <- predict(L, test_data)
  pred.errors[i] <- test_data$log.permeability - prediction
}
mse1 <- mean(pred.errors^2)

## Predict on a new sheet (from a different/new machine on same days):

# Cross-validation on machines
pred.errors <- rep(NA, 3)
for (i in 1:3) {
  train_data <- subset(sheet.data, machine != i)
  test_data <- subset(sheet.data, machine == i)
  
  ## We are predicting on a different machine, so we can't include machine in our model.
  L <- lm(log.permeability ~ day, data = train_data)
  prediction <- predict(L, test_data)
  pred.errors[i] <- mean((test_data$log.permeability - prediction)^2)
}
mse2 <- mean(pred.errors)

## Predict on a new sheet (from same machines on a different day):

# Cross-validation on days
pred.errors <- rep(NA, 9)
for (i in 1:9) {
  train_data <- subset(sheet.data, day != i)
  test_data <- subset(sheet.data, day == i)
  
  ## We are predicting on a different machine, so we can't include machine in our model.
  L <- lm(log.permeability ~ machine, data = train_data)
  prediction <- predict(L, test_data)
  pred.errors[i] <- mean((test_data$log.permeability - prediction)^2)
}
mse3 <- mean(pred.errors)

## Finally, predict on a new sheet (from different machine on a different day):

# Training set: include days != i and machines != j
# Test set: day == i and machine == j
pred.errors <- matrix(NA, 9, 3)
for (i in 1:9) {
  for (j in 1:3) {
    train_data <- subset(sheet.data, day != i & machine != j)
    test_data <- subset(sheet.data, day == i & machine == j)
    
    ## Neither day nor machine can be included in the model. Use raw mean.
    prediction <- mean(train_data$log.permeability)
    pred.errors[i,j] <- mean((test_data$log.permeability - prediction)^2)
    ## or
    L <- lm(log.permeability ~ 1, data = train_data)
    prediction <- predict(L, test_data)
    pred.errors[i,j] <- mean((test_data$log.permeability - prediction)^2)
  }}

mse4 <- mean(pred.errors)

## Which numbers are interestering for the manager?
## Same days can't be repeated, so prediction on a new day would be interesting, 
## If manager were to invest in a new machine, then prediction on new machine, new day would interesting. 


#### Binomial bootstrapping

x <- 19
n <- 100
## Estimate of p. 
hatp <- x / n
## Estimator: \hat{p}(x) = x/n

## Approx CI using formula:
c(hatp - sqrt(hatp*(1-hatp) / n) * 1.96, hatp +sqrt(hatp*(1-hatp) / n) * 1.96)


## Param. bootstrap
# 1000 replications
sims <- rbinom(1000, n, hatp) / 100

## Histogram
hist(sims)
## 95% CI
quantile(sims, c(0.025, 0.975))



x <- 99
n <- 100
## Estimate of p. 
hatp <- x / n
## Estimator: \hat{p}(x) = x/n

## Approx CI using formula:
c(hatp - sqrt(hatp*(1-hatp) / n) * 1.96, hatp +sqrt(hatp*(1-hatp) / n) * 1.96)


## Param. bootstrap
# 1000 replications
sims <- rbinom(1000, n, hatp) / 100

## Histogram
hist(sims)
## 95% CI
quantile(sims, c(0.025, 0.975))



### Humerus Non-paramteric versus parametric BS 
hum<-read.table("humerus.txt",header=TRUE)
summary(hum)

hum2<-split(hum$humerus,hum$code)
names(hum2)<-c("doa","alive")

hum$code <- factor(hum$code)
hum$code2 <- factor(hum$code, labels = c("doa","alive"))
str(hum)
summary(hum)

par(mfrow = c(1,1))
plot(humerus~code2,hum)


### Bootstrap:

k <- 10000
simx_samples <- replicate(k, sample(hum2$doa, replace = TRUE))
simy_samples <- replicate(k, sample(hum2$alive, replace = TRUE))

sim_mean_doa <- apply(simx_samples, 2, mean)
sim_mean_alive <- apply(simy_samples, 2, mean)
sim_mean_difs <- apply(simx_samples, 2, mean) - apply(simy_samples, 2, mean)

## One can also do a for loop instead. That might be a little more readable :)


hist(sim_mean_doa)
quantile(sim_mean_doa, probs = c(0.025, 0.975))
abline(v = quantile(sim_mean_doa, probs = c(0.025, 0.975)), col = 2, lty = 3, lwd = 3)


hist(sim_mean_alive)
quantile(sim_mean_alive, probs = c(0.025, 0.975))
abline(v = quantile(sim_mean_alive, probs = c(0.025, 0.975)), col = 2, lty = 3, lwd = 3)


hist(sim_mean_difs)
quantile(sim_mean_difs, probs = c(0.025, 0.975))
abline(v = quantile(sim_mean_difs, probs = c(0.025, 0.975)), col = 2, lty = 3, lwd = 3)

## confidence using normality assumption: 
t.test(hum2$doa)$conf
t.test(hum2$alive)$conf
t.test(hum2$doa,hum2$alive)$conf

