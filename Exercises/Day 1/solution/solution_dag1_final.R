## Brain exercise

## load data
braindata <- read.table("brainweight.txt", header = T)

plot(braindata$body, braindata$brain)

cor(braindata$brain, braindata$body)
cor(braindata$brain, braindata$body, method = "spearman")


## 2.
braindata$logbrain <- log(braindata$brain)
braindata$logbody <- log(braindata$body)

plot( braindata$logbody, braindata$logbrain)

cor(braindata$logbrain, braindata$logbody)

cor(braindata$logbrain, braindata$logbody, method = "spearman")


## 3.
model <- lm(logbrain ~ logbody, data = braindata)

summary(model)

## 4+5.
plot(braindata$logbody, braindata$logbrain)
abline(model)
plot(model) ## Residuals vs. fitted og qq-plot


### comparison: non-transformed:
model <- lm(brain ~ body, data = braindata)
plot(model) ## Residuals vs. fitted og qq-plot. Looks horrible. Good that we did a transformation!


#### Laborforce exercise
labor <- read.table("labor.txt", header = T)

boxplot(cbind(labor$x1968, labor$x1972))
boxplot(labor$x1968, labor$x1972)

# 1.
t.test(labor$x1968, labor$x1972, var.equal = T)

# 2.
t.test(labor$x1968, labor$x1972, var.equal = T, paired = T)

# 4.
mean(labor$x1972 - labor$x1968) ## can also be read from the t test.  




#### Log transformation exercise
## log of y

x <- seq(1.5, 2.5, length = 100) ## make x's
logy <- 3*x + rnorm(100, sd = 0.15) ## make log y_i s
y <- exp(logy) ## transform
plot(x,y)
L <- lm(log(y) ~ x)
summary(L)
coef(L)
coef_alpha <- coef(L)[1]
coef_beta <- coef(L)[2]
fit_model <- exp(coef_alpha +  coef_beta*x)
lines(x, fit_model)

## log-log
logx <- seq(1.5, 2.5, length = 100) ## make x's
logy <- 3*logx + rnorm(100, sd = 0.15) ## make log y_i s
y <- exp(logy) ## transform
x <- exp(logx) ## transform
plot(x,y)

L <- lm(log(y) ~ log(x))
summary(L)
coef(L)
coef_alpha <- coef(L)[1]
coef_beta <- coef(L)[2]

fit_model <- exp(coef_alpha +  coef_beta*log(x))
lines(x, fit_model)

## log of x
logx <- seq(1.5, 2.5, length = 100) ## make x's
y <- 3*logx + rnorm(100, sd = 0.15) ## make log y_i s
x <- exp(logx) ## transform
plot(x,y)

L <- lm(y ~ log(x))
summary(L)
coef(L)
coef_alpha <- coef(L)[1]
coef_beta <- coef(L)[2]

fit_model <- coef_alpha +  coef_beta*log(x)
lines(x, fit_model)


###### Calcium
calc <- read.table("calcium.txt", header = T)
View(calc)

## Two sample setting 

# 2
qqnorm(calc$Decrease[1:9], main = "Calcium")
qqline(calc$Decrease[1:9], main = "Calcium")

qqnorm(calc$Decrease[10:21], main = "Placebo")
qqline(calc$Decrease[10:21], main = "Placebo")


# 3
?var.test

var.test(Decrease ~ Treatment, data = calc)

x<-calc$End[calc$Treatment=="Calcium"]
y<-calc$End[calc$Treatment=="Placebo"]

var.test(x,y)

# 4
boxplot(calc$Decrease ~ calc$Treatment)

# 5
t.test(calc$Decrease ~ calc$Treatment, var.equal = T)

# 6 
wilcox.test(calc$Decrease ~ calc$Treatment)


?colnames

