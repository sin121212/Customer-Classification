library("ggplot2")
library("erer")
library("e1071")
library("caret")
library("tidyverse")
library("devtools")
library("nnet")
library("clusterGeneration")
library("gamlss.add")
library("dplyr")

######### load data #########
load("C:/Users/user/Desktop/cityu/Sem2/MS6218 Statistical Modelling in Marketing Engineering/Final_Project/catalog_data.RData")

# Split data
set.seed(2019)
L = nrow(catalog_DF)
ind = sample(1:L, L/2)
catalog_DF$validation_sample = 0
catalog_DF$validation_sample[ind] = 1
# 2 dataset
train_df <- subset(catalog_DF,validation_sample==0)
valid_df <- subset(catalog_DF,validation_sample==1)
colnames(catalog_DF)
# convert to factor
train_df[,2] <- factor(train_df[,2])
valid_df[,2] <- factor(valid_df[,2])

# 1.1:logistic regression (only estimation sample)
logit <- glm(formula = buytabw ~. -customer_no - validation_sample,
    family = binomial(link="logit"),
    data=train_df,x=TRUE)
summary(logit)

plot(logit)

# confusionMatrix
predicts <- as.numeric(logit$fitted.values >= 0.5)
CN_logit <- confusionMatrix(table(predicts,train_df$buytabw))
# marginal effect in log model
logit_eff <- maBina(logit,x.mean = TRUE,rev.dum = TRUE, digits = 4)
logit_eff

# 1.2: 
# NN
TrainingParameters <- trainControl(method = "repeatedcv", number = 5, repeats=2)
NN_model <- train(buytabw ~. -customer_no - validation_sample,data = train_df,
                 method = "nnet",
                 trControl= TrainingParameters,
                 preProcess=c("scale","center"),
                 na.action = na.omit)
NN_model
plot(NN_model)
# NN in train data
NN_train <-predict(NN_model, train_df)
# Create confusion matrix
CM_NN <-confusionMatrix(NN_train, train_df$buytabw)
CM_NN


# 2.1 :fit model in validation data
# logit
probabilities <- logit %>% predict(valid_df, type = "response")
min(probabilities)
max(probabilities)
# confusionMatrix
predicts <- as.numeric(probabilities >= 0.5)
confusionMatrix(table(predicts,valid_df$buytabw))

# NN
NN_valid <-predict(NN_model, valid_df)
# Create confusion matrix
CM_NN <-confusionMatrix(NN_valid, valid_df$buytabw)
CM_NN
# Prob
NN_valid_prob <- predict(NN_model, valid_df, type='prob')
min(NN_valid_prob)
max(NN_valid_prob)


# 3: Box plot of predicted purchase probabilities
valid_df$logit_prob = probabilities
valid_df$NN_prob = NN_valid_prob
valid_df$NN_pred = NN_valid

# logit
par(cex = 0.80)
boxplot(logit_prob ~ buytabw, data = valid_df,
        col = "hotpink1",
        xlab = "Customer did not buy (0) or bought (1)",
        ylab = "Predicted purchase probability",
        main = "logistic regression ")

# NN
par(cex = 0.80)
boxplot(valid_df$NN_prob[,2] ~ buytabw, data = valid_df,
        col = "blue",
        xlab = "Customer did not buy (0) or bought (1)",
        ylab = "Predicted purchase probability",
        main = "Neural Networks")

# 4:Scoring and segmentation
# N, the number of bins (groups) to create
createBins <- function(x, N) {
  cut_points = quantile(x, probs = seq(1/N, 1 - 1/N, by = 1/N), type = 2)
  cut_points = unique(cut_points)
  bins = cut(x, c(-Inf, cut_points, +Inf), label = 1:(length(cut_points) + 1))
  return(as.numeric(bins))
}

# logit
logit_bin <- createBins(valid_df$logit_prob,10)
valid_df$logit_bin <- logit_bin
# SS refer to Scoring and segmentation 
SS_logit_bin <-  valid_df %>%
  group_by(logit_bin) %>%
  summarize(No_of_observation = n(),
            No_of_buyers = sum(buytabw==1),
            Mean_predicted_purchase_prob = mean(logit_prob),
            Mean_observed_purchase_rate=sum(buytabw==1)/n() )
# Sorting
SS_logit_bin <- SS_logit_bin[order(-SS_logit_bin$logit_bin),]

# NN
NN_bin <- createBins(valid_df$NN_prob[,2],10)
valid_df$NN_bin <- logit_bin
# SS refer to Scoring and segmentation 
SS_NN_bin <-  valid_df %>%
  group_by(NN_bin) %>%
  summarize(No_of_observation = n(),
            No_of_buyers = sum(buytabw==1),
            Mean_predicted_purchase_prob = mean(NN_prob[,2]),
            Mean_observed_purchase_rate=sum(buytabw==1)/n() )

# Sorting
SS_NN_bin <- SS_NN_bin[order(-SS_NN_bin$NN_bin),]


# 5. Lift and gains
# logic
logit_avg_pre_purchase_prob <- mean(valid_df$logit_prob)
logit_avg_pre_purchase_prob
# LG refer to Lift and gains
LG_logit <- SS_logit_bin
# lift
LG_logit$lift = 100 * LG_logit$Mean_predicted_purchase_prob / logit_avg_pre_purchase_prob
# cum lift
logit_cum_pre_purchase_prob <- vector()
for (i in 1:10){
  logit_cum_pre_purchase_prob[i] <- mean(LG_logit$Mean_predicted_purchase_prob[1:i])
}
LG_logit$cum_lift = 100 * logit_cum_pre_purchase_prob / logit_avg_pre_purchase_prob
# cum gain
logit_cum_gain <- vector()
for (i in 1:10){
  logit_cum_gain[i] <- LG_logit$cum_lift[i] * (i/10)
}
LG_logit$cum_gain = logit_cum_gain
#plot lift
plot(LG_logit$logit_bin,LG_logit$lift,type='o', col='red',
     ylim=c(0,350), xlab = "Score", ylab='Lift',
     main="Lift of Logistic Regression")
#plot cum lift
plot(seq(10,100,by=10),LG_logit$cum_lift,type='o', col='red',
     ylim=c(0,350), xlab = "Cum % of customers mailed", ylab='Cumulative Lift',
     main="Cumulative Lift of Logistic Regression")
#plot cum gain
plot(seq(10,100,by=10),LG_logit$cum_gain,type='o', col='red',
     ylim=c(0,100), xlab = "Cum % of customers mailed", ylab='Cumulative Gain',
     main="Cumulative Gain of Logistic Regression")
lines(seq(10,100,by=10),seq(10,100,by=10), col='blue')
legend("bottomright", legend=c("% of customers captured by target mailing","random mailing"),
       col=c("red",'blue'), lty=1, cex=0.75)

# NN
NN_avg_pre_purchase_prob <- mean(valid_df$NN_prob[,2])
NN_avg_pre_purchase_prob
# LG refer to Lift and gains
LG_NN <- SS_NN_bin
# lift
LG_NN$lift = 100 * LG_NN$Mean_predicted_purchase_prob / NN_avg_pre_purchase_prob
# cum lift
NN_cum_pre_purchase_prob <- vector()
for (i in 1:10){
  NN_cum_pre_purchase_prob[i] <- mean(LG_NN$Mean_predicted_purchase_prob[1:i])
}
LG_NN$cum_lift = 100 * NN_cum_pre_purchase_prob / NN_avg_pre_purchase_prob
# cum gain
NN_cum_gain <- vector()
for (i in 1:10){
  NN_cum_gain[i] <- LG_NN$cum_lift[i] * (i/10)
}
LG_NN$cum_gain = NN_cum_gain
#plot lift
plot(LG_NN$NN_bin,LG_NN$lift,type='o', col='red',
     ylim=c(0,350), xlab = "Score", ylab='Lift',
     main="Lift of Neural Networks")
#plot cum lift
plot(seq(10,100,by=10),LG_NN$cum_lift,type='o', col='red',
     ylim=c(0,350), xlab = "Cum % of customers mailed", ylab='Cumulative Lift',
     main="Cumulative Lift of Neural Networks")
#plot cum gain
plot(seq(10,100,by=10),LG_NN$cum_gain,type='o', col='red',
     ylim=c(0,100), xlab = "Cum % of customers mailed", ylab='Cumulative Gain',
     main="Cumulative Gain of Neural Networks")
lines(seq(10,100,by=10),seq(10,100,by=10), col='blue')
legend("bottomright", legend=c("% of customers captured by target mailing","random mailing"),
       col=c("red",'blue'), lty=1, cex=0.75)

# 6: Profitability analysis
# expected profit
# logit
valid_df$logit_expected_profit = (valid_df$logit_prob * 26.9) - 1.4
# plot
hist(valid_df$logit_expected_profit,col='hotpink1',
     xlab = "Excerted Profit", ylab='Frequency',
     main="Expected Profit Base on Logistic Regression")

summary(valid_df$logit_expected_profit)

# + or - profit?
logit_sign_profit <- table(sign(valid_df$logit_expected_profit))
logit_sign_profit

# Pie Chart with Percentages
slices <- c(logit_sign_profit[1],logit_sign_profit[2]) 
lbls <- c("Negative (3103)", "Positive (6897)")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels 
lbls <- paste(lbls,"%",sep="") # ad % to labels 
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Customers Profitable of Logistic Regression")
# rank by predicted profit
valid_df = valid_df[order(-valid_df$logit_prob),]
# realized profits
valid_df$logit_actual_profit = ((as.numeric(valid_df$buytabw)-1) * 26.9) - 1.4
#plot cumulative realized profits
plot(seq(0.01,100,by=0.01),cumsum(valid_df$logit_actual_profit),type='o', col='red',
         xlab = "Cum % of customers mailed", ylab='Cumulative Actual Profit',
         main="Logistic Regression: Cumulative Actual Profit vs % of customers mailed")
points(x=68.7,y=max(cumsum(valid_df$logit_actual_profit)),col="blue",bg="blue",pch=24,cex=2.5)
points(x=100,y=cumsum(valid_df$logit_actual_profit)[10000],col="green",bg="green",pch=25,cex=2.5)

# maximum point
max(cumsum(valid_df$logit_actual_profit))
# tail point
cumsum(valid_df$logit_actual_profit)[10000]


# expected profit
# NN
valid_df$NN_expected_profit = (valid_df$NN_prob[,2] * 26.9) - 1.4
# plot
hist(valid_df$NN_expected_profit,col='blue',
     xlab = "Excerted Profit", ylab='Frequency',
     main="Expected Profit Base on Neural Networks")

summary(valid_df$NN_expected_profit)

# + or - profit?
NN_sign_profit <- table(sign(valid_df$NN_expected_profit))
NN_sign_profit

# Pie Chart with Percentages
slices <- c(NN_sign_profit[1],NN_sign_profit[2]) 
lbls <- c("Negative (3513)", "Positive (6487)")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels 
lbls <- paste(lbls,"%",sep="") # ad % to labels 
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Customers Profitable of Neural Networks")
# rank by predicted profit
valid_df = valid_df[order(-valid_df$NN_prob[,2]),]
# realized profits
valid_df$NN_actual_profit = ((as.numeric(valid_df$buytabw)-1) * 26.9) - 1.4
#plot cumulative realized profits
plot(seq(0.01,100,by=0.01),cumsum(valid_df$NN_actual_profit),type='o', col='red',
     xlab = "Cum % of customers mailed", ylab='Cumulative Actual Profit',
     main="Neural Networks: Cumulative Actual Profit vs % of customers mailed")
points(x=64.87,y=max(cumsum(valid_df$NN_actual_profit)),col="blue",bg="blue",pch=24,cex=2.5)
points(x=100,y=cumsum(valid_df$NN_actual_profit)[10000],col="green",bg="green",pch=25,cex=2.5)

# maximum point
max(cumsum(valid_df$NN_actual_profit))
# tail point
cumsum(valid_df$NN_actual_profit)[10000]

# 7 Recommended targeting strategy
# Actual Expected profit 
actual_expected_profit <- round(mean((as.numeric(valid_df$buytabw)-1) * 26.9 - 1.4),4)
actual_expected_profit
# logit Expected profit 
logit_expected_profit <- round(mean(valid_df$logit_expected_profit),4)
logit_expected_profit
# NN Expected profit
NN_expected_profit <- round(mean(valid_df$NN_expected_profit),4)
NN_expected_profit

# Create data frame to compare expected profit among 2 models and actual 
compare_profit_df <- data.frame(Model=c("Actual","Logistic Regression","Neural Networks"),
                             Expected_Profit=c(actual_expected_profit,logit_expected_profit,NN_expected_profit))
compare_profit_df$Different_From_Actual = abs(compare_profit_df$Expected_Profit - actual_expected_profit)

# logit conditional prob
logit_prob_df <- valid_df %>%
  group_by(buytabw) %>%
  summarize(con_prob = mean(logit_prob))
# Predict incremental volumn (PIV)
logit_PIV <- logit_prob_df$con_prob[2] - logit_prob_df$con_prob[1]
logit_PIV
# ROI
logit_ROI <- (logit_PIV * 26.9 - 1.4)/1.4
logit_ROI

# NN conditional prob
NN_prob_df <- valid_df %>%
  group_by(buytabw) %>%
  summarize(con_prob = mean(NN_prob[,2]))
# Predict incremental volumn (PIV)
NN_PIV <- NN_prob_df$con_prob[2] - NN_prob_df$con_prob[1]
NN_PIV
# ROI
NN_ROI <- (NN_PIV * 26.9 - 1.4)/1.4
NN_ROI



