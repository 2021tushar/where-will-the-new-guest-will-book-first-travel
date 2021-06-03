library("data.table")
library(tidyr)
library(bit64)
library(Matrix)
library(xgboost)
library("ggplot2")
#set as current directory
setwd("C:\\Users/Sachin/Desktop/airbnb-new-user-bookings-master/airbnb-new-user-bookings-master")
#Load training and testing data
train = fread("train_users_2.csv")
test = fread("test_users.csv")

#Load sessions data
sessions = fread("sessions.csv")
user_actions = sessions[, list(count=.N), by=c("user_id", "action")]
user_actions = spread(data=user_actions, key=action, value=count, fill=0, sep="_")

#Count number of each action_detail by user_id
user_action_details = sessions[, list(count=.N), by=c("user_id", "action_detail")]
user_action_details = spread(data=user_action_details, key=action_detail, value=count, fill=0, sep="_")

#Remove unused data to clear memory
rm(sessions)
gc()

#Convert timestamp_first_active into date format
train[, date_first_active := paste(substr(timestamp_first_active, 0, 4), substr(timestamp_first_active, 5, 6),
                                   substr(timestamp_first_active, 7, 8), sep="-")]
test[, date_first_active := paste(substr(timestamp_first_active, 0, 4), substr(timestamp_first_active, 5, 6),
                                  substr(timestamp_first_active, 7, 8), sep="-")]

#No session data before 2014-01-01 - remove this training data
train = train[date_first_active >= 20140101000000]

#Convert to date
train[, date_first_active := as.Date(date_first_active)]
test[, date_first_active := as.Date(date_first_active,"%y %m %d")]

#Add country_destination in order to combine training and testing data
test[, country_destination := "Test"]

#Combine training and testing data
data = rbind(train, test,fill=TRUE)

#Remove unused data to clear memory
rm(train, test)
gc()

#Set keys for joining
setkey(data, id)
setkey(user_actions, user_id)
setkey(user_action_details, user_id)

#Join data
data = user_actions[data]
data = user_action_details[data]

#Remove unused data to clear memory
rm(user_actions, user_action_details)

#Set NAs in merged data to missing value for xgboost (no session counts)
data[is.na(data)] = -999

#Observe age distribution
hist(data[, age])
summary(data[, age])
ggplot(data, aes(age)) + 
  geom_histogram(breaks=seq(0, 130, by=2), 
                 col="red", 
                 fill="green", 
                 alpha = .2)
#Clean age - set to missing value
data[, age := ifelse(age < 15, -999, age)]
data[, age := ifelse(age > 104, -999, age)]

hist(data[, age])
summary(data[, age])
ggplot(data, aes(age)) + 
  geom_histogram(breaks=seq(0, 130, by=2), 
                 col="red", 
                 fill="green", 
                 alpha = .2)
# #ggplot(data, 
#        aes(x = factor(affiliate_provider),fill = factor(affiliate_provider))) + 
#   geom_bar(stat = "count") + 
#   scale_y_continuous(breaks = seq(0,12,3), labels = c("0", "25%", "50%", "75%", "100%")) + 
#   coord_polar(theta='y') +
#   theme(axis.text.y = element_blank(), 
#         axis.title.y = element_blank(), 
#         axis.ticks.y = element_blank(),
#         axis.title.x = element_blank()) +
#   labs(fill = "affiliate_provider")
#graph for affilate provider
ggplot(data, 
       aes(x = (affiliate_provider),fill = factor(affiliate_provider))) + 
  geom_bar(stat = "count")
#affilate tracked
ggplot(data, aes(first_affiliate_tracked
)) + 
  geom_histogram(stat="count")
#affilate tracked
ggplot(data, aes(first_affiliate_tracked
)) + 
  geom_bar(stat='count')
#signup method
ggplot(data, 
       aes(x = (signup_method
),fill = factor(signup_method
))) + 
  geom_bar(stat = "count")
#affilate channel
ggplot(data, 
       aes(x = (affiliate_channel),
       fill = factor(affiliate_channel
       ))) + 
  geom_bar(stat = "count")
#gender
# ggplot(data, aes(x = (gender),
#            fill = factor(gender))) + 
#   geom_bar(stat = "count")
#first device
ggplot(data, 
       aes(x = (first_device_type),
           fill = factor(first_device_type ))) + 
  geom_bar(stat = "count")
#Convert data types
data[, date_account_created := as.Date(date_account_created)]
data[, month_account_created := as.integer(format(date_account_created, "%m"))]
data[, month_first_active := as.integer(format(date_first_active, "%m"))]

#Get training and testing indices
train_indices = data[, .I[country_destination != "Test"]]
test_indices = data[, .I[country_destination == "Test"]]

#Add variable for days available for user to book based on maximum date in the data set
data[train_indices, DaysAvailable := as.numeric(as.Date("2014-06-30") - date_account_created)]
data[test_indices, DaysAvailable := as.numeric(as.Date("2014-09-30") - date_account_created)]

#Extract target variable
train_country_destination = data[train_indices, country_destination]

#Remove columns
data[, c("date_account_created", "date_first_active", "timestamp_first_active", "date_first_booking",
         "country_destination") := NULL]

#Set NAs to missing value
data[is.na(data)] = -999

#Make valid variable names
names(data) = make.names(names(data))

#Sparsify data
data = sparse.model.matrix(~. -1, data)


#Split training and testing data
train = data[train_indices]
test = data[test_indices]

#Remove unused data to clear memory
rm(data)
gc()

#Recode target to numeric for xgboost
library(car)
train_country_destination = recode(train_country_destination,
                                   "'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6;
                                   'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11;")

#Extract names of training variables
train_names = names(train)

#Define evaluation metric
NDCG5 = function(preds, dtrain) {
  labels = getinfo(dtrain,"label")
  num.class = 12
  pred = matrix(preds, nrow = num.class)
  top = t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x = ifelse(top == labels,1,0)
  dcg = function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg = mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
}

#Set best parameters
param = list(objective = "multi:softprob",
             num_class = 12,
             booster = "gbtree",
             eta = 0.2,
             max_depth = 6,
             subsample = 0.85,
             colsample_bytree = 0.66
)

#Create xgb.DMatrix for xgboost
dtrain = xgb.DMatrix(data=data.matrix(train), label=train_country_destination, missing = -999)

#Set early.stop.round for xgboost
early.stop.round = 30

#Run 5-fold cross validation xgboost
set.seed(8)
XGBcv = xgb.cv(params = param,
               data = dtrain, 
               nrounds = 200,
               verbose = 1,
               early.stop.round = early.stop.round,
               nfold = 5,
               feval = NDCG5,
               maximize = T,
               prediction = T
)

#Extract nrounds of best iteration
nrounds = length(XGBcv$dt$test.ndcg5.mean) - early.stop.round

#Run xgboost on full data set
XGB = xgboost(params = param,
              data = dtrain,
              nrounds = nrounds,
              verbose = 1,
              eval_metric = NDCG5,
              maximize = T
)

# Compute and plot feature importance
Importance = xgb.importance(trainNames, model = XGB)
xgb.plot.importance(Importance[1:10,])
#Predict on test set
PredTest = predict(XGB, data.matrix(test[,trainNames]), missing = -999)

#Reshape predictions
Predictions = as.data.frame(matrix(PredTest, nrow=12))

#Recode to original destination names
rownames(Predictions) = c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')

#Extract top 5 predictions for each testing row
Predictions_Top5 = as.vector(apply(Predictions, 2, function(x) names(sort(x)[12:8])))

#Create test set ID vector
testId = testMerged$id
testIdMatrix = matrix(testId, 1)[rep(1,5), ]
testIds = c(testIdMatrix)

#Create submission
Submission = NULL
Submission$id = testIds
Submission$country = Predictions_Top5

#Save submission
Submission = as.data.frame(Submission)
write.csv(Submission, "XGB.csv", quote=F, row.names=F)
h=fread("XB1.csv")

ggplot(h, aes(x = factor(country),fill = factor(country))) + 
     geom_bar(stat = "count")
