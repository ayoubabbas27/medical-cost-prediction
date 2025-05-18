library(caret)

# Load Dataset
INS <- read.csv("insurance.csv")
head(INS)

# Check missing data
sum(is.na(INS))

# Preprocess Categorical Variables
INS$sex <- as.factor(INS$sex)
INS$smoker <- as.factor(INS$smoker)
INS$region <- as.factor(INS$region)

# Split the Data
set.seed(100)
TrainingIndex <- createDataPartition(INS$charges, p=0.8, list = FALSE)
TrainingSet <- INS[TrainingIndex, ]
TestingSet <- INS[-TrainingIndex, ]

# Build Training model
Model <- train(charges ~ ., data = TrainingSet,
               method = "lm",
               na.action = na.omit,
               preProcess = c("scale", "center"),
               trControl = trainControl(method = "none")
)

# Apply model for prediction
Model.training <- predict(Model, TrainingSet)
Model.testing <- predict(Model, TestingSet)

# Visualize
plot(TrainingSet$charges, Model.training, col = "blue", main = "Training Set Predictions", xlab = "Actual", ylab = "Predicted")
plot(TestingSet$charges, Model.testing, col = "blue", main = "Testing Set Predictions", xlab = "Actual", ylab = "Predicted")

# Evaluate
postResample(pred = Model.training, obs = TrainingSet$charges)
postResample(pred = Model.testing, obs = TestingSet$charges)

# View feature importance
plot(varImp(Model), main = "Feature Importance")

# View Model summary
summary(Model)
