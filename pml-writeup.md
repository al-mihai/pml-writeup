# Prediction Assignment Writeup - Quantifying the Weight Lifting Exercise Activity


## Synopsis

Human Activity Recognition - HAR has emerged as a key research activity in the recent years, but the emphasis was more on "how much" than "how well" a certain activity was done. 

Using data from the Weight Lifting Exercise Dataset of the HAR project [1] we apply machine learning techniques to predict with a high accuracy the correctness of the exercise execution.

## Data Processing

The data is read in data.frame structures as it can be seen below:


```r
pmlte<-read.table("pml-testing.csv", dec=".", fill=TRUE, quote="\"", 
                 row.names=NULL,
                 header=TRUE, sep=",")
pmltr<-read.table("pml-training.csv", dec=".", fill=TRUE, quote="\"", 
                  row.names=NULL,
                  header=TRUE, sep=",")
```

An exploratory approach of the data is taken, in order to gain more insight into the judicious choice of the features to be considered.

If we calculate the variance of all the features (the columns), we can see that many of them are "NA" or close to 0. We choose to ignore them and build a new dataset with a limited number of features - those that 'matter'/can potentially influence the prediction outcome.


```r
apply(pmltr,2,var)
```

```
##                        X                user_name     raw_timestamp_part_1 
##                3.209e+07                       NA                4.200e+10 
##     raw_timestamp_part_2           cvtd_timestamp               new_window 
##                8.307e+10                       NA                       NA 
##               num_window                roll_belt               pitch_belt 
##                6.146e+04                3.938e+03                4.996e+02 
##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
##                9.062e+03                5.994e+01                       NA 
##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
##                       NA                       NA                       NA 
##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
##                       NA                       NA                       NA 
##           max_picth_belt             max_yaw_belt            min_roll_belt 
##                       NA                       NA                       NA 
##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
##                       NA                       NA                       NA 
##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
##                       NA                       NA                       NA 
##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
##                       NA                       NA                       NA 
##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
##                       NA                       NA                       NA 
##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
##                       NA                       NA                       NA 
##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
##                4.299e-02                6.121e-03                5.824e-02 
##             accel_belt_x             accel_belt_y             accel_belt_z 
##                8.788e+02                8.167e+02                1.009e+04 
##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
##                4.119e+03                1.273e+03                4.252e+03 
##                 roll_arm                pitch_arm                  yaw_arm 
##                5.292e+03                9.414e+02                5.093e+03 
##          total_accel_arm            var_accel_arm             avg_roll_arm 
##                1.107e+02                       NA                       NA 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                       NA                       NA                       NA 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                       NA                       NA                       NA 
##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
##                       NA                       NA                3.974e+00 
##              gyros_arm_y              gyros_arm_z              accel_arm_x 
##                7.249e-01                3.060e-01                3.314e+04 
##              accel_arm_y              accel_arm_z             magnet_arm_x 
##                1.207e+04                1.813e+04                1.968e+05 
##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
##                4.077e+04                1.067e+05                       NA 
##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
##                       NA                       NA                       NA 
##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
##                       NA                       NA                       NA 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                       NA                       NA                       NA 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                       NA                       NA                       NA 
##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
##                       NA                       NA                4.890e+03 
##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
##                1.369e+03                6.809e+03                       NA 
##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
##                       NA                       NA                       NA 
##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
##                       NA                       NA                       NA 
##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
##                       NA                       NA                       NA 
##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
##                       NA                       NA                       NA 
## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
##                       NA                       NA                1.047e+02 
##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
##                       NA                       NA                       NA 
##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
##                       NA                       NA                       NA 
##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
##                       NA                       NA                       NA 
##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
##                       NA                2.276e+00                3.721e-01 
##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
##                5.229e+00                4.532e+03                6.521e+03 
##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
##                1.198e+04                1.154e+05                1.068e+05 
##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
##                1.959e+04                1.167e+04                7.922e+02 
##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
##                1.065e+04                       NA                       NA 
##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
##                       NA                       NA                       NA 
##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
##                       NA                       NA                       NA 
##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
##                       NA                       NA                       NA 
##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                       NA                       NA                       NA 
##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
##                       NA                1.011e+02                       NA 
##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
##                       NA                       NA                       NA 
##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
##                       NA                       NA                       NA 
##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
##                       NA                       NA                       NA 
##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
##                4.207e-01                9.614e+00                3.078e+00 
##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
##                3.261e+04                4.005e+04                1.915e+04 
##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
##                1.204e+05                2.595e+05                1.364e+05 
##                   classe 
##                       NA
```

```r
data<-subset(pmltr, select=c("roll_belt", "pitch_belt","yaw_belt","total_accel_belt",
"accel_belt_x","accel_belt_y","accel_belt_z", "magnet_belt_x","magnet_belt_y",
"magnet_belt_z","roll_arm","pitch_arm","yaw_arm","total_accel_arm","gyros_arm_x",
"accel_arm_x","accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z",
"roll_dumbbell","pitch_dumbbell","yaw_dumbbell","total_accel_dumbbell","gyros_dumbbell_x",
"gyros_dumbbell_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z",
"magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","roll_forearm","pitch_forearm",
"yaw_forearm","total_accel_forearm","gyros_forearm_y","gyros_forearm_z","accel_forearm_x",
"accel_forearm_y","accel_forearm_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z",
"classe"))
```

The total computation time for the training part will be decreased by this limitation of the number of the features.

We create 'training' and 'testing' datasets with this data.


```r
library(caret)
intrain<-createDataPartition(y=data$classe, p=0.75,list=FALSE)
training<-data[intrain,]
testing<-data[-intrain,]
```

We choose a "random forest" machine learning algorithm because we have a classification problem with many of parameters and this choice could have a good accuracy from the first try.

The 'testing' dataset will be used for cross-validation, after building the model.

For prefromance reasons, the 'randomForest' library was used, as it can be seen below.


```r
library("randomForest")
library("doParallel")
registerDoParallel(cores=detectCores())
modrf<-randomForest(classe ~ ., data = training)
prf<-predict(modrf,newdata=testing)
```

## Results
We obtain an overall accuracy of 99.5%. The "random forest" approach gives a very good result for this problem.


```r
confusionMatrix(prf, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    5    0    0    0
##          B    1  942    2    0    1
##          C    0    2  852    5    3
##          D    0    0    1  797    3
##          E    0    0    0    2  894
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.992, 0.997)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.994         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.993    0.996    0.991    0.992
## Specificity             0.999    0.999    0.998    0.999    1.000
## Pos Pred Value          0.996    0.996    0.988    0.995    0.998
## Neg Pred Value          1.000    0.998    0.999    0.998    0.998
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.174    0.163    0.182
## Detection Prevalence    0.285    0.193    0.176    0.163    0.183
## Balanced Accuracy       0.999    0.996    0.997    0.995    0.996
```

**[1]**Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. *Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements*. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.
