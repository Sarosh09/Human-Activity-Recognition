# Human-Activity-Recognition
## Identification and Prediction of Activities in the Weight Lifting Exercises
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
In this project, the goal was be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.  
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 
* exactly according to the specification (Class A),
* throwing the elbows to the front (Class B),
* lifting the dumbbell only halfway (Class C),
* lowering the dumbbell only halfway (Class D) and
* throwing the hips to the front (Class E).  

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. It was made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).  
My goal here is to predict the *"class"* with the help of other predictors. 

This project is a part of Coursera Practical Machine Learning Week 4 - Peer-graded Assignment: Prediction Assignment Writeup. 

Few Key points about the columns:  

* *"X"* is primary key for the data.
* *"user_name"* is the id of the users. This may help us see interesting patterns for each activity for different users.
* *"classe"* is the target for prediction.
* Column - *3 to 7* is not necessary for this project. (5 features)
* As mentioned above there are 4 different sensors used for data collection. For each sensor there are 38 different features.
* Each sensor("belt","arm","forearm","dumbbell") has raw accelerometer, gyroscope and magnetometer readings for x, y and z axis. (4 sensor * 3 feature * 3 axis = 36 features)
* Each sensor("belt","arm","forearm","dumbbell") has Euler angles (roll, pitch and yaw) feature.(4 sensor * 3 euler angles  = 12 features)
* For the Euler angles of each of the four sensors eight features were calculated: mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness. (4 sensor * 3 feature * 8 measures = 96 features)
* For accelerometer we also have "total" and "variance of total" feature for the 4 sensors. But for "belt", "variance of total" is given as "var_total_accel_belt", for the other sensors it is given as ("var_accel_arm","var_accel_dumbbell","var_accel_forearm"). So I am considering the "belt" one as a typo. (4 sensor * 2 feature = 8 features)
* There is another thing to note here. For "belt" Euler angles feature skewness is given as "skewness_roll_belt", "skewness_roll_belt.1" and "skewness_yaw_belt". I am also considering "skewness_roll_belt.1" as a typo and considering it as "skewness_pitch_belt".  

This machine learning algorithm is applied to predict the 20 test cases given in the test data set.
### Position of the Sensors
![](https://github.com/Sarosh09/Human-Activity-Recognition/blob/master/sensors_pos.png)
## reference  

* [Data Info](http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises)
* [Download Training data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
* [Download Test data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)
* [Qualitative Activity Recognition of Weight Lifting Exercises - Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201)  
