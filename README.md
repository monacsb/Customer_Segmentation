# Customer_Segmentation
This project adopted deep learning methods.
# Description

"Customer segmentation is a process to dividing customers into groups which possess common characteristics based on their age, gender, profession as well as interests. By doing so will enable the company will gain insights of customer’s buying pattern or preferences, therefore the company will design a marketing strategy to target the most profitable segments. <br />
An automobile company is considering entering new markets using their existing products. Prior entering the market, the analysts from the company conducted intensive market research and they found out that the behaviour of the new market is similar to their existing market." <br />

Therefore, as a analyst, I have focused to develop a deep learning model to predict the right group of datasets on the potential customers.  

# How to use it
Clone repo and run it. <br />
cust_seg.py is a script to train the model and it also gives you accuracy score at the end. 

# Requirement

Spyder <br />
Python 3.8 <br />
Windows 10 or even latest version <br/>
Anaconda prompt(anaconda 3)<br />

# Results

#Since we are looking at customer's buying pattern (spending score) then the possibilities of correlation could be related to gender,age, family size and even profession. Lets look at the heatmap visualization below to confirmed our interpretation. <br />
![heatmap](https://user-images.githubusercontent.com/103228610/169040569-c866c3c6-361f-4497-b5ba-479b526f816d.png) <br />

So looking at the correlation above, it is confirmed that the features variable has significant relation with the target variable (spending score). <br/>

Looking further on our results below is the F1-score.

F1-Score <br />
<img width="325" alt="report_generation" src="https://user-images.githubusercontent.com/103228610/169040084-862174ed-f4c4-4019-a7e4-0c610bbc1831.png">

Training accuracy and validation Accuracy <br />
![training_acc_validation_Acc](https://user-images.githubusercontent.com/103228610/169042049-952b31de-2a36-4c92-9472-5a40d36f627b.png) <br />

The training validation started to reduce slowly after third epoch and the early stopping occurs at the tenth epoch on overall 30 epochs. <br />
![training_loss_validation_loss](https://user-images.githubusercontent.com/103228610/169046772-f811ecdd-aa25-4caa-ae57-23aa3d40a11a.png)


As you can see the training loss has smaller loss then the validation loss, somehow it says a good one. It has at least prevent from underfitting by stopping at epoch 10. <br /> 

Finally, below is the predicted label. Showing mostly likely where most of our data prediction happened. <br /> 

![prediction label](https://user-images.githubusercontent.com/103228610/169045076-1cb49ee7-b225-4a8b-898c-e4b6d9cf5558.png) <br />


# Model
Sequential Model <br />
Optimizer = Adam <br />
Loss = Categorical Crossentropy <br />
Metrics = Accuracy <br />



# Credits
Thanks to Abishek Sudarshan for the dataset <br />
https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation
