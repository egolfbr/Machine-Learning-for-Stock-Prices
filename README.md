# Machine-Learning-for-Stock-Prices
For my senior design project at Miami University my team and I set out to try and develop a machine learning algorithm to advise and make stock investments. There were many parameters to consider such as investment amount, length, amount of risk, as well as how we measure that risk and determine probability of profit.
## Table of Contents
**[Introduction](#Introduction)**<br>
**[Machine Learning Overview](#Machine-learning)**<br>
**[Our Goal](#Our-Goal)**<br>
**[Tools](#Tools)**<br>
**[Programming](#Programming)**<br>

## Introduction

Ever since the stock market was founded, people have been trying to find a way to "beat" it. They do this by trying to correctly predict the future value to make a profitable investment. The stock market is essentially just an auction. Someone has a part of a company that they are willing to sell for a certain amount of money (ask price) and people want to buy that share for another price (ask price) and eventually the buyer will buy at a price and that will become the new price for the stock. That is a very simple overview but with millions of transactions and millions of dollars going in and out of companies daily, it becomes hard to predict what the price will do. There is also a plethora of data ranging from stock charts to technical data such as moving averages or momentum oscillators. This data makes this problem a good starting point for a machine learning algorithm. 

## Machine learning Overview

Machine learning algorithms learn through experiementation. This process is called training. A normal algorithm follows a certain set of rules while a machine learning algorithm learns the rules itself. Machine learning has two main categories: Classification and regression. Classification is used for sorting data into groups or bins. An example of this is spam detection. This algorithm is trained on data and each data points is labeled "spam" or "not spam." Then the algorithm learns the rules to differentiate between the two categories so that when it is fed an unlabeled piece of mail it can correctly predict wihch category it belongs to. Regression is good for predicting a future characteristic or attribute of data. There are a few main steps for building a machine learning algorithm. First we obtain the data, then we train the model on a subset of the data, then we feed the model test data to validate the model performance and then finally we input data that the model has not seen before and see how well it performs. Based on this metric we can then optimize the model. 


![Image of Machine learning Steps](picture/steps_pic.jpg)

Now I will go into detail about each step of this process. Gathering and preparing the data is the most time consuming part of this process. We need to find a way to access historical data and compile our own database of data with which we can train the model. There are many places that we can get this information. Kaggle is a website that has open source datasets that people can use. There are also APIs that you can use to call on historical stock data. One example is Alpha Vantage. In this API you can call a function that returns different types of data. After obtaining the data we need to prepare it. In this step we will feed the data through what is known as a "pipeline." This is when the data undergoes transformations so that it is understandable by the model. This is important expecially with stock data because certain values are going to be much larger than other values (such as volume and price) and the model could favor the larger numbers. To avoid this the data is normalized. This can be done with a custom function or with built in functions in TensorFlow API. 

![Image of Machine learning models](picture/models.webp)

## Our Goal

Our goal is to input historical stock data for a given stock ticker, and try and use this algorithm to tell us when to buy and sell the stock in order to reap the most rewards. As a benchmark we will use the S&P500 because that is the standard that most mutual funds and investors use when creating a new fund. Along with the input of the ticker, a user will enter the amount they are willing to invest and the risk level that they are comfortable with. There are many ways in which we can construct this algorithm one of which is using all the investment money atonce to buy as much of the stock as we can and then sell all of it when we receive a sell signal from the machine learning model. However, we could have to algorithm invest only portions of the investment amount based on certainty of profit and then sell certain amounts based on certainty of loss. All these will have to be considered when constructing the model. For now we will focus on just predicting the price and we will use another algorithm on the output to calculate the risk level and chance of profit. 

## Tools

For this project we decided to use the python programming language. Python is one of the fastestgrowing programming languages in popularity because of its simplicity and use of white space (indentation). It also has many powerful libraries available to download. One such library that we will use to create, train and evaluate the model is the scikit-learn library. This library has many tools and allows the user to input data and has built in functions for spliting the data up into training and testing. Another pwoerful tool we will be using is the Pandas library. Pandas is a powerful data library that allows us to read, write and organize excel files, csv files and many other data types. It comes preloaded with many unique tools that allow us to see null values, data types in each feature and many more. 
For our IDE I chose IDLE and Jupyter Notebook. Both of these IDEs have their pros and cons, but I especially like Jupyter because I can run cell by cell to see outputs and gain a deeper understanding of my code.
![Python](picture/python.jpg)
![Jupyter](picture/jupyter.png)


## Programming 

There are many different types of models and methods to build a successful machine learning algorithm. As mentioned above classification and regression are the two methods that machine learning performs really well. So for the first few trials of programming, I attempted to make a regression algorithm. This is because the goal is to predict future stock prices. I had limited success with this. I created about four different models and only two gave me results that were remotley acceptable. The first models results are below. As you can see, by the end of the short training period I was able to acheive an accpectable loss but not an accecpt able value loss. For this model I used an input layer of 27 neurons, a hidden layer of 15 neurons and an output layer of 1 neuron. For both models I used an activation function of sigmoid on all layers and an optimizer function of sgd on the weights. 

![Image of Machine learning results](picture/results_sigmoid.PNG)


The next trial I conducted was changing the input data to only price and volume. For the above model I trained it using a plethora of technical data compiled in the excel sheet. However, since the technical indicators are already made from price and volume data, presenting a model with the technical data could lead to overtraining since it will have seen the data again just transformed. Those results were not much better.

![Image of Machine learning results](picture/price_vol_results_1.PNG)

As you can see, the loss is well under 1 (very acceptable) but loss and accuracy are nowhere on the charts. The loss isn't there because the value is too high (in the 1000s) but accuracy isn't there for an unknown reason. Possibly an error in the code. 

After these unpleasant results I did a little more research in predicting values (stock prices) with machine learning. After that I decided to tweak my original model. Instead of having a different amounts of neurons per layer, I made all hidden layers have 15 neurons (choosen at random). I kept the same activation and optimization functions as the previous tests and wound up with much more pleasant result. 


![Image of Machine learning results](picture/results_trial3.PNG)

These results started with a high loss but eventually made it to under 1 which was good. The value loss is still really high and accuracy is still not on the chart but I will tackle one issue at a time. 

The next three trials will be the following: 
1. Tweak parameters of current model 
2. Create a model using tensorflow.estimator instead of keras
3. Create a new dataset to run a classification model instead

You might be wondering why I would try and run a classification model on something designed to predict the output. Well, instead of thinking about predicting the price we can try and classify a datapoint based on future data. To do this I will add a few columns to the price and volume data set. Each of these columns will contain a 1 or a 0. It will contain a 1 if the price increase X% from a certain time period before and a 0 if it did not increase by that percentage. This way we can ask the user for their investment length and use that column of data to optimize the model. Then we feed the model the price and volume data and have it try and predict if it will be a 1 or a 0, thus a classification model. 

### RNN

Before trying the above mentioned trials, I did a bit more research on specific stock predicting algorithms. What I found was that people found that using recurrent neural networks, worked much better than a feedforward model. A recurrent neural network (RNN) is similar to a feedforward neural network except that it has a feedback loop. This makes the next prediction based on input data as well as the previous time step prediction. Since stock prices are based off of historical data, this looked like a promising solution. 

### Future Work

As mentioned towards the end of the previous section there is a lot of work to be done in order to make this model valid for trading on the market. First what we need to do is generalize the model. As we mentioned earlier, we trained the model on a single stock. Next we need to experiment with generalizing this model to try and make it accurate with multiple stocks or any stock. In order to accomplish this first we will run different stock data through this trained model to see what the output is to get a base accuracy. Then we will construct our own dataset of multiple companies and their stock data. This will be constructed the following way:

[](picture/future_work.PNG]

As you can see, the data will be a 3D array with the height being the number of days we have access to (this will be limited by the stock with the least amount of stock data), the length will be equal to the amount of companies we will sample and the width will be the features we will look at. There are two things we need to keep in mind when constructing this data set. The first is we need to make sure we have a good sample of companies from the stock market. We can’t use just large-cap companies because then the model might become overfit and/or not predict low-cap companies well. We also need to have a good sample of each sector. If we only use tech stocks the model might become overfit and/or it might not predict consumer discretionary stocks well. The second thing we need to keep in mind is dimensionality. As with the polynomial regression model, this model may run the risk of becoming too computationally heavy for a standard computer that we have access to. That is because we will have a very large data set with many different features to try and determine their relationships. Thus we should try and keep the number of features small and pick a good sample size of companies so that we don’t have to have 100 different companies. After we obtain the above data set we will manipulate it so that we can feed it into a neural network and begin our testing. 
