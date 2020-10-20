# Machine-Learning-for-Stock-Prices
For my senior design project at Miami University my team and I set out to try and develop a machine learning algorithm to advise and make stock investments. There were many parameters to consider such as investment amount, length and amount of risk. As well as how we measure that risk and determine probability of profit.

# Introduction

Ever since the stock market was founded, people have been trying to find a way to "beat" it. That is, correctly predict the future value to make a profitable investment. The stock market is essentially just an auction. Someone has a part of a company that they are willing to sell for a certain amount of money (ask price) and people want to buy that share for another amounr (ask price) and eventually the buyer wil buy at a price and that will become the new price for the stock. That is a very simple overview but with millions of transactions and millions of dollars going in and out of companies daily, it becomes hard to predict what the price will do. There is also a plethora of data ranging from stock charts to technical data such as moving averages or momentum oscillators. This data makes this problem a good starting point for a machine learning algorithm. 

# Machine learning 

Machine learning algorithms learn through experiementation. This process is called training. A normal algorithm follows a certain set of rules. If this then do this, else do something else. A machine learning algorithm improve these steps by themselves so that humans do not have to constantly update the algorithm. Machine learning has two main categories: Classification and regression. Classification is used for sorting data into groups or bins. An example of this is spam detection. This algorithm is trained on labeled data (supervised learning) and when it sees a new email it can classify it as spam or not spam. Regression is good for predicting a future characteristic or attribute of data. This makes this method a great tool for predicting stock prices. As mentioned above supervised learning is when we train the machine learning algorithm with labeled data and then feed it data that is unlabeled. With the existence of supervised learning hints to the existence of unsupervised learning. Unsupervised learning is where we feed the algorithm data that is unlabeled in the training stage. This would be a good solution for our data because we don't have our data labeled however, regression relies on supervised learning. Thus, we will need to obtain some sort of labeled data or do some preprocessing ourselves to prep our data for the training stage. 

![Image of Machine learning Steps](picture/steps_pic.jpg)

Above are the steps we will follow to create our machine learning model and implement it. Gathering and prepping the data will take the most time as we will be gathering raw data from the New York Stock Exchange (NYSE). We will get the price and volume and from that we will create the technical indicators that we want to use to make our predicitions. After we create those indicators we will need to normalize the data. This would take an extreamly long time and we would have to create multiple new files for each company but with Scikit-learn python library, we can easily complete this step with just a few lines of code. The last prepping step to do is to label the data. In this step we will have to determine our length of investment and then calculate if each price point is a profit or loss. After we are done prepping we get to choose the model that best fits our needs. As mentioned before we will start with a regression model and move on from there. One interesting avenue to go down would be to use a clustering model. Clustering models work really well with unsupervised data which is what we will have. There has been some research done in regards to unsupervised regression models, however the goal of this project is not to come up with a new model but to use an existing one to try and better predict stock prices.

![Image of Machine learning models](picture/models.webp)

# Our goal

Our goal is to input historical stock data for a given stock ticker, and try and use this algorithm to tell us when to buy and sell the stock in order to reap the most rewards. As a benchmark we will use the S&P500 because that is the standard that most mutual funds and investors use when creating a new fund. Along with the input of the ticker, a user will enter the amount they are willing to invest and the risk level that they are comfortable with. 

# Output

The output of the algorithm is what we need to focus on in order to build a proper model. There are many ways in which we can construct this algorithm one of which is using all the investment money atonce to buy as much of the stock as we can and then sell all of it when we receive a sell signal from the machine learning model. However, we could have to algorithm invest only portions of the investment amount based on certainty of profit and then sell certain amounts based on certainty of loss. All these will have to be considered when constructing the model. For now we will focus on just predicting the price and we will use another algorithm on the output to calculate the risk level and chance of profit. 

# Tools

For this project we decided to use the python programming language. Python is one of the fastestgrowing programming languages in popularity because of its simplicity and use of white space (indentation). It also has many powerful libraries available to download. One such library that we will use to create, train and evaluate the model is the scikit-learn library. This library has many tools and allows the user to input data and has built in functions for spliting the data up into training and testing. Another pwoerful tool we will be using is the Pandas library. Pandas is a powerful data library that allows us to read, write and organize excel files, csv files and many other data types. It comes preloaded with many unique tools that allow us to see null values, data types in each feature and many more. 
For our IDE I chose IDLE and Jupyter Notebook. Both of these IDEs have their pros and cons, but I especially like Jupyter because I can run cell by cell to see outputs and gain a deeper understanding of my code.
