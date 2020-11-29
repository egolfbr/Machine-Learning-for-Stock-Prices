## Code

This folder will contain jupyter notebooks as well as other python scripts that I used to construct and learn. Some files I will label with 
"trial #" please note that these files may not execute correctly. That is because there might be errors that were integral to the learning process so I documented them. I will post notes about each file here as I upload them. For the data, I was able to get access to the Wharton School of Business database through my university. Other solutions would be using an API such as Alpha Vantage to grab more current data. I plan on using Alpha Vantage to get data to test my model. 


## ML_trial3_withh_all_data

In this notebook, I simplify the model to have 2 layers of 15 neurons. These numbers were chosen at random but the preformance of the model wasn't that bad. 
The model's training loss eventaually went below 1, but the validation loss indicated an overfit model or a vanishing gradient problem. To fix these I will need to tweak the model a bit more. 


## Trial 4

For trial 4 I created several models that utilized a few methods I researched that are supposed to help with overfitting and/or vanishing gradient problems. I made limited progress. With this model I was able to get the model to stop overfitting but the vanishing gradient is still an issue. I can tell this is the problem because when looking at the graph for this model, you can see the training loss (labeled 'Loss') goes down to close to 0 (good). But the validation loss (labeled 'val_loss') decreases and then stops decreasing. This likely means that the model has stopped learning. If the model was overfitting, the val_loss line would start to increase again.  Below is an image of the graph and the code is listed in the folder.

![Trial4 image]( /picture/trial4_graph.PNG)


### Tweak parameters

For these next iterations of my model, I choose to simply tweak the parameters of the model I had created. These parameters are as follows: Number of layers, number of neurons per layer, activation function, using bias neurons, using kernal initializers, and optimization functions. Not only did I spend time tweaking these parameters but I sought help through the machine learning community on [stackoverflow](http://stackoverflow.com). There were a limited number of replies on my post regarding optimization of a model but I was able to take what was said and change a few things in order to get a model that seemed to work differently. Not necessarilly better, but something more understandable for a beginner. I changed the activation function to the popular ReLu function, and used it for all layers (except the output). Then used the 'adam' optimizer for backpropagation and I also changed the model dimensions. That is I changed it to be two layers with two neurons each and then an output layer with one neuron. I did this because I changed the input data set to only include price and volume (two features). This is because before I was dealing with 26 features. I thought that if I could get a model to work with data that was more simple, then I could try and work with more complex data afterwards. 
The results were mixed and I tried about a dozen different models but I still have two major problems. The first being overfitting. This is common and there are many reasons that a model would be overfit. Such as a model that is too complex, data that is not correctly normalized, or the absence of regularizers on the weights. The next problem is that my model doesn't reach the desired range for loss and validation loss. However, the validation loss doesn't increase which means that it is still technically learning but at a much slower rate. This is called the vanishing gradient problem. As models get larger and more complex, the gradients on the lower levels get smaller and smaller eventually ending up around zero. This problem I am still quite new to and will be the subject of further research as the project goes on.

### Polynomial regression

I learn by getting my hands dirty and by trying examples and experimenting with the libraries and code available. This is why I did minimum research before jumping in and trying to build a model. It was probably a little naive of me to think that I would be able to produce a model with even remotely good results and that of course didn't happen. Thus, I went back to reading and research and found another avenue to experiment with: Polynomial regression. So far my trials have consisted of linear regression and sequential neural networks. However, my data is not linear in nature. There are many variables and relationships that affect the price of a stock. This seems to indicate that polynomial regression (PR) would be the perfect tool.

### Recurrent Neural Network (RNN)

After much more reading on the explicit problem of predicting stock prices, I learned that lots of people have had succues (to a degree) using RNNs. These models use the output of a layer as an additional input to the previous layer, adding a feedback loop in the model design. All the previous models I have tried are considered feedforward because they do not loop back. Adding this loop back can allow of the model to learn better from historical data and have "memory." This is extremly important when talking about stock data because for technical analysis predicitons, this is basically the definition: using historical technical data to make a prediction about the next price. I found a good tutorial about RNNs and they just so happen to be solving the same problem. The file that is labeled ["ece_448_final_fall_algorithm"](/code/ece_448_final_fall_algorithm.ipynb) is our recurrent neural network model. This is just the one that [Ahmed](towardsdatascience.com/getting-rich-quick-with-machine-learning-and-stock-market-predictions-696802da94fe) used in his model, but we decided to see if it was accurate in this scenario. Turns out this model is pretty accurate at predicting trends but on an indivdual level there is still a decent amount of error. Even if the average is 10%, sometimes when predicting a price like Amazon around $3000 you can get predictions of about $1000 or less. 

## sources
Ahmed, Yacoub. “Getting Rich Quick with Machine Learning and Stock Market Predictions.” Medium, Towards Data Science, 12 Jan. 2020, towardsdatascience.com/getting-rich-quick-with-machine-learning-and-stock-market-predictions-696802da94fe. 

