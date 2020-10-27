## Code

This folder will contain jupyter notebooks as well as other python scripts that I used to construct and learn. Some files I will label with 
"trial #" please note that these files may not execute correctly. That is because there might be errors that were integral to the learning process so I documented them. I will post notes about each file here as I upload them. For the data, I was able to get access to the Wharton School of Business database through my university. Other solutions would be using an API such as Alpha Vantage to grab more current data. I plan on using Alpha Vantage to get data to test my model. 


## ML_trial3_withh_all_data

In this notebook, I simplify the model to have 2 layers of 15 neurons. These numbers were chosen at random but the preformance of the model wasn't that bad. 
The model's training loss eventaually went below 1, but the validation loss indicated an overfit model or a vanishing gradient problem. To fix these I will need to tweak the model a bit more. 


## Trial 4

For trial 4 I created several models that utilized a few methods I researched that are supposed to help with overfitting and/or vanishing gradient problems. I made limited progress. With this model I was able to get the model to stop overfitting but the vanishing gradient is still an issue. I can tell this is the problem because when looking at the graph for this model, you can see the training loss (labeled 'Loss') goes down to close to 0 (good). But the validation loss (labeled 'val_loss') decreases and then stops decreasing. This likely means that the model has stopped learning. If the model was overfitting, the val_loss line would start to increase again.  Below is an image of the graph and the code is listed in the folder.

![Trial4 image](/picture/trial4_graph.png)
