#**Traffic Sign Recognition** 

##A project that is part of the Udacity Self-Driving Car Nano-Degree (SDCND)

### **Abstract** 
####In this project, a Neural Network (NN) is used to classify traffic signs for a self-driving car. The program is written in python and uses TensorFlow (TF) library to construct and train the NN. The [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is used to train, validate and test the performance of the NN.

---

##### **Important Note:** The report is written in parallel with the notebook. The notebook including cells that have been ran in html format is attached as "Traffic\_Sign\_Classifier.html"

---
###**The Organization of this report**

The work throughout this project is divided into steps. Each step has sub-steps as well which will be used to reference code and comments in the jupyter notebbok:

* **Step0:** Loading the data set from local disk
* **Step1:** Exploring and dataset pre-processing
* **Step2:** Designing and training a NN
* **Step3:** Test the NN performance
 

---

## Step 0: Loading the data set from local disk

This part is responsible of loading the datasets into numpy arrays. Three subsets are loaded: training, validation and testing subsets.



---
## Step 1: Exploring and dataset pre-processing

The first step in dataset exploration would be to check the length of each subset and number of labels. (Step 1.1)

The raw data are not suitable for direct feeding to the NN. For example, if we have three "stop signs" in the dataset but they are taken at different lighting conditions like in morning time, afternoon and night. It would make it easier for the NN if all three photos were taken at the same time (similar lighting conditions). It is not a good idea to drop all photos of signs taken at "unclear" lighting conditions as we will lose a valuable part of our dataset. Instead, we can correct the brightness of all the images such that they have similar lighting conditions.
To help us decide about which qualities we should pre-process, it would be a good idea if we can visualize parts of the dataset. Code in Step 1.2 selects a random image from the training set and displays it. The code in Step 1.3 selects an image from each sign unique class and shows it with its corresponding numeric label.
After displaying several images I found that some of them are so dark (the histogram is shifted towards zero). What I thought might be useful is to perform histogram equalization. The histogram equalization code is in step 1.4. The algorithm used performs the equalization on each channel independently. The best would be a histogram equalizer that accounts for the color nature of the image. In addition to histogram equalization, the images are normalized from 0-255 to 0.1-0.9.

Converting the image into grayscale might be an option. In my opinion, that would be a bad idea as it will discard the important color information of the image.

---

## Step 2: Designing and training a NN

LeNet-5 convolutional architecture is used for our problem. As a starting point, the sizes of the filters and other network parameters was based on [Car-ND-LeNet-Lab](https://github.com/udacity/CarND-LeNet-Lab) repository. **(Step 2.1)**

Softmax is used to normalize the probabilities of the outputs (so that they sum to one). The loss is calculated after softmax is compared to the one-hot encoded labels. Adam optimizer is used instead of the simple gradient decent algorithm. Adam optimizer has the advantage of momentum as it uses averaging when calculating the necessary parameters change. All of these form together the training pipeline. **(Step 2.2)**

Another pipeline for evaluating the accuracy of the model was used. This will be used for the validation and testing sets. **(Step 2.3)**

**(Step 2.4):** Now is the turn for the training loop. The trainer will perform the optimization pipeline for each batch in a single epoch. Batch size was chosen to be 128 (I fixed this- will not be tuned). The number of epochs will be changed such that it will be increased as improvement in accuracy is observed. The learning rate is varied from a starting large value to smaller values.

To sum it, the hyper-parameters that we are going to play with in order of importance are:

* The learning rate: *alpha*.
* The variance of the normal distribution: *sigma*.
* The mean of the normal distribution: *mu*.

If non of the above Works, a change in the NN architecture might be necessary.

The cell labelled "Training History" includes many of the NN training trials with the values of hyper-parameters and NN architecture used. Note that the actual chronological order might be different than what is mentioned below as ideas were "firing" in my head as the training goes. I started to use AWS GPU instance mid-way through.

### Qualities of the chosen NN architecture

The NN used consists of 5-layers. The first and second layers are convolutional while all other three are linear. A ReLU activation function is used at the output of each layer. Additionally, the first two layers use average pooling in order to shrink the size of the NN as we move forward. Valid padding is used for the first two layers which means that no zero paddings are added when the convolution is performed. A filter size of 5x5 is used for the first two layers with a stride of one in both directions. This size of the filter is big enough to capture many qualities in the image (edges, corners, shapes). The pooling window of 2x2 is the smallest possible as pooling is intended to reduce NN size at the expense of loosing some information. A larger pooling window size is unnecessary and will degrade the NN performance. For numeric details of the sizing, please refer to "Traffic\_Sign\_Classifier.html"

### Summary of training:

Here are the overall steps followed in training:

* First we fixed *mu* and *sigma* at 0 and 0.1 respectively and tried to vary *alpha*. The best validation accuracy we managed to get is 68%.
* The NN size is expanded. The architecture remained the same it was mostly an expansion in the size of hidden layers. Average pooling was considered instead of max pooling. The reason behind that is that max pooling will discard some important information that might be needed. Average pooling naturally acts as a filter in the image removing noise as well.
* The parameter *alpha* where varied in a logarithmic scale (with some fine-tuning occasionally). The best validation accuracy obtained was 74.6%. *sigma* was changed with no success. *mu* was changed to be slightly negative, again with no success...
* Those, I decided to further expand the NN as it helped in the previous step. A new six layer architecture was used with a higher number of parameters (This six layer architecture is commented out in one of the cells in Step 2). Again it was tried to vary *alpha* and *sigma* but the maximum reached accuracy was around 60%. The new 6-layer NN takes more than double time to train compared to the 5-layer NN. Because no benefit was added with this new architecture, I preferred to move back to the 5-layer NN.
* After enormous trials in changing the parameters, I found the solution. By using small positive value for *mu* the accuracy of the model was raised above 90% and sometimes, reaching an accuracy as high as 96%. These figures were not possible without histogram equalization (see pre-processing Step 1).

### Discussion of the solution

Why did a positive value of *mu* helped? In my opinion, that is because of the use of ReLU as an activation function. If a ReLU tensor is operating at values less than zero, the derivative with respect to that neuron would be zero. This effectively disables that particular neuron and hence, limit the NN ability to learn. Introducing a small positive *mu* would allow more ReLU to operate in their >0 learning regime. This is exactly similar to having positive value for the bias terms.

---

## Step 3: Test the NN performance

### (Step 3.1) Testing on the dataset provided by the [German Traffic Sign Benchmarks](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset):

Well, guess what: the accuracy of the model on the test set was **100%**. I did not iterate the training process to raise this value (i.e. did not fine tune the NN parameter to work well on the test set). The test set was used only once as a *last*  step in the work. (Step 3.1)

### (Step 3.2) Testing on external sign images found in the web

The external test dataset is located in the *Extra-Images* folder in the project folder. A photo editing tool was used to crop sign images and to scale them to 32x32 images. The external test dataset is first loaded into numpy arrays (the alpha channel data are dropped) and then pre-processed. Ten images were used and the accuracy was 70%.

### Discussion of the external dataset results

The top 3 logits output is displayed for each sign image. We will refer to each sign by their file name (they are name in series as sign#.png where # ranges from 1 to 10). For example, we can see that sign3 is misclassified with the true-negative sign having second highest logit value. The true-negative sign is the stop sign and the false positive sign is "Speed limit (70km/h)". Note that both signs are similar in having red and white colors but differ in sign border shape and some other features.
 
It is worth noting that the accuracy of the external test-set (70%) is much
lower than the provided test-set (100%). Clearly, the external test-set data collection and sign extraction methodology is different. It might not be possible to say that the accuracy on the external test-set is 30% less than the provided test set because the external test-set suffer from sparsity (10 samples only).