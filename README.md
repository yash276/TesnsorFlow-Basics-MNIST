# TesnsorFlow-Basics-MNIST
This Repository goes through from Basic-Advance Neural Network implementation on Tensorflow using MNIST Data.<br/><br/>
# Repository<br/>
It contains :<br/>
1.**CODE** ---> It contains all the necessary Configuration and Python File.<br/><br/>
![](images/code.png)<br/><br/><br/>
2. **DATA** ----> It contains the MNIST Dataset.The **ZIP** file contains the complete Dataset for MNIST i.e. 60,000 Training Images and 10000 Test Images<br/>
The **Training** folder has only 10000 images and The **Testing** folder has 100 images only. This due to my system memory constraint while training.<br/> 
You can increase and decrease the data.<br/><br/>
![](images/data.png)<br/><br/>
3. **pythonEnvironment.txt** ----> It contains all the python packages version that was present in tmy system at the time.<br/>
The main package you need to inatall in tensorflow.<br/>
# Installation<br/>
Tensorflow for **CPU** installation.<br/>
```
pip3 install tensorflow
```
Tensorflow for **GPU** installation.<br/>
```
pip3 install tensorflow-gpu
```
# Run<br/>
## Config File<br/>
Go inside the code directory you will find a **config.yaml** file. You can change the **BatchSize & Network** parameter over there before running the code.<br/>
The code has both Multi-Layer Perceptron Model and Convolution Neural Network model for you to run and experiment and play with it.The **Network** parameter dictates which model to run.**0** for MLP and **1** for CNN.<br/> 
To the run the code GO inside code directory and run the **Main** file<br/>
```
python3 main.py
```
# Output<br/>
1. Some Sample Input Images like displayed below will be saved under the name **inputImages.png**. Your displyed images and my displayed images can vary as they are selected randomly.<br/><br/>
![](images/inputImages.png)<br/><br/>
2. Cost Funtion Graph. According to your **Network** in Config a Cost function Image will be saved. **costMLP.png** for MLP model and **costCNN.png** for CNN model. It should look something like this. 
![](images/costMLP.png)<br/><br/>
# Credits<br/>
The code in this repository is inspired by my learning that I gained by the completion of the following Course. Make sure to check it out. Big thanks to Andrew Ng for such an amazing course.
[https://www.coursera.org/specializations/deep-learning]
