# OpenAttHetRL: An open source toolkit for End-to-end learning of attributed heterogeneous networks

This repository provides a standard training and testing framework for learning the latent representations of attributed heterogeneous networks in an end-to-end way.

## Usage

#### Installation

- Clone this repo. or download and unzip the code
- Enter OpenAttHetRL directory, and run the following code
    ```
    pip install .
    or
    pip install OpenAttHetRL
    ```


#### Examples
##### e.g.1 data preprocessing
 ```python
# import the class
from  OpenAttHetRL.preprocessing import  datapreprocessing 
# run preprocessing by passing the location and dataset
datapreprocessing.createdata("./data/","android") 
 ```
 ##### e.g. 2 generate train and test data for expert finding task
```python 
 # AttHetRL class to run the framework
from  OpenAttHetRL.framework import  AttHetRL
#generate train and test data for expert finding 
AttHetRL.prepare_train_test("data/android")
```
 ##### e.g.2 model training
 ```python
# AttHetRL class to run the framework
from  OpenAttHetRL.framework import  AttHetRL
#perform  function run_train to train the model 
#   parameters: 
#  "data/android": dataset name and location,
#  dim_node=32,dim_word=300  node and word embedding dimentions ,
#  epochs=10,batch_size=16   the number of epochs and batch size
#  returns: trainded model is stored in ./data/android/parsed/model/
model=AttHetRL.run_train("data/android",dim_node=32,dim_word=32
                         ,epochs=1,batch_size=16)
# save node and word embedding vectors
model.saveembedings()
#save the model
AttHetRL.saveModel("data/android/parsed/model/","model",model)
#load the model
loadedModle=AttHetRL.loadModel("data/android/parsed/model/","model") 
 ```
 
 ##### e.g.3 perform expert finding 
 ```python
# AttHetRL class to run the framework
from  OpenAttHetRL.framework import  AttHetRL
#load the model
loadedModle=AttHetRL.loadModel("data/android/parsed/model/","model")
#find topk related experts for a given question
q={"title":"How Can I fix GPS in my Samsung Galaxy S?","tags":["gps","samsung-galaxy-s"],"askerID":1089}
top10=AttHetRL.findTopKexperts(question=q,model=loadedModle,topk=10)
print(top10)
print("done!")
 ```
