# OpenAttHetRL: An open source toolkit for End-to-end learning of attributed heterogeneous networks

This repository provides a standard training and testing framework for learning the latent reperesntaions of attributed heterogeneous networks in an end-to-end way. 

## Usage

#### Installation

- Clone this repo. or download and unzip the code
- Enter OpenAttHetRL directory, and run the following code
    ```
    pip install .
    or
    pip install OpenAttHetRL
    ```


#### Example

 ```
#e.g.1 data preprocessing
# import the class
from  OpenAttHetRL.preprocessing import  datapreprocessing 
# run preprocessing by passing the location and dataset
datapreprocessing.createdata("./data/","android") 
 ```
