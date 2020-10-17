# **Surname Classification with Demographics: PyTorch**

**Introduction and Objective**
- I have used **Multi Layer Percepton Classifier** for the task of Classifying Surnames to their country of origin. Inferring Demographic Information such as Nationality from publicly observable Data has applications from Product Recommendations to ensuring fair outcomes for users accross different Demographics. However demographic and other self identifying attributes are collectively called **Protected Attributes**. I have used PyTorch for the Implementation of MLP Classifier. I hope you will gain some insights.

**PyTorch Dataset Class**
- PyTorch provides an abstraction for the Dataset by providing a Dataset Class. The Dataset Class is an abstract Operator. When using PyTorch with a new Dataset it is necessary to sub class the Dataset Class and Implement the getitem and len methods. I will implement two functions: the getitem method which returns a Data point when given an index and len method returns the length of the Dataset.

**The Vocabulary Class**
- The Vocabulary is the coordination of two Python Dictionaries that form a bijection between tokens or characters here and integers. The first dictionary maps characters to integers indices and the second maps the integers indices to characters. The add_token method is used to add new tokens into the Vocabulary and look_up method is used to retrieve an index and lookup_index is used to retrieve a token given an index.

**The Vectorizer Class**
- The Vocabulary converts individual tokens into Integers and The Surname Vectorizer is responsible for applying the Vocabulary and converting surname into Vector. Surnames are sequence of characters and each character is an individual token in the Vocabulary.

**The Model:Surname Classifier**
- The Surname Classifier is an Implementation of the Multi Layer Perceptron. The first Linear Layer maps the input vectors to an intermediate vector and the non linearity is applied to that vector. A second Linear Layer maps the Intermediate vector to the Prediction vector. In the last step the Softmax Function is optionally applied to make sure the outputs sum to 1 which is interpreted as Probabilities.

**Libraries and Dependencies**
- I have listed all the necessary Libraries and Dependencies required for this Project here:

```javascript
import torch
import pandas as pd
import numpy as np
import collections
import re, json, string, os

from argparse import Namespace
from IPython.display import display
from collections import Counter
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
```

**Getting the Data**
- I have used Google Colab for this Project so the process of downloading and reading the Data might be different in other platforms. I have used **The Surname Dataset** which is a collection of 10000 surnames from 18 different Nationalities collected from different name sources on the Internet. The first property of this Dataset is that it is fairly Imbalanced. The second property is that there is a valid and intuitive relationships between Nationality origin and Surname Orthography.

**Processing the Data**
- I have presented the techniques for processing the raw Dataset for Surname Classification Project using PyTorch here in the Snapshot.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2041b.PNG)

**The Vectorizer Class**
- The Vocabulary converts individual tokens into Integers and The Surname Vectorizer is responsible for applying the Vocabulary and converting surname into Vector. Surnames are sequence of characters and each character is an individual token in the Vocabulary. I have presented the Implementation of Vectorizer Class using PyTorch here in the Snapshot.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2042.PNG)

**The Model:Surname Classifier**
- The Surname Classifier is an Implementation of the Multi Layer Perceptron. The first Linear Layer maps the input vectors to an intermediate vector and the non linearity is applied to that vector. A second Linear Layer maps the Intermediate vector to the Prediction vector. In the last step the Softmax Function is optionally applied to make sure the outputs sum to 1 which is interpreted as Probabilities. I have presented the simple Implementation of MLP Classifier Model using PyTorch here in the Snapshot. 

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2043a.PNG)

**Model Evaluation and Inspection**
- I have presented the simple Implementation of the Inference and Inspection of the Model Evaluation using PyTorch here in the Snapshot.

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2043c.PNG)
