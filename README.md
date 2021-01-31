# Humor Detector

Multiple humor detection models with feature extraction.

## Requirements

Run the following command to install the required packages:

```sh
pip3 install -r requirements.txt
```

## Build

This project uses Cython, so in order to use the Humor Features module you'll need to build it using this command:

```sh
python3 setup.py build_ext --inplace
```

## Quick start

You can quickly start working with the Humor detector by importing HumorDetector object as following:

```python
import pandas as pd
from HumorDetector import HumorDetector

df = pd.read_csv('data/dataset_final.csv', sep=',', encoding='ISO-8859-1', header=0)
HumorDetector(dataset=df).performance_overview()
```

This code will display the pre-built models performances.
Humor detector will proceed directly to training the models using the pre-built Machine learning methods.

If you don't specify the dataset, Humor Detector will use the pre-trained models. You will need to specify the choosen Machine learning method.

```python
from HumorDetector import HumorDetector

result = HumorDetector().predict('some humoristic short text.', 'Logistic Regression')
print(result)
```