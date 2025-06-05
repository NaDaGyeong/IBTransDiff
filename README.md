# IBTransDiff

This repository provides an implementation of a diffusion-based time series forecasting framework guided by Transformer-derived global context and regularized via Information Bottleneck principles.

![Image](https://github.com/user-attachments/assets/20f92d21-6d9e-466b-b902-506ba3597d27)

## 1. Setup
Install required packages:
```bash
pip install -r requirements.txt
```

Environment:
* Python 3.8
* Pytorch 2.1.2
* Ubuntu 20.04


## 2. Dataset
We use benchmark datasets including Exchange, ETTm2, ILI, Electricity, and Weather, commonly used in time series forecasting.
We stricitly follow the preprocessing steps and data splits described in prior work.

The raw datasets can be downloaded from:
[Dataset](https://github.com/thuml/Autoformer/tree/main)


## 3. Run Experiment

### Full pipeline
```
python run.py
```
### Only test
```
python run.py --is_training False
```

## 4. Experiment Results
![Image](https://github.com/user-attachments/assets/304897c0-be56-4e4c-93af-e4ad75de4c7a)
![Image](https://github.com/user-attachments/assets/6e487c0f-fee1-4268-8c66-ca1959eed464)

