
# Analysing and Predicting Extreme Frequency Deviations:
# A Case Study in the Balearic Power Grid

This repository contains the code to reproduce the results of the paper:  
"Analysing and Predicting Extreme Frequency Deviations: A Case Study in the Balearic Power Grid" with the DOI:

This project aims to develop an XGBoost model to predict extreme frequency deviations in the Balearic power grid. The workflow includes loading and preprocessing historical frequency and generation data, detecting positive and negative frequency events, training a model to predict these events, and analyzing the model's decision-making process using SHAP values.

## Data

To run `main.py`, both generation and frequency data are required. The generation data can be found [here](https://demanda.ree.es/visiona/home) and should be named `generation_data.pkl`. The frequency data is not available online yet, but it can be requested.


## License

This code is licensed under the MIT License - see the LICENSE file for details.