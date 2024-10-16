This is my first machine learning project in Python, where I created a predictive algorithm that forecasts the next day's stock prices based on the past week's closing data. Initially focused on next-day predictions, I shifted emphasis to accurately predicting market trends and stock directions before trying to return back to the focus of price forecasting. Throughout this ongoing project, I have honed my programming skills in Python and grown to enjoy learning about machine learning while applying my data analytical skills to real-world datasets. I plan to work on this project more and improve my accuracy metrics as much as possible while learning more ways to make this model more advanced.

For the stock ticker data I used an API called Polygon. My access key only gives me 2 years worth of EOD stock prices. This model does not take into account corporate actions, politics, and current events shocks into account which will hinder accuracy and general prediction abilities. Looking to find a way to incorporate these measures into model. 

General libraries I used are: 

import numpy as np
import requests
import datetime as dt
import pandas as pd

Machine Learning and Stats Libraries:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

<!---
manugsrinivas/manugsrinivas is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
