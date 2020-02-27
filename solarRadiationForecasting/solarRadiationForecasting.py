
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

# In[1]:

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# ## The weather dataset
# This tutorial uses a [weather time series dataset](https://www.bgc-jena.mpg.de/wetter/) recorded by the [Max-Planck-Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/index.php/Main/HomePage).
# 
# This dataset contains 14 different features such as air temperature, atmospheric pressure, and humidity. These were collected every 10 minutes, beginning in 2003. For efficiency, you will use only the data collected between 2009 and 2016. This section of the dataset was prepared by François Chollet for his book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).


# In[3]:
csv_path = '/home/gbecerra/Javeriana/Research/ProyectoAlmacenamiento/ideam/'
csv_name = 'data_tot.csv'
df = pd.read_csv(csv_path + csv_name) 

# Let's take a glance at the data.

# In[4]:
df.head()

# As you can see above, an observation is recorded every 10 mintues. This means that, for a single hour, you will have 6 observations. Similarly, a single day will contain 144 (6x24) observations. 
# Given a specific time, let's say you want to predict the temperature 6 hours in the future. In order to make this prediction, you choose to use 5 days of observations. Thus, you would create a window containing the last 720(5x144) observations to train the model. Many such configurations are possible, making this dataset a good one to experiment with.
# The function below returns the above described windows of time for the model to train on. The parameter `history_size` is the size of the past window of information. The `target_size` is how far in the future does the model need to learn to predict. The `target_size` is the label that needs to be predicted.
# In both the following tutorials, the first 300,000 rows of the data will be the training dataset, and there remaining will be the validation dataset. This amounts to ~2100 days worth of training data.

# In[5]:
TRAIN_SPLIT = 40000

# Setting seed to ensure reproducibility.

# In[6]:
tf.random.set_seed(13)

# ## Part 1: Forecast a univariate time series
# First, you will train a model using only a single feature (temperature), and use it to make predictions for that value in the future.
# 
# Let's first extract only the temperature from the dataset.

# In[7]:
def create_time_steps(length):
	time_steps = []
	for i in range(-length, 0, 1):
		time_steps.append(i)
	return time_steps

# In[8]:
# features_considered = ['p (mbar)', 'T (degC)', 'sh (g/kg)', 'rho (g/m**3)']
features_considered = ['RSG_AUT_60', 'HRA2_2_MEDIA_H', 'PT_2_TT_H', 'VV_2_MEDIA_H', 'DV_2_VECT_MEDIA_H']

# In[9]:
features = df[features_considered]
features.index = df['Fecha']
features.head()

# Let's have a look at how each of these features vary across time.

# In[10]:
features.plot(subplots=True)

# As mentioned, the first step will be to normalize the dataset using the mean and standard deviation of the training data.

# In[11]:
dataset = features.values
data_mean = dataset.mean(axis=0)
data_std = dataset.std(axis=0)

# In[12]:
dataset = (dataset-data_mean)/data_std

# ### Single step model
# In a single step setup, the model learns to predict a single point in the future based on some history provided.
# 
# The below function performs the same windowing task as below, however, here it samples the past observation based on the step size given.

# In[13]:
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i, step)
		data.append(dataset[indices])

		if single_step:
			labels.append(target[i+target_size])
		else:
			labels.append(target[i:i+target_size])

	return np.array(data), np.array(labels)

# In this tutorial, the network is shown data from the last five (5) days, i.e. 720 observations that are sampled every hour. The sampling is done every one hour since a drastic change is not expected within 60 minutes. Thus, 120 observation represent history of the last five days.  For the single step prediction model, the label for a datapoint is the temperature 12 hours into the future. In order to create a label for this, the temperature after 72(12*6) observations is used.

# In[14]:
def plot_train_history(history, title):
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(loss))

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title(title)
	plt.legend()

	plt.show()

# In[15]:
past_history = 240
future_target = 24
STEP = 1

x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

# Let's check out a sample data-point.

# In[16]:
print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

# In[17]:
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

# Plotting a sample data-point.

# In[18]:
def multi_step_plot(history, true_future, prediction):
	plt.figure(figsize=(12, 6))
	num_in = create_time_steps(len(history))
	num_out = len(true_future)

	plt.plot(num_in, np.array(history[:, 0]), label='History')
	plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
			label='True Future')
	if prediction.any():
		plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
			label='Predicted Future')
	plt.legend(loc='upper left')
	plt.show()

# In this plot and subsequent similar plots, the history and the future data are sampled every hour.

# In[19]:

for x, y in train_data_multi.take(3):
	multi_step_plot(x[0], y[0], np.array([0]))

# Since the task here is a bit more complicated than the previous task, the model now consists of two LSTM layers. Finally, since 72 predictions are made, the dense layer outputs 72 predictions.

# In[20]:

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(24))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# Let's see how the model predicts before it trains.

# In[21]:

for x, y in val_data_multi.take(1):
	print (multi_step_model.predict(x).shape)

# In[22]:

EVALUATION_INTERVAL = 300
EPOCHS = 20

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

# In[23]:

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

# #### Predict a multi-step future
# Let's now have a look at how well your network has learnt to predict the future.

# In[24]:

numtest = 50
for x, y in val_data_multi.take(3):
	multi_step_plot(x[numtest], y[numtest], multi_step_model.predict(x)[numtest])

# ## Next steps
# This tutorial was a quick introduction to time series forecasting using an RNN. You may now try to predict the stock market and become a billionaire.
# 
# In addition, you may also write a generator to yield data (instead of the uni/multivariate_data function), which would be more memory efficient. You may also check out this [time series windowing](https://www.tensorflow.org/guide/data#time_series_windowing) guide and use it in this tutorial.
# 
# For further understanding, you may read Chapter 15 of [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/), 2nd Edition and Chapter 6 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).


#%%
