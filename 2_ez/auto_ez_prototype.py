import json
import numpy as np
import os
import datetime
import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
import astropy.io.fits as pyfits
import pickle
import sys
from itertools import chain
from multiprocessing import Pool
from tensorflow.keras.regularizers import l2
import multiprocessing
import time
start_time = time.time()
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Dense, Conv1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant, Zeros
import psutil
#from memory_profiler import profile
import gc
import shutil


#set multithreading limits and gpu device
tf.config.threading.set_inter_op_parallelism_threads(6)  # For coordinating independent operations
tf.config.threading.set_intra_op_parallelism_threads(6)  # For speeding up individual operations
gpus = tf.config.experimental.list_physical_devices('GPU')

strategy = tf.distribute.MirroredStrategy()



logdir = "tf_logs" 


# Function to adjust column names by appending a counter to duplicates
def adjust_column_names(names):
    counts = {}
    new_names = []
    for name in names:
        if name in counts:
            counts[name] += 1
            new_name = f"{name}.{counts[name]}"
        else:
            counts[name] = 0
            new_name = name
        new_names.append(new_name)
    return new_names




class data_producer():
    def __init__(self, data_points, min_amplitude, max_amplitude, min_x_value, max_x_value):
        self.data_points = data_points
        self.means = np.log10([
                        1033.82, 1215.24, 1240.81, 1305.53, 1335.31,
                        1397.61, 1399.8, 1549.48, 1640.4, 1665.85,
                        1857.4, 1908.734, 2326.0, 2439.5, 2799.117,
                        3346.79, 3426.85, 3727.092, 3729.875, 3889.0,
                        4072.3, 4102.89, 4341.68, 4364.436, 4862.68,
                        4932.603, 4960.295, 5008.240, 6302.046, 6365.536,
                        6529.03, 6549.86, 6564.61, 6585.27, 6718.29,
                        6732.67, 3934.777, 3969.588, 4305.61, 5176.7,
                        5895.6, 8500.36, 8544.44, 8664.52, 5578.5,
                        5894.6, 6301.7, 7246.0
                        ])#18, 19, 25, 24, 33, 34, 32
        
        self.columns = ["O VI", "Lyα", "N V", "O I", "C II", "Si IV", "Si IV + O IV", "C IV", "He II", "[O III]", "Al III", "C III", 
                        "C II", "Ne IV", "Mg II", "[Ne V]", "Ne VI", "[O II]", "[O II]", "He I", "[S II]", "Hδ", "Hγ", "[O III]", "Hβ", 
                        "[O III]", "[O III]", "[O III]", "O I", "O I", "N I", "[N II]", "Hα", "[N II]", "[S II]", "[S II]", "K", "H", "G", 
                        "Mg", "Na", "CaII", "CaII", "CaII", "Sky", "Sky", "Sky", "Sky", "Z"] #HB is idx 24  Ha is 32    
        
        self.gaussians_batch = []
        self.noisy_gaussians_batch = []
        self.lambdas = []

        self.min_val = min_amplitude
        self.max_val = max_amplitude

        self.min_x_value = min_x_value
        self.max_x_value = max_x_value
        
        
    def adjust_column_names(self, names):
        counts = {}
        new_names = []
        for name in names:
            if name in counts:
                counts[name] += 1
                new_name = f"{name}.{counts[name]}"
            else:
                counts[name] = 0
                new_name = name
            new_names.append(new_name)
        return new_names

    def initialize_data(self, wavelength_template, vary_height = False, full_line_range = False, true_ratios = True):
        
        means = tf.constant(self.means, dtype = tf.float32)

        pre_compiled_data = np.zeros((self.data_points, len(self.means)+1))

        if vary_height == False and full_line_range == False:
            for i in range(len(pre_compiled_data)):
                pre_compiled_data[i][18] = 1
                pre_compiled_data[i][19] = 1
                pre_compiled_data[i][24] = 1
                pre_compiled_data[i][25] = 1
                pre_compiled_data[i][32] = 1
                pre_compiled_data[i][33] = 1
                pre_compiled_data[i][34] = 1
                pre_compiled_data[i][-1] = i/pre_compiled_data.shape[0]#18, 19, 25, 24, 33, 34, 32
        
        elif vary_height == True and full_line_range == False:
            for i in range(len(pre_compiled_data)):
                pre_compiled_data[i][18] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][19] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][24] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][25] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][32] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][33] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][34] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())
                pre_compiled_data[i][-1] = i/pre_compiled_data.shape[0]#18, 25, 27, 33

        elif vary_height == False and full_line_range == True:
            for i in range(len(pre_compiled_data)):
                used_indices = set()
                while len(used_indices) < 4:
                    random_index = np.random.randint(0, len(self.means))#18, 34)
                    if random_index not in used_indices:
                        pre_compiled_data[i][random_index] = 1
                        used_indices.add(random_index)  # Mark this index as used for this iteration
                
                pre_compiled_data[i][-1] = i / pre_compiled_data.shape[0]

        elif vary_height == True and full_line_range == True:
            for i in range(len(pre_compiled_data)):
                used_indices = set()
                while len(used_indices) < 4:
                    random_index = np.random.randint(0, len(self.means))
                    if random_index not in used_indices:
                        pre_compiled_data[i][random_index] = (self.min_val + (self.max_val - self.min_val) * np.random.rand())  # Assign 1 to the unique random index
                        used_indices.add(random_index)  # Mark this index as used for this iteration
                
                pre_compiled_data[i][-1] = i / pre_compiled_data.shape[0]
        
        if true_ratios == True:
            
            #initialize ratios
            Ha_Hb = 2.86

            Nii_Ha_max = 10
            Nii_Ha_min = 10**-4

            Oiii_Hb_max = 10
            Oiii_Hb_min = 10**-2.5

            Sii_Ha_max = 10
            Sii_Ha_min = 10**-2.5

            Nii_Oii_max = np.sqrt(10)
            Nii_Oii_min = 10**-1

            Oii_Oiii_max = 100
            Oii_Oiii_min = 10**-1

            

            for i in range(len(pre_compiled_data)):
                
                Nii_Ha = np.random.uniform(Nii_Ha_min, Nii_Ha_max)
                Oiii_Hb = np.random.uniform(Oiii_Hb_min, Oiii_Hb_max)
                Sii_Ha = np.random.uniform(Sii_Ha_min, Sii_Ha_max)
                Nii_Oii = np.random.uniform(Nii_Oii_min, Nii_Oii_max)
                Oii_Oiii = np.random.uniform(Oii_Oiii_min, Oii_Oiii_max)


                Nii_subcontext = Ha_Hb*Nii_Ha/4
                
                Oiii_subcontext = Oiii_Hb

                Sii_coefficient = np.random.uniform(0.2, 2.0)
                Sii_subcontext = Ha_Hb*Sii_Ha/(Sii_coefficient+1)
                Sii_0 = Sii_subcontext*Sii_coefficient
                Sii_1 = Sii_subcontext
                

                
                
                pre_compiled_data[i][24] = 1
                pre_compiled_data[i][32] = Ha_Hb
                pre_compiled_data[i][31] = Nii_subcontext*3
                pre_compiled_data[i][33] = Nii_subcontext
                pre_compiled_data[i][27] = Oiii_subcontext
                pre_compiled_data[i][34] = Sii_0
                pre_compiled_data[i][35] = Sii_1
                pre_compiled_data[i][18] = (Nii_subcontext*3+Nii_subcontext)/Nii_Oii
                



                

                
                pre_compiled_data[i][-1] = i / pre_compiled_data.shape[0]

        self.lambdas = pre_compiled_data[:, -1]*2000+1000

        dataset = tf.data.Dataset.from_tensor_slices(pre_compiled_data).cache()
        dataset = dataset.batch(1024).prefetch(buffer_size=tf.data.AUTOTUNE)

        gaussians_batch = []
        print(np.ones(len(pre_compiled_data))*np.log10(1.01, dtype = np.float32))
        for step, batch in enumerate(dataset):
            tmp_wavelength_template = tf.cast(wavelength_template, dtype=tf.float32)
            batch = tf.cast(batch, dtype=tf.float32)
            gaussians_batch_tmp = compute_batch_gaussians_tf(tmp_wavelength_template, batch[:, :-1], np.random.uniform(1.000001, 1.1, size = len(batch)))# np.ones(len(batch))*1.01)
            gaussians_batch_tmp = slice_2d_tensor_by_1d_indices(gaussians_batch_tmp, batch[:, -1])
            for i in gaussians_batch_tmp.numpy():
                gaussians_batch.append(i)
            sys.stdout.flush()
        gaussians_batch = np.array(gaussians_batch)

        self.gaussians_batch = gaussians_batch

        adjusted_columns = adjust_column_names(self.columns)

        df = pd.DataFrame(pre_compiled_data, columns=adjusted_columns)

        print(df)
        return gaussians_batch, self.lambdas, df
    
    def noise_injector(self):
        #inject gaussian noise
        np.random.seed(42)  # for reproducible results
        spectra = self.gaussians_batch

        # standard deviation of the Gaussian noise
        noise_level = 0.05  

        # Generate Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_level, size=spectra.shape)

        # Add noise to the original spectra
        noisy_gaussians_batch = spectra + noise
        
        self.noisy_gaussians_batch = noisy_gaussians_batch

        return noisy_gaussians_batch




@tf.function
def compute_batch_gaussians_tf(template, batch_amplitudes, batch_std_devs):
    template = tf.cast(template, dtype = tf.float32)
    batch_amplitudes = tf.cast(batch_amplitudes, dtype = tf.float32)
    batch_std_devs = tf.experimental.numpy.log10(tf.cast(batch_std_devs+5e-7, dtype = tf.float32))
    # Constants for the means
    means = tf.constant(np.log10([
        1033.82, 1215.24, 1240.81, 1305.53, 1335.31,
        1397.61, 1399.8, 1549.48, 1640.4, 1665.85,
        1857.4, 1908.734, 2326.0, 2439.5, 2799.117,
        3346.79, 3426.85, 3727.092, 3729.875, 3889.0,
        4072.3, 4102.89, 4341.68, 4364.436, 4862.68,
        4932.603, 4960.295, 5008.240, 6302.046, 6365.536,
        6529.03, 6549.86, 6564.61, 6585.27, 6718.29,
        6732.67, 3934.777, 3969.588, 4305.61, 5176.7,
        5895.6, 8500.36, 8544.44, 8664.52, 5578.5,
        5894.6, 6301.7, 7246.0
    ]), dtype=tf.float32)

    # Ensure batch_std_devs is a 1D array
    if len(batch_std_devs.shape) != 1:
        raise ValueError("batch_std_devs must be a 1D array")

    # Expand batch_std_devs to match the dimensions needed for broadcasting
    std_dev_expanded = tf.reshape(batch_std_devs, (-1, 1, 1))  # [B, 1, 1]

    # Compute Gaussian distributions
    expanded_template = tf.expand_dims(template, 1)  # [N, 1]
    expanded_means = tf.expand_dims(means, 0)  # [1, M]

    # Apply broadcasting to calculate the Gaussians
    gaussians = tf.exp(-0.5 * tf.square((expanded_template - expanded_means) / std_dev_expanded))  # [B, N, M]

    # Transpose and expand gaussians for correct broadcasting
    gaussians = tf.transpose(gaussians, perm=[0, 2, 1])  # [B, M, N]

    # Expand batch amplitudes
    batch_amplitudes_expanded = tf.expand_dims(batch_amplitudes, 2)  # [B, M, 1]

    # Multiply Gaussians by batch amplitudes
    gaussians_scaled = gaussians * batch_amplitudes_expanded  # [B, M, N]

    # Sum along the means axis
    summed_gaussians = tf.reduce_sum(gaussians_scaled, axis=1)  # [B, N]

    return summed_gaussians


@tf.function
def slice_2d_tensor_by_1d_indices(data_2d, indices_1d):
    # Calculate continuous indices within allowed bounds
    idx_min = indices_1d * 2000.0
    idx_max = idx_min + 8000.0

    max_len = tf.cast(tf.shape(data_2d)[1], tf.float32)
    idx_max = tf.minimum(idx_max, max_len)

    # Create a meshgrid for the batch and indices
    idx_range = tf.linspace(0.0, 1.0, 8000)  # Create 8000 points between 0 and 1
    idx_range = tf.expand_dims(idx_range, 0)  # Shape: [1, 8000]

    # Interpolate between idx_min and idx_max
    idxs = idx_min[:, None] + idx_range * (idx_max - idx_min)[:, None]
    idxs = tf.clip_by_value(idxs, 0.0, max_len - 1.0)  # Ensure indices are within valid range

    # Perform bilinear interpolation
    idx_floor = tf.floor(idxs)
    idx_ceil = idx_floor + 1
    idx_ceil = tf.minimum(idx_ceil, max_len - 1.0)  # Ensure idx_ceil does not exceed data length

    idx_floor = tf.cast(idx_floor, tf.int32)
    idx_ceil = tf.cast(idx_ceil, tf.int32)

    # Get values at idx_floor and idx_ceil
    def gather_vals(data, indices):
        batch_indices = tf.tile(tf.range(tf.shape(data)[0])[:, None], [1, tf.shape(indices)[1]])
        gather_indices = tf.stack([batch_indices, indices], axis=-1)
        return tf.gather_nd(data, gather_indices)

    values_floor = gather_vals(data_2d, idx_floor)
    values_ceil = gather_vals(data_2d, idx_ceil)

    # Calculate the weights for interpolation
    weights = idxs - tf.cast(idx_floor, tf.float32)

    # Interpolate between floor and ceil values
    result_tensor = values_floor * (1.0 - weights) + values_ceil * weights

    return result_tensor


def inverted_relu(x):
    return -tf.nn.relu(x)  # Negate the output of the standard ReLU

class ScaledSigmoid(Layer):
    def __init__(self, min_val, max_val, steepness=0.1, **kwargs):
        super(ScaledSigmoid, self).__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.steepness = steepness  # Introduce steepness parameter

    def call(self, inputs, **kwargs):
        # Scale inputs by steepness factor before applying the sigmoid
        sigmoid = tf.nn.sigmoid(inputs * self.steepness)
        # Scale the output of the sigmoid from min_val to max_val
        return self.min_val + (self.max_val - self.min_val) * sigmoid

    def get_config(self):
        config = super(ScaledSigmoid, self).get_config()
        config.update({
            'min_val': self.min_val,
            'max_val': self.max_val,
            'steepness': self.steepness  # Make sure to include steepness in the config
        })
        return config




@tf.function
def find_min_euclidean_distance_index(large_arrays, tiny_arrays, alpha=0.9999, k=2750):
    # Ensure data types are consistent
    large_arrays = tf.cast(large_arrays, dtype=tf.float32)
    tiny_arrays = tf.cast(tiny_arrays, dtype=tf.float32)

    # Dimensions of the inputs
    batch_size = tf.shape(large_arrays)[0]
    large_length = tf.shape(large_arrays)[1]
    tiny_length = tf.shape(tiny_arrays)[1]

    # Determine the number of sliding windows possible
    num_windows = large_length - tiny_length + 1

    # Create indices for all windows
    indices = tf.expand_dims(tf.range(num_windows), 0) + tf.expand_dims(tf.range(tiny_length), 1)
    # Shape of indices: (tiny_length, num_windows)

    # Batch and tile indices to gather windows across the batch
    indices = tf.tile(indices[None, :, :], [batch_size, 1, 1])
    # Shape of indices: (batch_size, tiny_length, num_windows)

    # Gather windows from the large arrays
    large_windows = tf.gather(large_arrays, indices, batch_dims=1)
    # Shape of large_windows: (batch_size, tiny_length, num_windows)

    # Compute squared differences and mean over the tiny_length dimension to get the MSE
    squared_diff = tf.square(large_windows - tiny_arrays[:, :, None])
    mse = tf.reduce_mean(squared_diff, axis=1)
    # Shape of mse: (batch_size, num_windows)

    # Compute dot products and cosine similarities
    dot_products = tf.reduce_sum(tf.multiply(large_windows, tiny_arrays[:, :, None]), axis=1)
    norm_large = tf.norm(large_windows, axis=1)
    norm_tiny = tf.norm(tiny_arrays, axis=1, keepdims=True)
    cosine_similarities = dot_products / (norm_large * norm_tiny)
    # Shape of cosine_similarities: (batch_size, num_windows)

    # Hybrid loss calculation
    hybrid_loss = alpha * mse + (1 - alpha) * -cosine_similarities  # Maximizing cosine similarity is minimizing its negative
    # Find the indices of the top k smallest hybrid loss values in each batch
    values, indices = tf.nn.top_k(-hybrid_loss, k, sorted=True)
    values = -values  # Convert back to positive values

    # Calculate the average of these top k values
    top_k_hybrid_loss_avg = tf.reduce_mean(values, axis=1)
    loss = tf.reduce_mean(top_k_hybrid_loss_avg)

    # Return the indices corresponding to the smallest hybrid loss (i.e., best matches)
    best_match_indices = tf.argmin(hybrid_loss, axis=1)

    return best_match_indices, loss, hybrid_loss


def train_step(opt, autoencoder, batch_data, lammies, wavelength_template, alpha):
    with tf.GradientTape() as tape:
        #alpha = tf.exp(log_alpha)
        decoded = autoencoder(batch_data, training=True)  # [batch_size, len(numbers)]
        gaussians_batch = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1])

        best_starting_lambdas = find_min_euclidean_distance_index(
            gaussians_batch, batch_data
        )
        lammies = tf.cast(lammies, dtype=tf.float32)
        delta_lam = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(lammies, dtype=tf.float32)- tf.cast(best_starting_lambdas, dtype=tf.float32))))
        #print(lammies)
        #print(best_starting_lambdas)
        #print(tf.sqrt(tf.reduce_mean(tf.square(lammies-tf.cast(best_starting_lambdas, dtype=tf.float32)))))
        #print(decoded)
        
        gaussians_batch = slice_2d_tensor_by_1d_indices(gaussians_batch, decoded[:, -1])
        # Reconstruction loss

        true_lammy_loss = tf.reduce_mean(tf.square(lammies-(decoded[:, -1]*2000+1000)))
        reconstruction_loss = tf.reduce_mean(tf.square(tf.cast(batch_data, tf.float32) - tf.cast(gaussians_batch, tf.float32)))
        lammy_loss = tf.reduce_mean(tf.square(tf.cast(best_starting_lambdas, tf.float32) - tf.cast((decoded[:, -1] * 2000) + 1000, tf.float32)))
        loss =  lammy_loss + alpha * reconstruction_loss

    #trainable_variables = autoencoder.trainable_variables# + [log_alpha]
    autoencoder_gradients = tape.gradient(loss, autoencoder.trainable_variables)
    opt.apply_gradients(zip(autoencoder_gradients, autoencoder.trainable_variables))
    del best_starting_lambdas, lammies, decoded
    #gc.collect()()

    return reconstruction_loss, lammy_loss, loss, gaussians_batch, true_lammy_loss, opt, delta_lam
 



#@profile
def pretrain_step(opt, autoencoder, decoder, batch_data, lammies, wavelength_template, alpha, lambdas, test):
    
    with tf.GradientTape(persistent=True) as tape:
        #alpha = tf.exp(log_alpha)
        decoded = autoencoder(batch_data, training=True)  # [batch_size, len(numbers)]
        full_decoded = decoder(decoded, training=True)
        #decoded = tf.squeeze(decoded, axis=1)

        static_template = tf.concat([tf.ones((len(batch_data), 36), dtype=tf.float32), -tf.ones((len(batch_data), 8), dtype=tf.float32), tf.ones((len(batch_data), 4), dtype=tf.float32)], axis=1)#/tf.reduce_mean(tf.concat([tf.ones((len(batch_data), 36), dtype=tf.float32), -tf.ones((len(batch_data), 8), dtype=tf.float32), tf.ones((len(batch_data), 4), dtype=tf.float32)], axis=1))
        #pseudo_templates = compute_batch_gaussians_tf(wavelength_template, static_template)
        #gaussians_batch = slice_2d_tensor_by_1d_indices(pseudo_templates, decoded[:, -1])
        

        
        lammies = tf.cast(lammies, dtype=tf.float32)
        delta_lam = tf.sqrt(tf.reduce_mean(tf.square(tf.cast(lammies, dtype=tf.float32)- tf.cast(lambdas, dtype=tf.float32))))
        #print(lammies)
        #print(best_starting_lambdas)
        #print(tf.sqrt(tf.reduce_mean(tf.square(lammies-tf.cast(best_starting_lambdas, dtype=tf.float32)))))
        #print(decoded)
        
        gaussians_batch_full = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1], decoded[:, -1])#  static_template
        
        gaussians_batch_full_norm = tf.norm(gaussians_batch_full, ord='euclidean', axis=1, keepdims=True)
        
        gaussians_batch_full = gaussians_batch_full/gaussians_batch_full_norm

        # Distribute the cosine similarity calculation
        best_starting_lambdas, test_loss, hybrid_loss = find_min_euclidean_distance_index(gaussians_batch_full, batch_data, alpha = alpha, k = test)
        #print(best_starting_lambdas, test_loss)
        gaussians_batch = slice_2d_tensor_by_1d_indices(gaussians_batch_full, tf.cast((best_starting_lambdas)/2000, dtype = tf.float32))


        #Reconstruction loss
        #brute_force_lambda_loss = tf.reduce_mean(tf.square(tf.cast(best_starting_lambdas, tf.float32)-tf.cast((decoded[:, -1]*2000+1000), tf.float32)))
        #brute_force_lambda_loss = 0

        true_lammy_loss = tf.reduce_mean(tf.square(lammies-(decoded[:, -1]*2000+1000)))
        reconstruction_loss = tf.reduce_mean(tf.square(tf.cast(batch_data, tf.float32) - tf.cast(gaussians_batch, tf.float32)))
        #reconstruction_loss = 0
        
        full_recon_loss = tf.reduce_mean(tf.square(tf.cast(batch_data, tf.float32) - tf.cast(full_decoded, tf.float32)))
        #reconstruction_loss = 0
        lammy_loss = tf.reduce_mean(tf.square(tf.cast(lambdas, tf.float32) - tf.cast((decoded[:, -1] * 2000) + 1000, tf.float32)))
        lammy_loss = 0


        loss = (tf.reduce_mean(test_loss))
    #trainable_variables = autoencoder.trainable_variables# + [log_alpha]
    decoder_gradients = tape.gradient(full_recon_loss, decoder.trainable_variables)
    autoencoder_gradients = tape.gradient(loss, autoencoder.trainable_variables)
    grads_and_vars = list(chain(zip(decoder_gradients, decoder.trainable_variables),
            zip(autoencoder_gradients, autoencoder.trainable_variables),
            ))

                
    opt.apply_gradients(grads_and_vars)

    del lammies, decoded
    #gc.collect()


    return reconstruction_loss, lammy_loss, loss, gaussians_batch, true_lammy_loss, opt, delta_lam, full_decoded, full_recon_loss, hybrid_loss, best_starting_lambdas



def validation_step(autoencoder, decoder, batch_data, lammies, wavelength_template, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((batch_data, lammies))
    dataset = dataset.batch(batch_size)

    reconstruction_loss_accumulated = []
    gaussians_batch_accumulated = []
    gaussians_batch_full_accumulated = []
    true_lammy_loss_accumulated = []
    full_recon_loss_accumulated = []
    decoded_accumulated = []
    best_lambdas_accumulated = []
    true_decoded_accumulated = []

    for batch_data_segment, lammies_segment in dataset:
        #alpha = tf.exp(log_alpha)
        decoded = autoencoder.predict(batch_data_segment, verbose = 0)  # [batch_size, len(numbers)]
        #decoded = tf.squeeze(decoded, axis=1)

        #static_template = tf.concat([tf.ones((len(batch_data), 36), dtype=tf.float32), -tf.ones((len(batch_data), 8), dtype=tf.float32), tf.ones((len(batch_data), 4), dtype=tf.float32)], axis=1)
        #pseudo_templates = compute_batch_gaussians_tf(wavelength_template, static_template)
        #gaussians_batch = slice_2d_tensor_by_1d_indices(pseudo_templates, decoded[:, -1])
        

        
        lammies = tf.cast(lammies_segment, dtype=tf.float32)
        #print(lammies)
        #print(best_starting_lambdas)
        #print(tf.sqrt(tf.reduce_mean(tf.square(lammies-tf.cast(best_starting_lambdas, dtype=tf.float32)))))
        #print(decoded)
        
        gaussians_batch_full = compute_batch_gaussians_tf(wavelength_template, decoded[:, :-1], decoded[:, -1])
        gaussians_batch_full_norm = tf.norm(gaussians_batch_full, ord='euclidean', axis=1, keepdims=True)
        gaussians_batch_full = gaussians_batch_full/gaussians_batch_full_norm

        
        # Distribute the cosine similarity calculation
        best_starting_lambdas, test_loss, hybrid_loss = find_min_euclidean_distance_index(gaussians_batch_full, batch_data_segment)
        #print(best_starting_lambdas, test_loss)
        gaussians_batch = slice_2d_tensor_by_1d_indices(gaussians_batch_full, tf.cast((best_starting_lambdas)/2000, dtype = tf.float32))


        #Reconstruction loss
        #brute_force_lambda_loss = tf.reduce_mean(tf.square(tf.cast(best_starting_lambdas, tf.float32)-tf.cast((decoded[:, -1]*2000+1000), tf.float32)))
        #brute_force_lambda_loss = 0

        true_lammy_loss = tf.reduce_mean(tf.square(lammies-(decoded[:, -1]*2000+1000)))
        reconstruction_loss = tf.reduce_mean(tf.square(tf.cast(batch_data_segment, tf.float32) - tf.cast(gaussians_batch, tf.float32)))
        #reconstruction_loss = 0
        

        loss = (tf.reduce_mean(test_loss))
        
        reconstruction_loss_accumulated.append(reconstruction_loss)
        true_lammy_loss_accumulated.append(true_lammy_loss)
        full_recon_loss_accumulated.append(loss)

        
        for i in range(len(decoded[:, -1])):
            gaussians_batch_accumulated.append(gaussians_batch[i])
            true_decoded_accumulated.append(decoded[i])
            decoded_accumulated.append(decoded[:, -1]*2000+1000)
            best_lambdas_accumulated.append(best_starting_lambdas.numpy()[i])
            gaussians_batch_full_accumulated.append(gaussians_batch_full.numpy()[i])

    # Average losses over all batches
    total_reconstruction_loss = tf.reduce_mean(reconstruction_loss_accumulated)
    total_true_lammy_loss = tf.reduce_mean(true_lammy_loss_accumulated)
    total_full_recon_loss = tf.reduce_mean(full_recon_loss_accumulated)

    return total_reconstruction_loss, gaussians_batch_accumulated, total_true_lammy_loss, total_full_recon_loss, decoded_accumulated,  best_lambdas_accumulated, gaussians_batch_full_accumulated, np.array(true_decoded_accumulated)


def build_model():

    
    input_img = Input(shape=(8000, 1))  # Adjusted for a 1D CNN, assuming each spectrum is 8000 points long with 1 channel


    """y = Flatten()(input_img)
    y = Dense(1024)(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)"""

    x = tf.keras.layers.Conv1D(filters=2, activation='relu',kernel_size=6, strides=1, padding='valid')(input_img)
    x = BatchNormalization()(x)
    #x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid')(x)
    x = Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(filters=4, activation='relu',kernel_size=6, strides=1, padding='valid')(x)
    #x = BatchNormalization()(x)
    #x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid')(x)
    x = Dropout(0.3)(x)


    x = Flatten()(x)
    #x = Concatenate()([x, y])


    sigmoid_part = Dense(1, activation='linear')(x)#, bias_initializer=tf.keras.initializers.Constant(np.log10(1.01)))(x)  # Use linear here because ScaledSigmoid applies the sigmoid
    sigmoid_part = ScaledSigmoid(min_val=1.000001, max_val=1.1)(sigmoid_part)


    both0 = Dense(3, activation='linear')(x)#, bias_initializer=tf.keras.initializers.Constant(1.0)
    emission0 = Dense(4, activation='relu')(x)
    both1 = Dense(2, activation='linear')(x)
    emission1 = Dense(1, activation='relu')(x)
    both2 = Dense(1, activation='linear')(x)
    emission2 = Dense(2, activation='relu')(x)
    both3 = Dense(4, activation='linear')(x)
    emission3 = Dense(2, activation='relu')(x)
    both4 = Dense(1, activation='linear')(x)
    emission4 = Dense(1, activation='relu')(x)
    both5 = Dense(2, activation='linear')(x)
    emission5 = Dense(1, activation='relu')(x)
    both6 = Dense(1, activation='linear')(x)
    emission6 = Dense(7, activation='relu')(x)
    both7 = Dense(1, activation='linear')(x)
    emission7 = Dense(3, activation='relu')(x)

    absorption = tf.keras.layers.Dense(units=8, activation=inverted_relu)(x)
    skyline = Dense(4, activation='relu')(x)



    # Concatenating the two parts back together
    decoded = Concatenate()([both0, emission0, both1, emission1, both2, emission2, both3, emission3, both4, emission4, both5,
                             emission5, both6, emission6, both7, emission7, absorption, skyline, sigmoid_part])

    x = Dense(64, activation='relu')(decoded)
    x = Dense(512, activation='relu')(decoded)
    #x = Dense(2048, activation='relu')(decoded)    

    full_decoded = Dense(8000, activation='linear')(x)


    autoencoder = Model(input_img, decoded)
    decoder = Model(decoded, full_decoded)
    

    return autoencoder, decoder





#@profile
def train_and_evaluate_autoencoder(data, validation, lammy, validation_lammy, train_dataframe, validation_dataframe, output_dir, epochs=200, batch_size=16, patience=25):
    """
    Trains an autoencoder on the provided data, evaluates its training loss, and returns the encoder model.
    """
    """if not os.path.exists(output_dir):
        os.makedirs(output_dir)"""

    autoencoder, decoder = build_model()

    print(autoencoder.summary())
    
    emissions = np.log10([
                        1033.82, 1215.24, 1240.81, 1305.53, 1335.31,
                        1397.61, 1399.8, 1549.48, 1640.4, 1665.85,
                        1857.4, 1908.734, 2326.0, 2439.5, 2799.117,
                        3346.79, 3426.85, 3727.092, 3729.875, 3889.0,
                        4072.3, 4102.89, 4341.68, 4364.436, 4862.68,
                        4932.603, 4960.295, 5008.240, 6302.046, 6365.536,
                        6529.03, 6549.86, 6564.61, 6585.27, 6718.29,
                        6732.67, 3934.777, 3969.588, 4305.61, 5176.7,
                        5895.6, 8500.36, 8544.44, 8664.52, 5578.5,
                        5894.6, 6301.7, 7246.0
                        ])

    opt = Adam(learning_rate=0.00002)#, clipnorm=1)
    test = 25
        
    wavelength_template = tf.constant(np.log10(np.logspace(np.log10(1000), np.log10(12000), num=11000, base=10)), dtype = tf.float32)


    #lambdas = precalculate_lambdas(data, wavelength_template, 64)
    lambdas = tf.zeros(len(data))

    dataset = tf.data.Dataset.from_tensor_slices((data, lammy, lambdas))
    dataset = dataset.shuffle(buffer_size=len(data), reshuffle_each_iteration=True).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


    alpha = 0

    for epoch in range(epochs):
        
        start_time = time.time()
        if epoch > 3:
            alpha = 0.5
        collect_lammy_loss = []
        collect_reconstruction_loss = []
        collect_true_lammy_loss = []
        collect_delta_lam = []
        collect_loss = []
        for step, (batch_data, lammies, lambdas) in enumerate(dataset):
            if epoch < 5000:
                #alpha=5e-7
                #with strategy.scope():
                reconstruction_loss, lammy_loss, loss, gaussians_batch, true_lammy_loss, opt, delta_lam, full_decoded, full_recon_loss, hybrid_loss, best_starting_lambdas = pretrain_step(opt, autoencoder, decoder, batch_data, lammies, wavelength_template, alpha, lambdas, test)
                collect_delta_lam.append(delta_lam)
                collect_reconstruction_loss.append(reconstruction_loss)
                collect_true_lammy_loss.append(true_lammy_loss)
                collect_loss.append(loss)
                collect_lammy_loss.append(lammy_loss)
            else:
                alpha = 1e-10
                reconstruction_loss, lammy_loss, loss, gaussians_batch, true_lammy_loss, opt, delta_lam = train_step(opt, autoencoder, batch_data, lammies, wavelength_template, alpha)
        """if epoch % 2:

            test = test-1"""
        

        lammy_loss = np.mean(collect_lammy_loss)
        reconstruction_loss = np.mean(collect_reconstruction_loss)
        true_lammy_loss = np.mean(collect_true_lammy_loss)
        loss = np.mean(collect_loss)
        delta_lam = np.mean(collect_delta_lam)
        print(alpha)
        print(f'\nEPOCH:{epoch}\n TRUE LAMMY LOSS: {true_lammy_loss}\n LAMMY LOSS: {lammy_loss}\n RECONSTRUCTION LOSS: {reconstruction_loss}\n FULL RECONSTRUCTION LOSS: {full_recon_loss}\n TOTAL LOSS: {loss}\n DELTA LAMBDA: {delta_lam}')
        sys.stdout.flush()

        valid_reconstruction_loss, valid_gaussians_batch, valid_true_lammy_loss, valid_full_recon_loss, valid_decoded, valid_best_lambdas, valid_full_gaussians, true_valid_decoded = validation_step(autoencoder, decoder, validation, validation_lammy, wavelength_template)
        lr = opt.learning_rate.numpy()
        print(f'\nVALIDATION:\n VALIDATION FULL RECON LOSS: {valid_full_recon_loss}\n VALIDATION RECON LOSS: {valid_reconstruction_loss} \n VALIDATION LAMMY LOSS: {valid_true_lammy_loss}\n ADAM LEARNING RATE: {lr}\n ')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"EPOCH RUNTIME: {execution_time} s")
        model_filename = f'autoencoder_model_attention_test.keras'
        model_save_path = os.path.join(output_dir, model_filename)

        autoencoder.save('auto_ez_prototype.keras')
        print(f"Autoencoder Model saved to {model_save_path}")
        if epoch % 5 == 0:
            
            wavelength_template_temp = wavelength_template.numpy()

            plt.figure()
            plt.plot(wavelength_template_temp[np.array(lammies[0]-1000, dtype = np.int32): np.array(lammies[0]-1000, dtype = np.int32)+8000], batch_data[0])
            plt.plot(wavelength_template_temp[np.array(best_starting_lambdas[0], dtype = np.int32): np.array(best_starting_lambdas[0], dtype = np.int32)+8000], gaussians_batch[0], alpha = 0.8)
            #plt.plot(full_decoded[0], alpha = 0.5)
            plt.xlabel('Wavelength [log($\AA$)]')
            plt.ylabel('Flux')
            plt.savefig(f'output_plots/emission_line{epoch}.pdf')

            plt.figure()
            plt.ylabel('Flux')
            plt.xlabel(r'Wavelength ($\AA$)')  
            print(valid_full_gaussians[0][0])
            first_validation = validation_dataframe.iloc[0]
            locs = [loc for loc, val in enumerate(first_validation.values[:-1]) if val>0]
            for loc in locs:

                closest_index = (np.abs(wavelength_template_temp - emissions[loc])).argmin()

                plt.axvline(x=wavelength_template_temp[closest_index], color='orange', linestyle=':')
                plt.text(wavelength_template_temp[closest_index], valid_full_gaussians[0][closest_index], validation_dataframe.columns[loc], color='orange', rotation=0, verticalalignment='bottom', horizontalalignment='center', fontsize = 8)

                closest_index = (np.abs(wavelength_template_temp - emissions[loc]+(validation_lambdas[0]*2000+1000))).argmin()



            plt.plot(wavelength_template_temp[np.array(validation_lammy[0]-1000, dtype = np.int32): np.array(validation_lammy[0]-1000, dtype = np.int32)+8000], validation[0])
            plt.plot(wavelength_template_temp[np.array(valid_best_lambdas[0], dtype = np.int32): np.array(valid_best_lambdas[0], dtype = np.int32)+8000], valid_gaussians_batch[0], alpha = 0.8)
            plt.xlabel('Wavelength [log($\AA$)]')
            plt.ylabel('Flux')
            plt.savefig(f'validation_plots/emission_line{epoch}.pdf')

            plt.figure()
            plt.ylabel('Predicted Redshift')
            plt.xlabel('True Redshift')

            valid_best_lambdas_plot = np.array(valid_best_lambdas)
            validation_lammy_plot = np.round(np.array(validation_lammy, dtype = np.int32)-1000)
            z_pred = np.array((10.0**(np.array(wavelength_template_temp[valid_best_lambdas_plot]-wavelength_template[0], dtype = np.float32)))-1)
            z_true = np.array((10**(np.array(wavelength_template_temp[validation_lammy_plot]-wavelength_template[0], dtype = np.float32)))-1)
            z_outlier = np.abs([i for i in (z_pred-z_true)/z_true])
            print(f'Number of outliers: {len([i for i in z_outlier if i>0.2])}')
            
            test1 = [i for i, n in enumerate(z_outlier) if n>0.2]

            plt.plot(z_true, z_pred, 'o', markersize = 0.5)
            plt.savefig(f'z_plots/z_plot{epoch}.pdf')



            print(np.shape(true_valid_decoded[:, 24]))
            plt.figure()
            plt.plot(true_valid_decoded[:, 32]/true_valid_decoded[:, 24], validation_dataframe['Hα']/validation_dataframe['Hβ'], 'o', markersize = 0.5)
            plt.xlabel('Predicted H_a/HB')
            plt.ylabel('True Ha/HB')
            plt.savefig(f'ha_hb/ha_hb{epoch}.pdf')



            # Yo# Your existing code
            plt.figure()
            plt.ylabel('loss')
            plt.xlabel('indice')
            # Compute x-values for the plot
            x_values = (10**((np.array(range(len(hybrid_loss[0])))*(wavelength_template_temp[1]-wavelength_template_temp[0]))))-1
            # Plot the lines
            plt.plot(x_values, hybrid_loss[0])
            # Compute the x-value for the orange dot
            orange_dot_x = (10**((tf.cast(tf.floor(lammies[0]), tf.int32).numpy()*(wavelength_template_temp[1]-wavelength_template_temp[0]))))-1
            # Plot the orange dot
            plt.plot(orange_dot_x, hybrid_loss[0][tf.cast(tf.floor(lammies[0]), tf.int32).numpy()].numpy(), 'o')
            # Plot vertical dotted line at the x-value of the orange dot
            plt.axvline(x=orange_dot_x, color='orange', linestyle='dotted')
            # Find the x-value at the minimum of the first line
            min_x_value = x_values[np.argmin(hybrid_loss[0])]
            # Plot vertical dotted line at the x-value of the minimum of the first line
            plt.axvline(x=min_x_value, color='blue', linestyle='dotted')
            # Calculate the difference between the x-values
            x_difference = abs(orange_dot_x - min_x_value)
            # Create a text box with the x-values and the difference
            textstr = '\n'.join((
                f'True Redshift: {orange_dot_x:.2f}',
                f'Predicted REdshift: {min_x_value:.2f}',
                f'$\Delta$z: {x_difference:.2f}',
            ))
            # Add a text box in the top right corner
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.gcf().text(0.95, 0.95, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
            # Save the figure
            plt.savefig(f'hybrid/hybrid_check{epoch}.pdf')

            if os.path.exists(f'outliers/{epoch}'):
                shutil.rmtree(f'outliers/{epoch}')
            
            os.makedirs(f'outliers/{epoch}')

            outlier_idxs = [i for i, n in enumerate(z_outlier) if n>0.2]

            for outlier_idx in outlier_idxs:
                plt.figure()
                plt.ylabel('Flux')
                plt.xlabel(r'Wavelength ($\AA$)')  
                first_validation = validation_dataframe.iloc[outlier_idx]
                locs = [loc for loc, val in enumerate(first_validation.values[:-1]) if val>0]
                for loc in locs:

                    closest_index = (np.abs(wavelength_template_temp - emissions[loc])).argmin()

                    plt.axvline(x=wavelength_template_temp[closest_index], color='orange', linestyle=':')
                    plt.text(wavelength_template_temp[closest_index], valid_full_gaussians[outlier_idx][closest_index], validation_dataframe.columns[loc], color='orange', rotation=0, verticalalignment='bottom', horizontalalignment='center', fontsize = 8)

                    closest_index = (np.abs(wavelength_template_temp - emissions[loc]+(validation_lambdas[outlier_idx]*2000+1000))).argmin()



                plt.plot(wavelength_template_temp[np.array(validation_lammy[outlier_idx]-1000, dtype = np.int32): np.array(validation_lammy[outlier_idx]-1000, dtype = np.int32)+8000], validation[outlier_idx])
                plt.plot(wavelength_template_temp[np.array(valid_best_lambdas[outlier_idx], dtype = np.int32): np.array(valid_best_lambdas[outlier_idx], dtype = np.int32)+8000], valid_gaussians_batch[outlier_idx], alpha = 0.8)
                plt.xlabel('Wavelength [log($\AA$)]')
                plt.ylabel('Flux')
                plt.savefig(f'outliers/{epoch}/emission_line{outlier_idx}.pdf')


            plt.close('all')
 

        del reconstruction_loss, lammy_loss, loss, gaussians_batch, true_lammy_loss, valid_gaussians_batch

    
    print(f"Autoencoder Model saved to {model_save_path}")

    return autoencoder





def main():
    # Print the current script
    with open(__file__, 'r') as file:
        # Read the content of the file
        content = file.read()
        # Print the content
        print(content)



if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork')  # or 'forkserver'
    except:
        True == True
    #main()

    wavelength_template = np.log10(np.logspace(np.log10(1000), np.log10(12000), num=11000, base=10))
    print(wavelength_template)
    data_points = 6000
    min_amplitude = 1
    max_amplitude = 10
    min_lambda = 3000
    max_lambda = 11000

    data_producer = data_producer(data_points, min_amplitude, max_amplitude, min_lambda, max_lambda)

    pure_data, lambdas, dataframe = data_producer.initialize_data(wavelength_template, vary_height = False, full_line_range=False, true_ratios = True)
    noisy_data = data_producer.noise_injector()

    noisy_norms = np.linalg.norm(noisy_data, axis=1, keepdims=True)
    pure_norms = np.linalg.norm(pure_data, axis=1, keepdims=True)

    noisy_data = noisy_data/noisy_norms
    pure_data = pure_data/pure_norms


    indices = np.arange(lambdas.shape[0])
    np.random.shuffle(indices)

    lambdas = lambdas[indices]
    pure_data = pure_data[indices]
    noisy_data = noisy_data[indices]

    train_dataframe = dataframe.iloc[:5000]
    train_lambdas = lambdas[:5000]
    train_data = noisy_data[:5000]

    validation_dataframe = dataframe.iloc[5000:]
    validation_lambdas = lambdas[5000:]
    validation_data = noisy_data[5000:]

    
    train_and_evaluate_autoencoder(train_data, validation_data, train_lambdas,validation_lambdas, train_dataframe, validation_dataframe, '/fred/oz149/aabd/f_width_f_height_v_lines/ai_test/pseudo_amplitudes/', epochs=2000, batch_size= 32, patience=25)

 
    model = load_model('ai_test/pseudo_amplitudes/autoencoder_model.keras',  custom_objects={'inverted_relu': inverted_relu, 'ScaledSigmoid': ScaledSigmoid})


