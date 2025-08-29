# Feature extraction

This directory is responsible for extracting features from denoised and segmented EEG data. Features are extracted in three domains: Time-domain, Frequency-domain and Time-Frequency domain. Each of these domains has its own file with the corresponding functions implemented. Each file has a function called `<DOMAIN>_feature_extraction`, which takes in the denoised EEG data from one song and returns a dictionary containing all relevant features. Below, you find an overview of all computed features. In [feature_extraction.py](feature_extraction.py), there is an interface function that returns a dictionary where each feature from each domain are calculated.

| **Domain**         | **Feature Name**        | **Description**                                           |
| ------------------ | ----------------------- | --------------------------------------------------------- |
| **Time**           | `max`                   | Maximum value of the signal                               |
|                    | `min`                   | Minimum value of the signal                               |
|                    | `std`                   | Standard deviation of the signal                          |
|                    | `mean`                  | Mean value of the signal                                  |
|                    | `median`                | Median value of the signal                                |
|                    | `var`                   | Variance of the signal                                    |
|                    | `range`                 | Peak-to-peak range of the signal                          |
|                    | `skew`                  | Skewness of the signal distribution                       |
|                    | `kurtosis`              | Kurtosis of the signal distribution                       |
|                    | `rms`                   | Root mean square value of the signal                      |
|                    | `zcr`                   | Zero crossing rate                                        |
|                    | `h_mobility`            | Hjorth mobility parameter                                 |
|                    | `h_complexity`          | Hjorth complexity parameter                               |
|                    | `energy`                | Total energy (sum of squares) of the signal               |
|                    | `shannon_entropy`       | Shannon entropy of the signal                             |
| **Frequency**      | `spectral_entropy`      | Shannon entropy of the normalized power spectral density  |
|                    | `DELTA_relative`        | Relative power in the delta band                          |
|                    | `THETA_relative`        | Relative power in the theta band                          |
|                    | `ALPHA_relative`        | Relative power in the alpha band                          |
|                    | `BETA_relative`         | Relative power in the beta band                           |
|                    | `ALPHA_BETA_ratio`      | Ratio of alpha power to beta power                        |
|                    | `THETA_ALPHA_ratio`     | Ratio of theta power to alpha power                       |
|                    | `THETA_BETA_ratio`      | Ratio of theta power to beta power                        |
| **Time-Frequency** | `tfd_band_i_mean`       | Mean of wavelet coefficients in subband *i*               |
|                    | `tfd_band_i_median`     | Median of wavelet coefficients in subband *i*             |
|                    | `tfd_band_i_std`        | Standard deviation of wavelet coefficients in subband *i* |
|                    | `tfd_band_i_variance`   | Variance of wavelet coefficients in subband *i*           |
|                    | `tfd_band_i_entropy`    | Shannon entropy of normalized wavelet energy in subband *i*|
|                    | `tfd_band_i_relative_energy` | Relative energy of subband *i* with respect to total wavelet energy |
|                    | `tfd_band_i_skewness`   | Skewness of wavelet coefficients in subband *i*           |
|                    | `tfd_band_i_kurtosis`   | Kurtosis of wavelet coefficients in subband *i*           |
