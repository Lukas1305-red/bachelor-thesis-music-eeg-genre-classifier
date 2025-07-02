# Feature extraction

This directory is responsible for extracting features from denoised and segmented EEG data. Features are extracted in three domains: Time-domain, Frequency-domain and Time-Frequency domain. Each of these domains has its own file with the corresponding functions implemented. Each file has a function called `<DOMAIN>_feature_extraction`, which takes in the denoised EEG data from one song and returns a dictionary containing all relevant features. Below, you find an overview of all computed features. In [feature_extraction.py](feature_extraction.py), there is an interface function that returns a dictionary where each feature from each domain are calculated.

| **Domain**         | **Feature Name**                  | **Description**                                                                               |
| ------------------ | --------------------------------- | --------------------------------------------------------------------------------------------- |
| **Time**           | `max`                             | Maximum value of the signal                                                                   |
|                    | `min`                             | Minimum value of the signal                                                                   |
|                    | `std`                             | Standard deviation                                                                            |
|                    | `mean`                            | Mean (average)                                                                                |
|                    | `median`                          | Median                                                                                        |
|                    | `var`                             | Variance                                                                                      |
|                    | `range`                           | Peak-to-peak value (max - min)                                                                |
|                    | `skew`                            | Skewness (asymmetry)                                                                          |
|                    | `kurtosis`                        | Kurtosis (tailedness)                                                                         |
|                    | `rms`                             | Root mean square                                                                              |
|                    | `zcr`                             | Zero crossing rate                                                                            |
|                    | `h_mobility`                      | Hjorth mobility                                                                               |
|                    | `h_complexity`                    | Hjorth complexity                                                                             |
|                    | `mad`                             | Mean absolute deviation                                                                       |
|                    | `moment3`                         | 3rd central moment                                                                            |
|                    | `moment4`                         | 4th central moment                                                                            |
|                    | `ar_model.coefficients`           | Autoregressive model coefficients (lags=10)                                                   |
|                    | `ar_model.intercept`              | Intercept term of AR model                                                                    |
|                    | `ar_model.aic`                    | Akaike Information Criterion                                                                  |
|                    | `ar_model.bic`                    | Bayesian Information Criterion                                                                |
|                    | `ar_model.residuals`              | Residuals of the AR model                                                                     |
| **Frequency**      | `spectral_entropy`                | Shannon entropy of the normalized PSD                                                         |
|                    | `mean_freq`                       | Mean frequency of the PSD                                                                     |
|                    | `rms_bandwidth`                   | Root mean square bandwidth                                                                    |
|                    | `ALPHA`, `BETA`, `THETA`, `DELTA` | Absolute band power via Welch, FFT, or Periodogram                                            |
|                    | `*_relative`                      | Relative band power in % of total power (e.g. `ALPHA_relative`)                               |
|                    | `*_peak`                          | Peak frequency in each band (e.g. `ALPHA_peak`)                                               |
|                    | `alpha_beta_ratio`                | Ratio: Alpha power / Beta power                                                               |
|                    | `theta_alpha_ratio`               | Ratio: Theta power / Alpha power                                                              |
|                    | `theta_beta_ratio`                | Ratio: Theta power / Beta power                                                               |
| **Time-Frequency** | `dwt.approx.mean`, etc.           | Discrete Wavelet Transform (mean, std, skew, kurtosis, entropy of approximation coefficients) |
|                    | `dwt.detailed.mean`, etc.         | Discrete Wavelet Transform (same stats for detail coefficients)                               |
|                    | `cwt.<band>.mean`, etc.           | Continuous Wavelet Transform: mean, std, skewness, kurtosis, entropy per band                 |
