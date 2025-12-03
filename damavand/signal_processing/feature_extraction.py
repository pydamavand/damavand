import pandas as pd
import numpy as np
import pycatch22

def feature_extractor(signals, features):
    """
    feature_extractor(signals, features) - Extracting features from input signals

    Arguments:
    signals -- A pd.DataFrame() including signals in its rows
    features -- A python dict where:
                - keys are feature names
                - values are tuples of (function, args, kwargs) where:
                  * function: the feature extraction function
                  * args: tuple of positional arguments (optional)
                  * kwargs: dict of keyword arguments (optional)
                Example: {
                    'feature1': (func1, (), {}),
                    'feature2': (func2, (arg1,), {'param1': value1}),
                }

    Return Value:
    A pd.DataFrame() containing the feature values for each signal
    """
    def apply_feature(row, func_tuple):
        """
        apply_feature(row, func_tuple) - Applies a feature extraction function to a given row.

        Arguments:
        row -- A pd.Series() including a signal
        func_tuple -- A tuple of (function, args, kwargs) where:
                        * function: the feature extraction function
                        * args: tuple of positional arguments (optional)
                        * kwargs: dict of keyword arguments (optional)

        Return Value:
        The result of the feature extraction function when applied to the given row
        """
        func, args, kwargs = func_tuple
        return func(row, *args, **kwargs)

    feature_values = signals.apply(
        lambda row: pd.Series([
            apply_feature(row, feat_info) for feat_info in features.values()
        ]),
        axis=1
    )
    feature_values.columns = features.keys()
    
    return feature_values


# Time domain signals

def smsa(arr):
    return np.square(np.mean(np.sqrt(np.abs(arr))))

def rms(arr):
    return np.sqrt(np.mean(np.square(arr)))

def peak(arr):
    return np.max(np.abs(arr))

def crest_factor(arr):
    return peak(arr) / rms(arr)

def clearance_factor(arr):
    return peak(arr) / smsa(arr)

def shape_factor(arr):
    return rms(arr) / np.mean(np.abs(arr))

def impulse_factor(arr):
    return peak(arr) / np.mean(np.abs(arr))

# Frequency domain features

def spectral_centroid(spectrum, freq_axis):
    return np.sum(spectrum * freq_axis) / np.sum(spectrum)

def P17(spectrum, freq_axis):
    return np.sqrt(np.mean(np.square(np.subtract(freq_axis, spectral_centroid(spectrum, freq_axis))) * spectrum))

def P18(spectrum, freq_axis):
    return np.sqrt(np.sum(np.square(freq_axis) * spectrum) / np.sum(spectrum))

def P19(spectrum, freq_axis):
    return np.sum(np.power(freq_axis, 4) * spectrum) / np.sum(np.square(freq_axis) * spectrum)

def P20(spectrum, freq_axis):
    return np.sum(np.square(freq_axis) * spectrum) / np.sqrt(np.sum(spectrum) * np.sum(np.power(freq_axis, 4) * spectrum))

def P21(spectrum, freq_axis):
    return P17(spectrum, freq_axis) / spectral_centroid(spectrum, freq_axis)

def P22(spectrum, freq_axis):
    return np.mean(np.power(np.subtract(freq_axis, spectral_centroid(spectrum, freq_axis)), 3) * spectrum) / np.power(P17(spectrum, freq_axis), 3)

def P23(spectrum, freq_axis):
    return np.mean(np.power(np.subtract(freq_axis, spectral_centroid(spectrum, freq_axis)), 4) * spectrum) / np.power(P17(spectrum, freq_axis), 4)

def P24(spectrum, freq_axis):
    return np.mean(np.sqrt(np.subtract(freq_axis, spectral_centroid(spectrum, freq_axis))) * spectrum) / np.sqrt(P17(spectrum, freq_axis))

# Catch-22 feature extraction

def catch_features(df, include_additionals = True):
  feature_names, feature_values = [], []
  for index, row in df.iterrows():
    results = pycatch22.catch22_all(row.to_numpy(), short_names = True)

    if include_additionals:

      results['names'].extend(["DN_Mean", "DN_Spread_Std"])
      results['values'].extend([np.mean(row.to_numpy()), np.std(row.to_numpy())])

    feature_values.append(results["values"])

  return pd.DataFrame(feature_values, columns=results["names"])