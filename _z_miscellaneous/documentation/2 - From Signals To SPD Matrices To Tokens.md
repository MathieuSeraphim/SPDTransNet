<h1 style="text-align: center;">From Signals To SPD Matrices To Tokens*</h1>

*Work in progress.*

This file regroups explanations for all steps of the transformation of our data, up to the first learned model
component.
This touches upon multiple different modules within this repository, notably those found within the `_3_2_data_modules`
[directory](../../_3_data_management/_3_2_data_modules) in the [SPD processing](#dataset_processing) section
below.  
The main presentation of these components can be found in [the next documentation file](./3%20-%20Formatting%20The%20Model%20Inputs.md).

**Important**: this repository generates many `.pkl` files, all composed of nested lists and dictionaries, making them
difficult to manually inspect. To remedy this, we have included  the `nested_dicts_and_lists_exploration.py`
[standalone Python script](../standalone_tests/nested_dicts_and_lists_exploration.py), which pretty-prints the storage
architecture of the considered files.

<h2 style="text-align: center;">Dataset Extraction</h2>

This project was thought out to be dataset agnostic. Given an original dataset's files being copied into the
`_2_1_original_datasets` [folder](../../_2_data_preprocessing/_2_1_original_datasets), the corresponding
[extraction script](../../_2_data_preprocessing/_2_2_data_extraction/_extraction_scripts) should generate a
"standardized" set of `.pkl` files in a dedicated folder within the `2_2_data_extraction`
[directory](../../_2_data_preprocessing/_2_2_data_extraction), each corresponding to a single recording.  
The expected structure of such a file is available [here](./extras/MASS_SS3_extracted%20-%20pretty-print%20of%20the%20structure%20of%20the%20recording%200001%20pkl%20file.txt),
obtained by applying the `nested_dicts_and_lists_exploration.py` script (above) on the `MASS_SS3_extracted/0001.pkl`
[generated file](../../_2_data_preprocessing/_2_2_data_extraction/MASS_SS3_extracted/0001.pkl).

The `.saved_keys.txt` [file](../../_2_data_preprocessing/_2_2_data_extraction/MASS_SS3_extracted/.saved_keys.txt) will
also be generated, containing the keys common to every `.pkl` file in the extracted dataset folder. It is not used
elsewhere in the project.

If converted to this general structure, any dataset may be utilized with this project. In particular, any MASS subset
should be convertible by creating the appropriate [configuration file](../../_1_configs/_1_z_miscellaneous/dataset_extraction),
with minimal modifications to the 
[extraction script](../../_2_data_preprocessing/_2_2_data_extraction/_extraction_scripts/MASS_extraction.py).

<h2 style="text-align: center;">Preprocessors And Data Readers</h2>

The `Preprocessor` [classes](../../_2_data_preprocessing/_2_3_preprocessors) are designed to apply a number of
transformations onto the raw data, and to save all epoch-wise data in separate `.pkl` files in a directory within the
generated `_2_4_preprocessed_data` [directory](../../_2_data_preprocessing/_2_4_preprocessed_data).  
To allow for more flexibility, a `Preprocessor` class might be configured to generate multiple versions of the
preprocessed data, allowing for the selection of the wanted version through hyperparameters during execution. The
`DataReader` [classes](../../_3_data_management/_3_1_data_readers) perform this task, navigating the nested lists and
dictionaries to obtain the wanted data.

For preprocessing EEG signals into covariance matrices, the `SPDFromEEGPreprocessor`
[class](../../_2_data_preprocessing/_2_3_preprocessors/SPD_matrices_from_EEG_signals/SPDFromEEGPreprocessor.py)
is used, along with its corresponding `SPDFromEEGDataReader`
[class](../../_3_data_management/_3_1_data_readers/SPD_matrices_from_EEG_signals/SPDFromEEGDataReader.py).
The arguments used to instantiate the `SPDFromEEGPreprocessor` class are detailed within the
`parse_initialization_arguments` method, located
[here](../../_2_data_preprocessing/_2_3_preprocessors/SPD_matrices_from_EEG_signals/SPDFromEEGPreprocessor.py).

The data structure within the generated `.pkl` being somewhat complex, it is advised that you generate them and analyze
them through the `nested_dicts_and_lists_exploration.py` script (above) before modifying or creating a `Preprocessor`
and/or `DataReader` class.

<h2 if="preprocessing" style="text-align: center;">Preprocessing Pipeline*</h2>

The `SPDFromEEGPreprocessor` class first applies transformations to the original extracted signals. It then subdivides
them into epoch subwindows, and computes covariance matrices (and potentially statistic vectors, for augmentation
purposes - see [here](#augmentation)) for each subwindow.
Finally, it may also compute recording-wise matrices (and vectors) for whitening purposes (see 
[here](#whitening)).

It is designed to potentially compute and save multiple configurations in `.pkl` files (see ), so that the wanted configuration may be
requested on-the-fly during hyperparameter researches.
In this section, "default" configuration refers to the one applied to the data used to validate our model in the paper.  
By default, we subdivide each epoch into 30 1s subwindows, computing 30 SPD matrices per epoch channel.
As we use 8 EEG signals, our covariance matrices are of size 8 $\times$ 8.

Signal preprocessing pipeline:
- Z-score normalization (optional).
  - Default: applied.
- Channel-wise transformations (optional) - so far, only bandpass filtering with a 4th order Butterworth filter is
implemented.
  - Default: one channel unfiltered, 6 filtered: bands $\delta$ (0.5 to 4 Hz), $\theta$ (4 to 8 Hz),
  $\alpha$ (8 to 13 Hz), low $\beta$ (13 to 22 Hz), high $\beta$ (22 to 30 Hz) and $\gamma$ (30 to 45 Hz).

Epoch-wise preprocessing pipeline (as applied to each epoch subwindow):
- Covariance matrix estimation (multiple methods).
  - Default: standard estimator, through the `numpy.cov` function.
- Computation of statistic vectors (optional) - the vector containing a statistic computed for each signal individually
for the length of the subwindow, used for matrix augmentation.
  - Default: the average Power Spectral Density over the 1s subwindow (proportional to the signal variance found in
  the matrix diagonal, but empirically gives us the best results).

Recording-wise preprocessing pipeline (as applied to each full recording, i.e. each subject for MASS-SS3):
- Recording-wise matrix computation, for each channel (optional) - necessary for whitening.
  - Recording-wise covariance matrix: covariance matrix estimated over the entire recording (optional) - equivalent to
  the Euclidean mean of the corresponding subwindow-derived matrices.
  - Affine invariant estimated mean of the corresponding subwindow-derived matrices (optional) - through the
  `pyriemann.utils.mean.mean_riemann` function.
  - Default: the affine invariant estimated mean, yielding better performance.
- Computation of recording-wise statistic vectors (if applying both augmentation and whitening) - Euclidean mean of the
subwindow-derived statistic vectors.
  - Default: average of average PSDs.
- Computation of variance-only versions of the above recording-wise matrices (optional) - to study the importance of 
the covariance information, cf. our previous work.
  - Default: unused in the latest version of the model.

*Note: we verify during preprocessing that all considered matrices are indeed properly SPD, and not just Positive
Semi-Definite.*

<h2 id="dataset_processing" style="text-align: center;">On-The-Fly SPD Matrix Processing*</h2>

As stated in the beginning of this file, this section refers to the logic encoded within the `_3_2_data_modules`
[directory](../../_3_data_management/_3_2_data_modules).  
These operations were done within the model itself, but the corresponding PyTorch NN
[layers](../../_3_data_management/_3_2_data_modules/SPD_matrices_from_EEG_signals/layers) have been shifted to the data
management portion of the architecture, running only during the network initialization.  
As said layers don't currently contain trainable network parameters, this doesn't cause issue.
The implementation of this processing is further documented [here](./3%20-%20Formatting%20The%20Model%20Inputs.md).

In this section, we call $SPD(n)$ the set of $n \times n$ SPD matrices.

<h3 id="spd_processing" style="text-align: center;">SPD-To-SPD Processing*</h3>

Both the following operations are optional.  
As seen in [the Preprocessing Pipeline section above](#preprocessing), by
default, both are applied sequentially to our 1s subwindow-derived matrices, with:
- the PSD vector as statistic matrix ($k$ = 1),
- the recording-wise Affine invariant mean matrix as whitening matrix.

<h4 id="augmentation" style="text-align: center;">SPD Matrix Augmentation*</h4>

Given a covariance matrix $C \in SPD(n)$ and any matrix $V_{\alpha} \in \mathbb{R}^{n \times k}$, $k \geq 1$, we can
compute the corresponding augmented matrix $A$:
\[A = \left(\begin{array}{c|c}
    \\
    C+V_{\alpha} \cdot V_{\alpha}^T&V_{\alpha}\\
    \\
    \hline
    V_{\alpha}^T&I_k\\
    \end{array}\right)
\in SPD(n+k) \]

In our work, we set $V_{\alpha} = V \times \alpha$, with $\alpha \in \mathbb{R}$ being the augmentation factor
controlling for the prominence of a given augmentation matrix $V$ within our augmented matrix.  
This augmentation factor is a model hyperparameter (cf. [here](./3%20-%20Formatting%20The%20Model%20Inputs.md)).

<h4 id="whitening" style="text-align: center;">SPD Matrix Whitening*</h4>

<h3 id="tokenization" style="text-align: center;">Tokenization*</h3>

[//]: # (Cutoff)






