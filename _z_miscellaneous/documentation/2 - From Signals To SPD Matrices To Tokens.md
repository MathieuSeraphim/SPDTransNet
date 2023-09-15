<h1 style="text-align: center;">From Signals To SPD Matrices To Tokens*</h1>

*Work in progress.*

This file regroups explanations for all steps of the transformation of our data, up to the first learned model
component.
This touches upon multiple different modules within this repository, notably those found within the `_3_2_data_modules`
[directory](../../_3_data_management/_3_2_data_modules) in the [last](#spd_processing) [two](#tokenization) sections
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

<h2 style="text-align: center;">Preprocessing Pipeline*</h2>

The `SPDFromEEGPreprocessor` class first applies transformations to the original extracted signals. It then subdivides
them into epoch subwindows, and computes covariance matrices (and potentially statistic vectors, for augmentation
purposes - see [next Section](#spd_processing)) for each subwindow.
Finally, it may also compute recording-wise matrices (and vectors) for whitening purposes (see 
[next Section](#spd_processing)).

It is designed to potentially compute and save multiple configurations in `.pkl` files (see ), so that the wanted configuration may be
requested on-the-fly during hyperparameter researches.

Signal transformations

<h2 id="spd_processing" style="text-align: center;">SPD Data Processing*</h2>

[//]: # (Augmentation, whitening)

<h2 id="tokenization" style="text-align: center;">Tokenization*</h2>

[//]: # (Cutoff)






