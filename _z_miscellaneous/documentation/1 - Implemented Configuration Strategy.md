<h1 style="text-align: center;">Implemented Configuration Strategy</h1>

*Work in progress.*

This repository is built to be highly modular and customizable, with class instantiations handled through `.yaml`
files located in the `_1_configs` [directory](../../_1_configs).  
As stated in [the main README file](../../README.md#caveats), this instantiation is done in a non-standard way, through
a workaraund utilized in the various `Wrapper` classes throughout the repository.

<h2 style="text-align: center;">The Main Classes</h2>

The first 6 folders in the `_1_configs` [directory](../../_1_configs) correspond to major classes instantiated as stated
above.  
Some of these classes contain a "base" class, designed to be inherited by others. Some `.yaml` configuration files have
been included to instantiate said classes for testing purposes, and are preceded by an underscore
(e.g. `_sequence_to_classification_base_model_config.yaml` found [here](../../_1_configs/_1_5_models/_sequence_to_classification_base_model_config.yaml),
instantiating the `SequenceToClassificationBaseModel` [class](../../_4_models/_4_1_sequence_based_models/SequenceToClassificationBaseModel.py)).

Folder-by-folder, these `.yaml` files instantiate the following:
- the `_1_1_preprocessing` [folder](../../_1_configs/_1_1_preprocessing): the classes handling the transformation of
the raw data into SPD matrices corresponding to different frequency bands, within the `_2_data_preprocessing`
[directory](../../_2_data_preprocessing) - further documentation [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md);
- the `_1_2_data_reading` [folder](../../_1_configs/_1_2_data_reading): the classes handling the parsing of
pre-processed data, within the `_3_1_data_readers` [directory](../../_3_data_management/_3_1_data_readers) -
further documentation [here](./3%20-%20Formatting%20The%20Model%20Inputs.md);
- the [folder](../../_1_configs/): the classes within the [directory](../../) - further documentation [here](./);
- the [folder](../../_1_configs/): the classes within the [directory](../../) - further documentation [here](./);
- the [folder](../../_1_configs/): the classes within the [directory](../../) - further documentation [here](./);
- the [folder](../../_1_configs/): the classes within the [directory](../../) - further documentation [here](./).