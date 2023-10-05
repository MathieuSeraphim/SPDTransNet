<h1 style="text-align: center;">Implemented Configuration Strategy</h1>

This repository is built to be highly modular and customizable, with class instantiations handled through `.yaml`
files located in the `_1_configs` [directory](../../_1_configs).  
As stated in [the main README file](../../README.md#caveats), this instantiation is done in a non-standard way, through
a workaround utilized in the various `Wrapper` classes throughout the repository.

**Important:** the below configurations might not be fully referenced in the main documentation file describing their
use case. This present file is intended to present this information in a centralized location.

<h2 style="text-align: center;">The Main Classes</h2>

The first 6 folders in the `_1_configs` [directory](../../_1_configs) correspond to major classes instantiated as stated
above.  
Some of these classes contain a "base" class, designed to be inherited by others. Some `.yaml` configuration files have
been included to instantiate said classes for testing purposes, and are preceded by an underscore
(e.g. `_sequence_to_classification_base_model_config.yaml` found [here](../../_1_configs/_1_5_models/_sequence_to_classification_base_model_config.yaml),
instantiating the `SequenceToClassificationBaseModel` [class](../../_4_models/_4_1_sequence_based_models/SequenceToClassificationBaseModel.py)).

Folder-by-folder, these `.yaml` files instantiate the following:
- the `_1_1_preprocessing` [folder](../../_1_configs/_1_1_preprocessing): the `Preprocessing` classes,
handling the transformation of the raw data into SPD matrices corresponding to different frequency bands, defined within
the `_2_data_preprocessing` [directory](../../_2_data_preprocessing) -
further documentation [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md);
- the `_1_2_data_reading` [folder](../../_1_configs/_1_2_data_reading): the `DataReader` classes, handling the parsing
of pre-processed data defined within the `_3_1_data_readers` [directory](../../_3_data_management/_3_1_data_readers) -
further documentation [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md);
- the `_1_3_datasets` [folder](../../_1_configs/_1_3_datasets): the classes inheriting the PyTorch `Dataset` class,
used by compatible `DataModule` classes and utilizing a `DataReader` class, defined within their `DataModule`'s
directory (see below) - further documentation [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md)
and [here](./3%20-%20Formatting%20The%20Model%20Inputs.md);
- the `_1_4_data_modules` [folder](../../_1_configs/_1_4_data_modules): the `DataModule` classes mentioned above,
initializing the `Dataset` objects and handling the batch creation logic, defined within the `_3_2_data_modules`
[directory](../../_3_data_management/_3_2_data_modules) - further documentation [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md)
and [here](./3%20-%20Formatting%20The%20Model%20Inputs.md);
- the `_1_5_models` [folder](../../_1_configs/_1_5_models): the `Model` classes defined within the `_4_models`
[directory](../../_4_models), and in particular the `_4_1_sequence_based_models` [sub-directory](../../_4_models/_4_1_sequence_based_models),
whose role is pretty self-explanatory - further documentation [here](./4%20-%20The%20SPDTransNet%20Model.md);
- the `_1_6_trainer` [folder](../../_1_configs/_1_6_trainer): the PyTorch Trainer class, used to define the training
loop and manage the utilized `Model` and `DataModule` classes - further documentation [here](./5%20-%20Running%20The%20Model.md).

<h3 style="text-align: center;">Important Caveats</h3>

Regarding the `_1_4_data_modules` and `_1_5_models` folders, keep in mind the following:
- The hyperparameters within their `.yaml` files are "default" values, and are not guaranteed to yield high performance.
- Any hyperparameters within them included in the hyperparameter research will override those during execution.
  - As such, when training our model with *specific* hyperparameters, those are usually stored in the
  `past_runs_hyperparameters` [folder](../../_1_configs/_1_z_miscellaneous/execution/past_runs_hyperparameters)
    (further documentation [here](./5%20-%20Running%20The%20Model.md)).
- Any hyperparameter within them *not* included in the hyperparameter research will take the value in the `.yaml` file.
  - As such, three non-base `.yaml` files are included in each folder, corresponding to different epoch sequence
  lengths ($L$ in the paper). For a given value of $L$, the corresponding `.yaml` files in each folder must be used.
- Any hyperparameter *not* within them *not* included in the hyperparameter research will take the default value defined
in the corresponding class' `__init__` constructor method. If no default value is present, the code will crash.

<h2 style="text-align: center;">Miscellaneous Configurations</h2>

The `_1_z_miscellaneous` [folder](../../_1_configs/_1_z_miscellaneous) stores all configuration files not used to
instantiate a single component class of the architecture.

Included are the following folders:
- the `dataset_extraction` [folder](../../_1_configs/_1_z_miscellaneous/dataset_extraction), whose `.yaml` files may be
used by Python scripts in the `_extraction_scripts` [folder](../../_2_data_preprocessing/_2_2_data_extraction/_extraction_scripts) -
further documentation [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md);
- the `channel_wise_transformations` [folder](../../_1_configs/_1_z_miscellaneous/channel_wise_transformations),
containing signal transformation instrictions for preprocessing in `.yaml` format, and
[the code to read them](../../_1_configs/_1_z_miscellaneous/channel_wise_transformations/utils.py) -
further documentation [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md);
- the `cross_validation_folds` [folder](../../_1_configs/_1_z_miscellaneous/cross_validation_folds), containing the
`.yaml` files that define MASS-SS3's 31 cross-validation folds - further documentation
[here](../../_1_configs/_1_z_miscellaneous/cross_validation_folds/MASS_SS3/folds_generation/Fold%20Specifics.md);
- the `execution` [folder](../../_1_configs/_1_z_miscellaneous/cross_validation_folds), also accessible through
[a symbolic link](../../_5_execution/_5_z_configs) in the `_5_execution` [directory](../../_5_execution) - it is
documented primarily [here](./5%20-%20Running%20The%20Model.md).