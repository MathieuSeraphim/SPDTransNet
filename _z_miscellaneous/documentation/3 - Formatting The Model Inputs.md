<h1 style="text-align: center;">Formatting The Model Inputs</h1>

*Work in progress. This documentation should be completed by the end of September 2023. In the meantime,
do not hesitate to [contact us](mailto:mathieu.seraphim@unicaen.fr) for more information.*

In [PyTorch](https://pytorch.org/docs/1.11/), `Dataset` classes contain the logic necessary to load samples from a given
dataset one item at a time, and are passed as argument to a `DataLoader` class to form batches.
Typically, different subsets of a given dataset (i.e. the training, test, and sometimes validation sets) are each
handled by a distinct `Dataset` + `DataLoader` pair.  
In [PyTorch Lightning](https://lightning.ai/docs/pytorch/1.9.5/), the initialization of `Dataset` and  `DataLoader`
classes is handled by a single `DataModule` class.  

Within this repository, the `_3_2_data_modules` [directory](../../_3_data_management/_3_2_data_modules) contains our
implementation of said classes. In this file, we shall go over the technical aspects of their implementation.  
In particular, it shall focus on the utilized `VectorizedSPDFromEEGDataModule`
[class](../../_3_data_management/_3_2_data_modules/SPD_matrices_from_EEG_signals/VectorizedSPDFromEEGDataModule.py),
and its corresponding `StandardVectorizedSPDFromEEGDataset`
[class](../../_3_data_management/_3_2_data_modules/SPD_matrices_from_EEG_signals/datasets/StandardVectorizedSPDFromEEGDataset.py).

**IMPORTANT:** as alluded to [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md#dataset_processing),
the augmentation, whitening and tokenization are handled by the `StandardVectorizedSPDFromEEGDataset` class, which
generates temporary files to store these tokens during execution. This is expanded upon in [the relevant section](#tmp_storage).

<h2 style="text-align: center;">Training, Validation and Test Sets</h2>

<h3 style="text-align: center;">Training Set Rebalancing</h3>

<h3 style="text-align: center;">Test Set Trimming</h3>

<h2 style="text-align: center;">Model Input Batches</h2>

<h2 id="tmp_storage" style="text-align: center;">Initialization and Token Temporary Storage</h2>

[//]: # (SVD time)

