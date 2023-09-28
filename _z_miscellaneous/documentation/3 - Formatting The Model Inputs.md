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
The arguments used to instantiate the aforementioned `VectorizedSPDFromEEGDataModule` class are detailed within 
[the class' constructor](../../_3_data_management/_3_2_data_modules/SPD_matrices_from_EEG_signals/VectorizedSPDFromEEGDataModule.py).

**IMPORTANT:** as alluded to [here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md#dataset_processing),
the augmentation, whitening and tokenization are handled by the `StandardVectorizedSPDFromEEGDataset` class, which
generates temporary files to store these tokens during execution. This is expanded upon in [the relevant section below](#tmp_storage).

<h2 style="text-align: center;">Training, Validation and Test Sets</h2>

Our supervised classification model's training and evaluation requires a division of our dataset into training,
validation and test sets, as we operate both hyperparameter research and checkpointing / early stopping strategies.
More information on such divisions can be found in
[this Wikipedia article](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets).  
As the training, validation and test set need to be decoupled, epochs from one recording cannot be divided into multiple
sets.

We validate our model on MASS-SS3 through 31-fold
[cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), the specifics of which are explained
[here](../../_1_configs/_1_z_miscellaneous/cross_validation_folds/MASS_SS3/folds_generation/Fold%20Specifics.md).  
Essentially, our folds define a 50/10/2 split of our 62 subjects/recordings, corresponding to the training, validation
and test sets respectively - with the union of all test sets corresponding to the entire dataset, without overlap.  
Note that as the number of epoch per recording varies, collating the predictions from all 31 sets and then computing
performance statistics is ***not*** equivalent to doing so on each set and averaging the results. In this repository, we
only do the latter, with our reasoning explained [here](https://doi.org/10.1007/978-3-031-44240-7_7) (Section 4.3).

<h3 id="rebalancing" style="text-align: center;">Training Set Rebalancing</h3>

<h3 style="text-align: center;">Test Set Trimming</h3>

<h2 style="text-align: center;">Model Input Batches</h2>

<h3 style="text-align: center;">Data duplication</h3>

Not counting border effects and trimmings, each epoch appears 21 times in the dataset, leading to large amounts of data
duplication.  
With the largest class in MASS-SS3 (corresponding to the N2 sleep stage) comprising roughly 56% of the dataset, the
[training set rebalancing](#rebalancing) engenders a further duplication of around 2.8 times (assuming a homogeneous
repartition of sleep stages in each fold).

As such, an important amount of data duplication is created - around 2100% for validation and test sets, up to roughly
5880% for training sets.  
This needs to be taken into account when considering batch sizes (affected by sequence duplication) and the number of
batches by training cycle (affected by the rebalancing).  
(By training cycle, I mean training epoch, but as thr term "epoch" usually refers to EEG segments in this repository,
this might be a source of confusion...)

<h2 id="tmp_storage" style="text-align: center;">Initialization and Token Temporary Storage</h2>

Even without considering the dataset duplications mentioned above, the quantity of data in our dataset makes memory
storage impractical. As such, we have found it necessary to store the preprocessed inputs on hard drive.

[//]: # (SVD time)

