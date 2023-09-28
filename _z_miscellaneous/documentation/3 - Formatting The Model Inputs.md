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

<h2 id="sequences" style="text-align: center;">Sequence Formation</h2>

As stated in the paper, like much of the state-of-the-art models, our SPDTransNet model takes sequences of epochs as
input.  
These sequences are built from a central epoch to classify, flanked on either side by $\ell$ context epochs, for a total
sequence length $L = 2 \cdot \ell + 1$.  
One may think of these sequences in two equivalent ways:
- For each recording, we create a maximum number of sequences of odd length $L$ with maximum overlap, classifying only
said sequences' central epoch.
- For each epoch to classify (i.e. every epoch except the first and last $\ell$ epochs of each recording), we create a
sequence of $L$ epochs in total, with the epoch to classify in the central position.

Any mention of "classified epochs" in this file is a shorthand way of referring to these sequences with a single epoch
being classified by the model.

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

The MASS-SS3 dataset, like all EEG sleep scoring datasets we encountered, is highly imbalanced, with the post-extraction
distribution of epochs being as follows:

| **Total** | N3   | N2    | N1   | REM   | Awake |
|:---------:|:----:|:-----:|:----:|:-----:|:-----:|
| 59317     | 7653 | 29802 | 4839 | 10581 | 6442  |

with 50.24% of epochs in the N2 sleep stage and only 8.16% in the N1 sleep stage.

To ensure that the model is shown equal amounts of each class during training (i.e. rebalancing), we randomly oversample
classified epochs belonging to smaller classes, making sure that every class has as much classified epoch as the N2
sleep stage post-rebalancing.

This is only done during training to essentially counteract the implicit bias brought forth against smaller classes
during training. Applying the same operation to the validation and test sets isn't justifiable with thes argument, and
is not done in this repository.

<h3 id="trimming" style="text-align: center;">Test Set Trimming</h3>

In the paper, we compare ourselves to three sequence-based models, each with their own border effects.  
As such, even with the same folds, the test sets differ somewhat:
- We do not classify the first and last $\ell$ epochs (cf. [here](#sequences)) of each recording, with $\ell$ = 6, 10 or
14 in the paper.
- [DeepSleepNet](https://github.com/akaraspt/deepsleepnet) cuts the test set recordings into non-overlapping sequences,
discarding the remaining epochs if any; hence, 0 to 24 epochs are not classified for each recording's end.
- [IITNet](https://github.com/gist-ailab/IITNet-official) only uses previous epochs as context, with a sequence length
of 10 - i.e. the first 9 epochs of each recording are not classified.
- [GraphSleepNet](https://github.com/ziyujia/GraphSleepNet) defines its sequences like we do, with $\ell$ = 2.

As such, we have elected to ensure that the first and last 24 epochs are not classified for all models, but only for the
test sets, as only test set results are used for inter-model comparisons.  
For our model, assuming $\ell$ = 10, we further trim the first and last 14 epochs in our test set recordings.

<h2 style="text-align: center;">Model Input Batches</h2>

After tokenization, the `VectorizedSPDFromEEGDataModule` passes to the model tensors of the following shape:  
$(N, C, L, S, d(m))$  
with:
- $N$ the batch size,
- $C$ the number of channels (7 in the paper),
- $L$ the length of the sequence of epochs (6, 10 or 14 in the paper),
- $S$ the number of tokens per epoch (30 in the paper),
- $d(m)$ the length of SPD matrix-derived token sizes, with $m$ the matrix size (see
[here](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md#tokenization)).

<h3 id="data-duplication" style="text-align: center;">Data duplication</h3>

Not counting [border effects](#sequences) and [trimmings](#trimming), each epoch appears 21 times in the dataset,
leading to large amounts of data duplication.  
With the largest class in MASS-SS3 comprising roughly 50% of the dataset, the [training set rebalancing](#rebalancing)
engenders a further duplication of around 2.5 times (assuming a homogeneous repartition of sleep stages in each fold).

As such, an important amount of data duplication is created - totalling around 2100% of the original size for validation
and test sets, up to roughly  5250% for training sets.  
This needs to be taken into account when considering batch sizes (affected by sequence duplication) and the number of
batches by training cycle (affected by the rebalancing).  
(By training cycle, I mean training epoch, but as thr term "epoch" usually refers to EEG segments in this repository,
this might be a source of confusion...)

<h2 id="tmp_storage" style="text-align: center;">Initialization and Token Temporary Storage</h2>

Even without considering the dataset duplications mentioned above, the quantity of data in our dataset makes memory
storage impractical. As such, we have found it necessary to store the preprocessed inputs on hard drive.

The most time-consuming element of running the model is the SVD computation done as part of
[the tokenization process](./2%20-%20From%20Signals%20To%20SPD%20Matrices%20To%20Tokens.md#tokenization),
followed by on-the-fly loading of the aforementioned preprocessed inputs from stored files. Both operations are more
costly than all other operating computations by orders of magnitude.

The first effect is exacerbated by the data duplication issues [mentioned above](#data-duplication), but can be
mitigated by being done in advance. This cannot be done in advance, as at least the `augmentation_factor` hyperparameter
tends to change between runs, requiring a dedicated tokenization computation per run.  
However, as no trained parameters are utilized for this tokenization, it can be done once during initialization, saving
time overall.

These computed tokens must be temporarily stored on disk, which can be done in one of two ways:
- Separate files per epoch. This is the faster option overall by about an order of magnitude, but generates a large
number of files (more than 50000 per training runs, corresponding to 60 recordings' worth). On machines with a limited
number of inodes or a high minimum file size, this can be problematic.
- One single `.zip` file, managed using the `zarr` library. Even with no compression, this is way more time-consuming.

Note that tn both cases, this assumes that you will have enough disk space to essentially duplicate the MASS-SS3 dataset
once per run.  
In our case, disk space wasn't an issue, but we only had enough inodes allocated for 5 simultaneous runs, even in a
highly multi-GPU environment. Hence, we configured our hyperparameter researches to do 5 simultaneous time-efficient
runs, and ran our 31-fold cross-validations using 31 simultaneous single-file runs.

In both cases, the temporary files should be automatically destroyed by the end of the run, ***except*** if it ended in
an error stage. In that case, you may have to delete them manually.  
Please refer to [the relevant section in the main README file](../../README.md#reproducing_results) for more
information.


