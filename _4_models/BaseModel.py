import warnings
from os.path import isdir, join
from typing import Any, List, Dict, Union
import numpy as np
import pandas as pd
import seaborn
import torch
import torchmetrics
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from torch.nn import Module
from _4_models.utils import get_loss_function_from_dict, combine_dict_outputs
from pipeline_utils import import_class_from_path


# Defines the basic operations and metrics common to the pipeline


class BaseModel(LightningModule):

    TRAINING_SET_NAME = "training"
    VALIDATION_SET_NAME = "validation"
    TEST_SET_NAME = "test"
    USUAL_SET_NAMES_TUPLE = (TRAINING_SET_NAME, VALIDATION_SET_NAME, TEST_SET_NAME)
    PREDICTION = "prediction"

    DEFAULT_OPTIMIZATION_CONFIG_DICT = {
        "optimizer": {
            "class_path": "torch.optim.Adam",
            "extra_kwargs": {
                "betas": [0.9, 0.999],
                "eps": 1.e-7,
                "weight_decay": 1.e-5
            }
        },
        "scheduler": {
            "class_path": "torch.optim.lr_scheduler.ExponentialLR",
            "extra_kwargs": {
                "gamma": 0.94
            }
        }
    }

    def __init__(self, loss_function_config_dict: Dict[str, Any], class_labels_list: List[str],
                 learning_rate: float, optimisation_config_dict: Union[Dict[str, Any], None] = None):
        super(BaseModel, self).__init__()
        loss_function = get_loss_function_from_dict(loss_function_config_dict, self)
        self.loss = loss_function  # Typically from torch.nn.functional

        self.labels = class_labels_list
        self.number_of_classes = len(class_labels_list)

        self.learning_rate = learning_rate
        if optimisation_config_dict is not None:
            self.optimisation_config_dict = optimisation_config_dict
        else:
            self.optimisation_config_dict = self.DEFAULT_OPTIMIZATION_CONFIG_DICT

        self.train_standard_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_of_classes, average="micro")
        self.val_standard_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_of_classes, average="micro")
        self.test_standard_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_of_classes, average="micro")
        self.standard_accuracy = {
             self.TRAINING_SET_NAME: self.train_standard_accuracy,
             self.VALIDATION_SET_NAME: self.val_standard_accuracy,
             self.TEST_SET_NAME: self.test_standard_accuracy
        }

        self.train_macro_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_of_classes, average="macro")
        self.val_macro_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_of_classes, average="macro")
        self.test_macro_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.number_of_classes, average="macro")
        self.macro_accuracy = {
             self.TRAINING_SET_NAME: self.train_macro_accuracy,
             self.VALIDATION_SET_NAME: self.val_macro_accuracy,
             self.TEST_SET_NAME: self.test_macro_accuracy
        }

        self.train_kappa = torchmetrics.CohenKappa(task="multiclass", num_classes=self.number_of_classes)
        self.val_kappa = torchmetrics.CohenKappa(task="multiclass", num_classes=self.number_of_classes)
        self.test_kappa = torchmetrics.CohenKappa(task="multiclass", num_classes=self.number_of_classes)
        self.kappa = {
             self.TRAINING_SET_NAME: self.train_kappa,
             self.VALIDATION_SET_NAME: self.val_kappa,
             self.TEST_SET_NAME: self.test_kappa
        }

        self.train_MF1 = torchmetrics.F1Score(task="multiclass", num_classes=self.number_of_classes, average="macro")
        self.val_MF1 = torchmetrics.F1Score(task="multiclass", num_classes=self.number_of_classes, average="macro")
        self.test_MF1 = torchmetrics.F1Score(task="multiclass", num_classes=self.number_of_classes, average="macro")
        self.MF1 = {
             self.TRAINING_SET_NAME: self.train_MF1,
             self.VALIDATION_SET_NAME: self.val_MF1,
             self.TEST_SET_NAME: self.test_MF1
        }

        self.train_F1_scores = torchmetrics.F1Score(task="multiclass", num_classes=self.number_of_classes, average="none")
        self.val_F1_scores = torchmetrics.F1Score(task="multiclass", num_classes=self.number_of_classes, average="none")
        self.test_F1_scores = torchmetrics.F1Score(task="multiclass", num_classes=self.number_of_classes, average="none")
        self.F1_scores = {
             self.TRAINING_SET_NAME: self.train_F1_scores,
             self.VALIDATION_SET_NAME: self.val_F1_scores,
             self.TEST_SET_NAME: self.test_F1_scores
        }

        self.val_normalized_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.number_of_classes, normalize="true")
        self.test_normalized_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.number_of_classes, normalize="true")
        self.normalized_confusion_matrix = {
             self.VALIDATION_SET_NAME: self.val_normalized_confusion_matrix,
             self.TEST_SET_NAME: self.test_normalized_confusion_matrix
        }

        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.number_of_classes, normalize="none")
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.number_of_classes, normalize="none")
        self.confusion_matrix = {
             self.VALIDATION_SET_NAME: self.val_confusion_matrix,
             self.TEST_SET_NAME: self.test_confusion_matrix
        }

        self.current_set = None
        self.on_step = False

        self.save_confusion_matrices_to_folder = None
        self.run_id = None

        if not type(self) == BaseModel:
            self.obtain_example_input_array()
        else:
            warnings.warn("This is a base instance of a model, that must be inherited.")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def preprocess_input(self, input, set_name):
        raise NotImplementedError

    def process_loss_function_inputs(self, predictions, groundtruth, set_name):
        return {"input": predictions, "target": groundtruth.long()}

    def general_step(self, batch, batch_idx, step_strategy):
        if step_strategy in self.USUAL_SET_NAMES_TUPLE:
            set_name = step_strategy
        elif step_strategy == self.PREDICTION:
            set_name = self.TEST_SET_NAME
        else:
            raise NotImplementedError

        self.current_set = set_name
        self.on_step = True

        x, y = batch

        inputs_as_dict = self.preprocess_input(x, set_name)  # In case forward() has multiple inputs
        logits = self(**inputs_as_dict)
        loss_function_inputs_as_dict = self.process_loss_function_inputs(logits, y, set_name)
        logits, y = loss_function_inputs_as_dict["input"], loss_function_inputs_as_dict["target"]

        step_loss = None
        
        if step_strategy != self.PREDICTION:
            self.step_level_update_metrics(logits, y, set_name)
            step_loss = self.loss(**loss_function_inputs_as_dict)
            self.log("on_step_loss/%s" % set_name, step_loss, on_step=True, on_epoch=False)
        
        return loss_function_inputs_as_dict, step_loss

    def training_step(self, batch, batch_idx):
        loss_function_inputs_as_dict, loss = self.general_step(batch, batch_idx, self.TRAINING_SET_NAME)
        return loss

    def training_epoch_end(self, outputs):
        self.epoch_level_log_metrics(outputs,  self.TRAINING_SET_NAME)

    # predicting: set to True if the function is called by predict_step (no metrics logged)
    def validation_step(self, batch, batch_idx, predicting=False):
        loss_function_inputs_as_dict, _ = self.general_step(batch, batch_idx, self.VALIDATION_SET_NAME)
        return loss_function_inputs_as_dict

    def validation_epoch_end(self, outputs):
        self.epoch_level_log_metrics(outputs,  self.VALIDATION_SET_NAME)

    # predicting: set to True if the function is called by predict_step (no metrics logged)
    def test_step(self, batch, batch_idx, predicting=False):
        loss_function_inputs_as_dict, _ = self.general_step(batch, batch_idx, self.TEST_SET_NAME)
        return loss_function_inputs_as_dict

    def test_epoch_end(self, outputs):
        self.epoch_level_log_metrics(outputs,  self.TEST_SET_NAME)

    # You can change the default prediction behavior of inheriting classes by setting overwrite_default_behavior to
    # another default value
    def predict_step(self, batch, batch_idx, dataloader_idx=0, overwrite_default_behavior=True):
        if not overwrite_default_behavior:
            return super(BaseModel, self).predict_step(batch, batch_idx, dataloader_idx)
        loss_function_inputs_as_dict, _ = self.general_step(batch, batch_idx, self.PREDICTION)
        return loss_function_inputs_as_dict

    def step_level_update_metrics(self, logits, y, set_name):

        self.standard_accuracy[set_name].update(logits, y.long())
        self.macro_accuracy[set_name].update(logits, y.long())
        self.MF1[set_name].update(logits, y.long())
        self.F1_scores[set_name].update(logits, y.long())
        self.kappa[set_name].update(logits, y.long())

        if set_name != self.TRAINING_SET_NAME:
            self.normalized_confusion_matrix[set_name].update(logits, y.long())
            self.confusion_matrix[set_name].update(logits, y.long())

    def epoch_level_log_metrics(self, outputs, set_name):

        self.current_set = set_name
        self.on_step = False

        epoch_level_standard_accuracy = self.standard_accuracy[set_name].compute().detach().cpu()
        epoch_level_macro_accuracy = self.macro_accuracy[set_name].compute().detach().cpu()

        self.log("standard_acc/%s" % set_name, epoch_level_standard_accuracy, sync_dist=True)
        self.log("macro_acc/%s" % set_name, epoch_level_macro_accuracy, sync_dist=True)

        epoch_level_MF1 = self.MF1[set_name].compute().detach().cpu()
        epoch_level_F1_scores = self.F1_scores[set_name].compute().detach().cpu()
        epoch_level_kappa = self.kappa[set_name].compute().detach().cpu()

        self.log("mf1/%s" % set_name, epoch_level_MF1, sync_dist=True)
        if set_name == self.VALIDATION_SET_NAME:
            self.log("hp_metric", epoch_level_MF1, sync_dist=True)
        self.log("kappa/%s" % set_name, epoch_level_kappa, sync_dist=True)

        for i in range(self.number_of_classes):
            self.log("f1_%s/%s" % (self.labels[i], set_name), epoch_level_F1_scores[i], sync_dist=True)

        # Equivalent to the true mean if all batches have the same number of elements
        # In reality, the last batch is ofter smaller, but the differences are minor
        if set_name == self.TRAINING_SET_NAME:
            list_loss = []
            for output in outputs:
                list_loss.append(output["loss"])
            epoch_level_loss = torch.mean(torch.FloatTensor(list_loss))
            self.log("loss/%s" % set_name, epoch_level_loss, sync_dist=True)

        else:
            loss_inputs_dict = combine_dict_outputs(outputs)

            epoch_level_loss = self.loss(**loss_inputs_dict)
            self.log("loss/%s" % set_name, epoch_level_loss, sync_dist=True)
            self.process_confusion_matrix(set_name)

        self.reset_metrics(set_name)

    def reset_metrics(self, set_name):
        self.standard_accuracy[set_name].reset()
        self.macro_accuracy[set_name].reset()
        self.MF1[set_name].reset()
        self.F1_scores[set_name].reset()
        self.kappa[set_name].reset()
        if set_name != self.TRAINING_SET_NAME:
            self.confusion_matrix[set_name].reset()
            self.normalized_confusion_matrix[set_name].reset()

    def save_computed_confusion_matrices(self, save_folder_full_path: str, run_id: Union[str, None] = None):
        assert isdir(save_folder_full_path)
        self.save_confusion_matrices_to_folder = save_folder_full_path
        self.run_id = run_id

    def process_confusion_matrix(self, set_name):
    
        normalized_confusion_matrix_tensor = self.normalized_confusion_matrix[set_name].compute().detach().cpu()
        normalized_confusion_matrix = pd.DataFrame(normalized_confusion_matrix_tensor.numpy(), index=self.labels, columns=self.labels)

        plt.figure(figsize=(12, 7))
        normalized_confusion_matrix_figure = seaborn.heatmap(normalized_confusion_matrix, annot=True,fmt=".3g").get_figure()
        plt.close("all")  # Might be useful, or not

        confusion_matrix_tensor = self.confusion_matrix[set_name].compute().detach().cpu()
        confusion_matrix = pd.DataFrame(confusion_matrix_tensor.numpy(), index=self.labels, columns=self.labels)

        plt.figure(figsize=(12, 7))
        confusion_matrix_figure = seaborn.heatmap(confusion_matrix, annot=True, fmt="d").get_figure()

        # Plots the sums wor each row / column
        plt.text(6, 0, "TOTAL")
        plt.text(-0.2, 5.6, "TOTAL")
        plt.text(6.0, 5.6, str(np.sum(confusion_matrix.sum(axis=1))))
        for i in range(5):
            plt.text(6, 0.6 + i, str(confusion_matrix.sum(axis=1)[i]))
            plt.text(0.4 + i, 5.6, str(confusion_matrix.sum(axis=0)[i]))
        plt.close("all")  # Might be useful, or not

        tensorboard = self.logger.experiment
        tensorboard.add_figure("normalized_confusion_matrix/%s" % set_name, normalized_confusion_matrix_figure, self.current_epoch)
        tensorboard.add_figure("confusion_matrix/%s" % set_name, confusion_matrix_figure, self.current_epoch)

        if self.save_confusion_matrices_to_folder is not None:

            run_string = ""
            if self.run_id is not None:
                run_string = "run_%s_" % self.run_id

            normalized_confusion_matrix_filename = join(
                self.save_confusion_matrices_to_folder,
                "%s%s_normalized_confusion_matrix_for_epoch_%d.png" % (run_string, set_name, self.current_epoch)
            )
            confusion_matrix_filename = join(
                self.save_confusion_matrices_to_folder,
                "%s%s_confusion_matrix_for_epoch_%d.png" % (run_string, set_name, self.current_epoch)
            )
            normalized_confusion_matrix_figure.savefig(normalized_confusion_matrix_filename)
            confusion_matrix_figure.savefig(confusion_matrix_filename)

    def configure_optimizers(self):
        return_dict = {}

        assert "optimizer" in self.optimisation_config_dict.keys() and "class_path" in self.optimisation_config_dict["optimizer"].keys()
        optimizer_class = import_class_from_path(self.optimisation_config_dict["optimizer"]["class_path"])
        optimizer_kwargs = {}
        if "extra_kwargs" in self.optimisation_config_dict["optimizer"].keys():
            optimizer_kwargs = self.optimisation_config_dict["optimizer"]["extra_kwargs"]

        optimizer = optimizer_class(self.parameters(), lr=self.learning_rate, **optimizer_kwargs)
        return_dict["optimizer"] = optimizer

        if "scheduler" in self.optimisation_config_dict.keys():
            assert "class_path" in self.optimisation_config_dict["scheduler"].keys()
            scheduler_dict = {}

            scheduler_class = import_class_from_path(self.optimisation_config_dict["scheduler"]["class_path"])
            scheduler_kwargs = {}
            if "extra_kwargs" in self.optimisation_config_dict["scheduler"].keys():
                scheduler_kwargs = self.optimisation_config_dict["scheduler"]["extra_kwargs"]

            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            scheduler_dict["scheduler"] = scheduler

            if "extra_config" in self.optimisation_config_dict["scheduler"].keys():
                for key, value in self.optimisation_config_dict["scheduler"]["extra_config"].items():
                    scheduler_dict[key] = value

            return_dict["lr_scheduler"] = scheduler_dict

        return return_dict

    def obtain_example_input_array(self):
        raise NotImplementedError

    @staticmethod
    def get_block_dict(block: Module):
        block_class_path = block.__module__ + "." + block.__class__.__name__
        block_dict = {"class_path": block_class_path}
        return block_dict