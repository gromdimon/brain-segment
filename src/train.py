"""
This file is used to train the segmentation model.
"""

import os
import time

import torch
from monai.data import decollate_batch
from pydantic import BaseModel

from src.model import SegmentationModel
from src.utils import load_data


class Config(BaseModel):
    max_epochs: int = 30
    val_interval: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_segmentation_model(config: Config):
    device = torch.device(config.device)
    train_loader, train_ds = load_data(train=True)
    val_loader, val_ds = load_data(train=False)

    seg_model = SegmentationModel(device)
    model = seg_model.get_model()
    loss_function = seg_model.get_loss_function()
    optimizer = torch.optim.Adam(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    dice_metric, dice_metric_batch, post_trans = seg_model.get_metrics()

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()
    for epoch in range(config.max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config.max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            if step > 10:
                continue
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            if device.type == "cuda" and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # if (epoch + 1) % config.val_interval == 0:
        #     evaluate_model(
        #         epoch,
        #         model,
        #         val_loader,
        #         dice_metric,
        #         dice_metric_batch,
        #         post_trans,
        #         device,
        #         best_metrics_epochs_and_time,
        #         metric_values,
        #         metric_values_tc,
        #         metric_values_wt,
        #         metric_values_et,
        #         best_metric,
        #         best_metric_epoch,
        #         total_start,
        #     )

    total_time = time.time() - total_start
    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}."
    )


def evaluate_model(
    epoch,
    model,
    val_loader,
    dice_metric,
    dice_metric_batch,
    post_trans,
    device,
    best_metrics_epochs_and_time,
    metric_values,
    metric_values_tc,
    metric_values_wt,
    metric_values_et,
    best_metric,
    best_metric_epoch,
    total_start,
):
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = model(
                val_inputs
            )
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric = dice_metric.aggregate().item()
        metric_values.append(metric)
        metric_batch = dice_metric_batch.aggregate()
        metric_tc = metric_batch[0].item()
        metric_values_tc.append(metric_tc)
        metric_wt = metric_batch[1].item()
        metric_values_wt.append(metric_wt)
        metric_et = metric_batch[2].item()
        metric_values_et.append(metric_et)
        dice_metric.reset()
        dice_metric_batch.reset()

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            best_metrics_epochs_and_time[0].append(best_metric)
            best_metrics_epochs_and_time[1].append(best_metric_epoch)
            best_metrics_epochs_and_time[2].append(time.time() - total_start)
            torch.save(
                model.state_dict(),
                os.path.join("models", "best_metric_model.pth"),
            )
            print("saved new best metric model")
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
            f"\nbest mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )


if __name__ == "__main__":
    config = Config()
    train_segmentation_model(config)
