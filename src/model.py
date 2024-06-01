"""Class for the segmentation model."""

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose


class SegmentationModel:
    def __init__(self, device, in_channels=4, out_channels=3, features=(8, 16, 32)):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.model = self.create_model()
        self.loss_function = self.create_loss_function()
        self.dice_metric, self.dice_metric_batch, self.post_trans = self.create_metrics()

    def create_model(self):
        model = UNet(
            spatial_dims=3,  # 3D image
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=self.features,
            strides=(2, 2),
            num_res_units=2,
            act="PRELU",
            norm="INSTANCE",
            dropout=0,
            bias=True,
            adn_ordering="NDA",
        )
        return model.to(self.device)

    def create_loss_function(self):
        return DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True
        )

    def create_metrics(self):
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        return dice_metric, dice_metric_batch, post_trans

    def get_model(self):
        return self.model

    def get_loss_function(self):
        return self.loss_function

    def get_metrics(self):
        return self.dice_metric, self.dice_metric_batch, self.post_trans
