import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt


def GE2G_Loss(score_matrix, eps=1e-9):
    count_rec, n_speakers = score_matrix.shape
    n_blocks = count_rec // n_speakers

    loss = 0

    for idx in range(n_speakers):
        for jdx in range(n_blocks):
            speakers_res = score_matrix[idx * n_blocks + jdx]

            loss -= speakers_res[idx] - torch.log(torch.sum(torch.exp(speakers_res)) + eps)

    return loss


def get_predictions(score_matrix, threshold):
    return score_matrix >= threshold


def get_ground_truth(predictions):
    ground_truth = torch.zeros_like(predictions, dtype=bool)
    n_speakers = predictions.shape[1]
    samples_per_speaker = predictions.shape[0] // n_speakers

    for i in range(n_speakers):
        ground_truth[i * samples_per_speaker: (i + 1) * samples_per_speaker, i] = True

    return ground_truth


def get_far(ground_truth, predictions):
    """
    FAR: FP / (FP + TN)
    """
    fp = float(torch.sum(predictions & (~ground_truth)))
    tn = float(torch.sum((~predictions) & (~ground_truth)))
    far = fp / (fp + tn) if fp != 0 else 0
    return far


def get_frr(ground_truth, predictions):
    """
    FRR: FN / (FN + TP)
    """
    fn = float(torch.sum((~predictions) & ground_truth))
    tp = float(torch.sum(predictions & ground_truth))
    frr = fn / (fn + tp) if fn != 0 else 0
    return frr


def get_metrics(score_matrix, threshold):
    preds = get_predictions(score_matrix, threshold)
    ground_truth = get_ground_truth(preds)

    far = get_far(ground_truth, preds)
    frr = get_frr(ground_truth, preds)

    return far, frr


def get_eer(far, frr):
    maximum = np.maximum(far, frr)
    eer = np.min(maximum)
    eer_threshold_index = (np.argmin(maximum) + (len(maximum) - np.argmin(maximum))[::-1]) // 2
    return eer, eer_threshold_index


def smooth(array, window_size):
    return [np.mean(array[max(0, i - window_size): i + 1]) for i in range(len(array))]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.conv_block(x)


class ConvEncoder(nn.Module):
    def __init__(self,
                 in_time,
                 in_features,
                 in_channels=1,
                 embedding_size=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LayerNorm([in_channels, in_time, in_features]),
            nn.LeakyReLU(inplace=True),
            ConvBlock(in_channels=in_channels, out_channels=64),
            nn.Dropout2d(p=0.2, inplace=True),
            ConvBlock(in_channels=64, out_channels=128),
            nn.Dropout2d(p=0.2, inplace=True),
            ConvBlock(in_channels=128, out_channels=256),
            nn.Dropout2d(p=0.2, inplace=True),
            ConvBlock(in_channels=256, out_channels=512),
            nn.Dropout2d(p=0.2, inplace=True),
            ConvBlock(in_channels=512, out_channels=1024),
            nn.Dropout2d(p=0.2, inplace=True),
            ConvBlock(in_channels=1024, out_channels=2048),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * (in_time // 64) * (in_features // 64), embedding_size)
        )

    def forward(self, x):
        x = self.conv(x)

        return self.dense(x)


class Discriminator(nn.Module):
    def __init__(self,
                 input_size=256,
                 d_size=200,
                 hidden_size=256,
                 negative_slope=0.2,
                 use_cos=True,
                 use_net=True):
        super().__init__()
        assert use_cos or use_net

        self.cos_disc = nn.CosineSimilarity(dim=1)
        self.d_size = d_size

        self.net_disc = nn.Sequential(
            nn.Linear(2 * input_size, hidden_size),
            nn.LeakyReLU(negative_slope),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope),

            nn.Linear(hidden_size, 1)
        )

        self.use_cos = use_cos
        self.use_net = use_net
        self.score_combinator = nn.Linear(2, 1)

    def forward(self, X, specimen_X):
        if self.use_cos:
            cos_output = self.cos_disc(X[:, :self.d_size], specimen_X[:, :self.d_size]) \
                [..., None]

        if self.use_net:
            specimen_X_rep = specimen_X.repeat(X.shape[0], 1)

            net_output = self.net_disc(torch.cat([X, specimen_X_rep], dim=1))

        if self.use_cos and self.use_net:
            output = torch.cat([cos_output, net_output], dim=1)
            return self.score_combinator(output)
        elif self.use_cos:
            return cos_output
        elif self.use_net:
            return net_output


class BiometryModel(pl.LightningModule):
    def __init__(self, encoder, discriminator):
        super().__init__()

        self._encoder = encoder
        self._discriminator = discriminator
        self._loss = GE2G_Loss
        self._specimen_d_vector = None

        self.train_loss_history = []
        self.val_loss_history = []

        # parameters for EER
        self.n_points = 501
        self.a = -25
        self.b = 25
        self.train_eer_history = []
        self.val_eer_history = []
        self.window_size = 5

        self.threshold = 0

        self.train_threshold_index_history = []
        self.last_train_outputs = None

    def set_specimen_d_vector(self, x):
        encod_sounds = self._encoder(x)
        self._specimen_d_vector = torch.mean(encod_sounds, dim=0, keepdims=True)

    def get_specimen_d_vector(self):
        return self._specimen_d_vector

    def forward(self, x):
        assert self._specimen_d_vector != None, "The voice sample vector is not set"

        encod_sounds = self._encoder(x)

        return self._discriminator(encod_sounds, self._specimen_d_vector)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _step(self, batch, step_type):
        assert step_type in {"train", "val", "treshold"}

        X_batch, specimen_X_batch, _ = batch

        # loss calculation
        loss = 0
        score_blocks = []

        n_speakers, n_records_per_speaker, n_channels, n_times, n_features = X_batch.shape
        n_specimens = specimen_X_batch.shape[1]

        specimen_X = specimen_X_batch.view(n_speakers * n_specimens, n_channels, n_times, n_features)
        encod_specimen_X = self._encoder(specimen_X).view(n_speakers, n_specimens, -1)

        X = X_batch.view(n_speakers * n_records_per_speaker, n_channels, n_times, n_features)
        encod_X = self._encoder(X)

        list_speakers_dist = []

        for specimen_x in encod_specimen_X:
            specimen_x = torch.mean(specimen_x, dim=0, keepdims=True)
            list_speakers_dist.append(self._discriminator(encod_X, specimen_x))

        speakers_dist = torch.cat(list_speakers_dist, dim=1)

        if step_type == "treshold":
            threshold_grid = np.linspace(self.a, self.b, num=self.n_points)

            metrics = [
                (threshold, *get_metrics(score_matrix=speakers_dist, threshold=threshold))
                for threshold in threshold_grid
            ]
            return {"treshold_metrics": metrics}
        else:
            loss = self._loss(speakers_dist)
            self.log(f"{step_type}_loss", loss)

            threshold_grid = np.linspace(self.a, self.b, num=self.n_points)

            metrics = [
                (threshold, *get_metrics(score_matrix=speakers_dist, threshold=threshold))
                for threshold in threshold_grid
            ]
            return {"loss": loss, "metrics": metrics}

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, step_type="train")

    def validation_step(self, val_batch, batch_idx):
        test_batch = [elem[:-2] for elem in val_batch]
        batch_treshold = [elem[-2:] for elem in val_batch]

        return {**self._step(test_batch, step_type="val"), **self._step(batch_treshold, step_type="treshold")}

    def _calculate_treshold(self, outputs):
        threshold_grid = np.array([item[0] for item in outputs[0]["treshold_metrics"]])

        far = np.array([0] * len(threshold_grid))
        frr = np.array([0] * len(threshold_grid))

        for output in outputs:
            for i, item in enumerate(output["treshold_metrics"]):
                far[i] += item[1]
                frr[i] += item[2]

        far = far / len(outputs)
        frr = frr / len(outputs)

        maximum = np.maximum(far, frr)
        cur_eer_threshold_index = (np.argmin(maximum) + len(maximum) - np.argmin(maximum[::-1])) // 2
        cur_threshold = threshold_grid[cur_eer_threshold_index]

        return cur_threshold, cur_eer_threshold_index

    def _estimate_metrics(self, outputs, eer_history, axs, end_type):
        threshold_grid = np.array([item[0] for item in outputs[0]["metrics"]])

        far = np.array([0] * len(threshold_grid))
        frr = np.array([0] * len(threshold_grid))

        for output in outputs:
            for i, item in enumerate(output["metrics"]):
                far[i] += item[1]
                frr[i] += item[2]

        far = far / len(outputs)
        frr = frr / len(outputs)

        maximum = np.maximum(far, frr)
        cur_eer = np.min(maximum)
        cur_eer_threshold_index = (np.argmin(maximum) + len(maximum) - np.argmin(maximum[::-1])) // 2
        cur_threshold = threshold_grid[cur_eer_threshold_index]

        eer = None
        eer_threshold_index = None
        threshold = None

        if self.train_threshold_index_history:
            eer_threshold_index = self.train_threshold_index_history[-1]
            eer = maximum[eer_threshold_index]
            threshold = threshold_grid[eer_threshold_index]
            axs[1].plot(cur_threshold, cur_eer, "o", label="best threshold", c="red")

        axs[1].plot(threshold_grid, far, label="FAR")
        axs[1].plot(threshold_grid, frr, label="FRR")

        if eer_threshold_index:
            eer_history.append(eer)
            threshold = threshold_grid[eer_threshold_index]
            axs[1].plot(threshold, eer, "o", label="threshold", c="green")

        axs[1].set_xlabel("threshold")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(range(len(eer_history)), eer_history, label="EER")
        axs[2].plot(
            range(len(eer_history)),
            smooth(eer_history, window_size=self.window_size),
            label=f"{self.window_size}-smooth EER",
            c="aqua"
        )
        axs[2].set_xlabel("epoch")
        axs[2].legend()
        axs[2].grid(True)

    def training_epoch_end(self, outputs):
        loss_values = [float(output['loss']) for output in outputs]
        avg_loss = np.mean(loss_values)
        self.train_loss_history.append(avg_loss)
        self.last_train_outputs = outputs

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([float(output['loss']) for output in outputs])
        self.val_loss_history.append(avg_loss)

        # train logs
        if self.last_train_outputs:
            _, cur_eer_threshold_index = self._calculate_treshold(outputs)
            self.train_threshold_index_history.append(cur_eer_threshold_index)

            fig, axs = plt.subplots(1, 3, figsize=(18, 4))

            axs[0].plot(range(len(self.train_loss_history)), self.train_loss_history, label="loss")
            axs[0].plot(
                range(len(self.train_loss_history)),
                smooth(self.train_loss_history, window_size=self.window_size),
                label=f"{self.window_size}-smooth loss",
                c="aqua"
            )
            axs[0].set_xlabel("epoch")
            axs[0].legend()
            axs[0].grid(True)

            self._estimate_metrics(self.last_train_outputs, self.train_eer_history, axs, end_type="train")
            plt.show()

            print(f"Train loss: {self.train_loss_history[-1]:.3f}\t",
                  f"Train EER: {self.train_eer_history[-1]:.3f}\t",
                  f"Smoothed train EER: {smooth(self.train_eer_history, window_size=self.window_size)[-1]:.3f}")

        # val logs
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))

        axs[0].plot(range(len(self.val_loss_history)), self.val_loss_history, label="loss")
        axs[0].plot(
            range(len(self.val_loss_history)),
            smooth(self.val_loss_history, window_size=self.window_size),
            label=f"{self.window_size}-smooth loss",
            c="aqua"
        )
        axs[0].set_xlabel("epoch")
        axs[0].legend()
        axs[0].grid(True)

        self._estimate_metrics(outputs, self.val_eer_history, axs, end_type="val")
        plt.show()

        if self.val_eer_history:
            print(f"Val loss: {self.val_loss_history[-1]:.3f}\t",
                  f"Val EER: {self.val_eer_history[-1]:.3f}\t",
                  f"Smoothed val EER: {smooth(self.val_eer_history, window_size=self.window_size)[-1]:.3f}")
