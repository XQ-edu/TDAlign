import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import MICN, DLinear, PatchTST, SegRNN, TimesNet, iTransformer
from utils.metrics import compute_loss_sgn, metric
from utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop, visual

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if self.args.adaptive:
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def _build_model(self):
        model_dict = {
            "DLinear": DLinear,
            "PatchTST": PatchTST,
            "TimesNet": TimesNet,
            "MICN": MICN,
            "iTransformer": iTransformer,
            "SegRNN": SegRNN,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        num_params_model = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters in model: {num_params_model}")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.adaptive:
            model_optim = optim.Adam(
                [*self.model.parameters(), self.alpha],
                lr=self.args.learning_rate,
            )
        else:
            model_optim = optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_criterion(self):
    #     if self.args.loss == "mse":
    #         self.criterion = nn.MSELoss()
    #     elif self.args.loss == "mae":
    #         self.criterion = nn.L1Loss()

    def _comput_loss(self, y_pred, y_true, x, flag=None):
        x = x.to(y_true.device)

        if (
            self.args.d == 1 and self.args.k == 1
        ):  # TDOS: 1-order differencing between y_t and y_{t+1}
            tmp = torch.cat([x[:, -1:, :], y_true], dim=1)
            d_true = torch.diff(tmp, n=1, dim=1)
            tmp = torch.cat([x[:, -1:, :], y_pred], dim=1)
            d_pred = torch.diff(tmp, n=1, dim=1)
        elif self.args.d > 1 and self.args.k == 1:  # TDOS: d-order differencing
            tmp = torch.cat([x[:, -self.args.d :, :], y_true], dim=1)
            d_true = torch.diff(tmp, n=self.args.d, dim=1)
            tmp = torch.cat([x[:, -self.args.d :, :], y_pred], dim=1)
            d_pred = torch.diff(tmp, n=self.args.d, dim=1)
        elif self.args.d == 1 and self.args.k > 1:  # difference between y_t and y_{t+k}
            tmp = torch.cat([x[:, -self.args.k :, :], y_true], dim=1)
            d_true = tmp[:, self.args.k :, :] - tmp[:, : -self.args.k, :]
            tmp = torch.cat([x[:, -self.args.k :, :], y_pred], dim=1)
            d_pred = tmp[:, self.args.k :, :] - tmp[:, : -self.args.k, :]
        else:
            raise ValueError("Invalid d and k values")

        if self.args.task == "improve":
            loss_y = torch.mean(torch.mean(torch.abs(y_pred - y_true), dim=2), dim=1)
            loss_d = torch.mean(torch.mean(torch.abs(d_pred - d_true), dim=2), dim=1)
            loss_sgn = torch.sum(
                torch.sum(torch.mul(d_pred, d_true) < 0, dim=2), dim=1
            ).float() / (d_pred.shape[1] * d_pred.shape[2])

            if self.args.adaptive:
                loss = torch.mean(self.alpha * loss_y + (1 - self.alpha) * loss_d)
            elif self.args.no_sgn:
                loss = torch.mean(loss_y + loss_d)
            elif self.args.no_d:
                loss = torch.mean(loss_y * loss_sgn)
            else:
                loss = torch.mean(loss_sgn * loss_y + (1 - loss_sgn) * loss_d)

            loss_y_mae = torch.mean(loss_y)
            loss_d_mae = torch.mean(loss_d)
            loss_sgn = torch.mean(loss_sgn)

            loss_y_mse = nn.MSELoss()(y_pred, y_true)
            loss_d_mse = nn.MSELoss()(d_pred, d_true)

        elif self.args.task == "origin":
            loss_y_mae = nn.L1Loss()(y_pred, y_true)
            loss_d_mae = nn.L1Loss()(d_pred, d_true)
            loss_sgn = torch.sum(torch.mul(d_pred, d_true) < 0).float() / (
                d_pred.shape[0] * d_pred.shape[1] * d_pred.shape[2]
            )
            loss_y_mse = nn.MSELoss()(y_pred, y_true)
            loss_d_mse = nn.MSELoss()(d_pred, d_true)
            if self.args.loss == "mse":
                loss = loss_y_mse
            elif self.args.loss == "mae":
                loss = loss_y_mae

        return loss, loss_y_mae, loss_d_mae, loss_sgn, loss_y_mse, loss_d_mse

    def save_epoch_loss(
        self,
        setting,
        e_loss_y_mae,
        e_loss_d_mae,
        e_loss_sgn,
        e_loss_y_mse,
        e_loss_d_mse,
        flag=None,
    ):
        results_path = os.path.join(self.args.results, setting)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        e_loss = np.array(
            [e_loss_y_mae, e_loss_d_mae, e_loss_sgn, e_loss_y_mse, e_loss_d_mse]
        )
        np.save(results_path + "/" + f"{flag}_e_loss.npy", e_loss)

    def vali(self, vali_data, vali_loader, flag=None):
        vali_loss = []
        vali_loss_y_mae = []
        vali_loss_d_mae = []
        vali_loss_sgn = []
        vali_loss_y_mse = []
        vali_loss_d_mse = []

        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_label = batch_y[:, -self.args.label_len :, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(
                            substr in self.args.model
                            for substr in {"Linear", "TST", "SegRNN"}
                        ):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                else:
                    if any(
                        substr in self.args.model
                        for substr in {"Linear", "TST", "SegRNN"}
                    ):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss, loss_y_mae, loss_d_mae, loss_sgn, loss_y_mse, loss_d_mse = (
                    self._comput_loss(
                        pred,
                        true,
                        batch_x,
                        flag=flag,
                    )
                )

                vali_loss.append(loss.cpu().item())
                vali_loss_y_mae.append(loss_y_mae.cpu().item())
                vali_loss_d_mae.append(loss_d_mae.cpu().item())
                vali_loss_sgn.append(loss_sgn.cpu().item())
                vali_loss_y_mse.append(loss_y_mse.cpu().item())
                vali_loss_d_mse.append(loss_d_mse.cpu().item())

        vali_loss = np.average(vali_loss)
        vali_loss_y_mae = np.average(vali_loss_y_mae)
        vali_loss_d_mae = np.average(vali_loss_d_mae)
        vali_loss_sgn = np.average(vali_loss_sgn)
        vali_loss_y_mse = np.average(vali_loss_y_mse)
        vali_loss_d_mse = np.average(vali_loss_d_mse)

        self.model.train()
        return (
            vali_loss,
            vali_loss_y_mae,
            vali_loss_d_mae,
            vali_loss_sgn,
            vali_loss_y_mse,
            vali_loss_d_mse,
        )

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        if self.args.save_epoch_loss:
            e_train_loss_y_mae = []
            e_train_loss_d_mae = []
            e_train_loss_sgn = []
            e_train_loss_y_mse = []
            e_train_loss_d_mse = []

            e_vali_loss_y_mae = []
            e_vali_loss_d_mae = []
            e_vali_loss_sgn = []
            e_vali_loss_y_mse = []
            e_vali_loss_d_mse = []

            e_test_loss_y_mae = []
            e_test_loss_d_mae = []
            e_test_loss_sgn = []
            e_test_loss_y_mse = []
            e_test_loss_d_mse = []

        model_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if any(substr in self.args.model for substr in {"TST", "SegRNN"}):
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate,
            )
        else:
            scheduler = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_y_mae = []
            train_loss_d_mae = []
            train_loss_sgn = []
            train_loss_y_mse = []
            train_loss_d_mse = []

            self.model.train()
            epoch_time = time.time()
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_label = batch_y[:, -self.args.label_len :, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(
                            substr in self.args.model
                            for substr in {"Linear", "TST", "SegRNN"}
                        ):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                else:
                    if any(
                        substr in self.args.model
                        for substr in {"Linear", "TST", "SegRNN"}
                    ):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                loss, loss_y_mae, loss_d_mae, loss_sgn, loss_y_mse, loss_d_mse = (
                    self._comput_loss(outputs, batch_y, batch_x, "train")
                )
                train_loss.append(loss.item())
                train_loss_y_mae.append(loss_y_mae.item())
                train_loss_d_mae.append(loss_d_mae.item())
                train_loss_sgn.append(loss_sgn.item())
                train_loss_y_mse.append(loss_y_mse.item())
                train_loss_d_mse.append(loss_d_mse.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    model_optim.zero_grad()

                if self.args.lradj == "TST":
                    adjust_learning_rate(
                        model_optim, scheduler, epoch + 1, self.args, printout=False
                    )
                    scheduler.step()

            print(
                "Epoch: {0}, Steps: {1}, Cost Time: {2}".format(
                    epoch + 1, train_steps, time.time() - epoch_time
                )
            )
            train_loss = np.average(train_loss)
            train_loss_y_mae = np.average(train_loss_y_mae)
            train_loss_d_mae = np.average(train_loss_d_mae)
            train_loss_sgn = np.average(train_loss_sgn)
            train_loss_y_mse = np.average(train_loss_y_mse)
            train_loss_d_mse = np.average(train_loss_d_mse)

            (
                vali_loss,
                vali_loss_y_mae,
                vali_loss_d_mae,
                vali_loss_sgn,
                vali_loss_y_mse,
                vali_loss_d_mse,
            ) = self.vali(vali_data, vali_loader, "vali")
            (
                test_loss,
                test_loss_y_mae,
                test_loss_d_mae,
                test_loss_sgn,
                test_loss_y_mse,
                test_loss_d_mse,
            ) = self.vali(test_data, test_loader, "test")

            print(
                f"train_loss_y_mae: {train_loss_y_mae:.7f} | train_loss_d_mae: {train_loss_d_mae:.7f} | train_loss_sgn: {train_loss_sgn:.7f} | train_loss_y_mse: {train_loss_y_mse:.7f} | train_loss_d_mse: {train_loss_d_mse:.7f} | train_loss: {train_loss:.7f}\nvali_loss_y_mae: {vali_loss_y_mae:.7f} | vali_loss_d_mae: {vali_loss_d_mae:.7f} | vali_loss_sgn: {vali_loss_sgn:.7f} | vali_loss_y_mse: {vali_loss_y_mse:.7f} | vali_loss_d_mse: {vali_loss_d_mse:.7f} | vali_loss: {vali_loss:.7f}\ntest_loss_y_mae: {test_loss_y_mae:.7f} | test_loss_d_mae: {test_loss_d_mae:.7f} | test_loss_sgn: {test_loss_sgn:.7f} | test_loss_y_mse: {test_loss_y_mse:.7f} | test_loss_d_mse: {test_loss_d_mse:.7f} | test_loss: {test_loss:.7f}"
            )

            if self.args.save_epoch_loss:
                e_train_loss_y_mae.append(train_loss_y_mae)
                e_train_loss_d_mae.append(train_loss_d_mae)
                e_train_loss_sgn.append(train_loss_sgn)
                e_train_loss_y_mse.append(train_loss_y_mse)
                e_train_loss_d_mse.append(train_loss_d_mse)

                e_vali_loss_y_mae.append(vali_loss_y_mae)
                e_vali_loss_d_mae.append(vali_loss_d_mae)
                e_vali_loss_sgn.append(vali_loss_sgn)
                e_vali_loss_y_mse.append(vali_loss_y_mse)
                e_vali_loss_d_mse.append(vali_loss_d_mse)

                e_test_loss_y_mae.append(test_loss_y_mae)
                e_test_loss_d_mae.append(test_loss_d_mae)
                e_test_loss_sgn.append(test_loss_sgn)
                e_test_loss_y_mse.append(test_loss_y_mse)
                e_test_loss_d_mse.append(test_loss_d_mse)

            early_stopping(vali_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping!!!")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = model_path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))

        if self.args.save_epoch_loss:
            self.save_epoch_loss(
                setting,
                e_train_loss_y_mae,
                e_train_loss_d_mae,
                e_train_loss_sgn,
                e_train_loss_y_mse,
                e_train_loss_d_mse,
                flag="train",
            )
            self.save_epoch_loss(
                setting,
                e_vali_loss_y_mae,
                e_vali_loss_d_mae,
                e_vali_loss_sgn,
                e_vali_loss_y_mse,
                e_vali_loss_d_mse,
                flag="vali",
            )
            self.save_epoch_loss(
                setting,
                e_test_loss_y_mae,
                e_test_loss_d_mae,
                e_test_loss_sgn,
                e_test_loss_y_mse,
                e_test_loss_d_mse,
                flag="test",
            )

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag="test")

        y_preds = []
        y_trues = []
        d_preds = []
        d_trues = []
        inputx = []

        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                input_x = batch_x
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_label = batch_y[:, -self.args.label_len :, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(
                            substr in self.args.model
                            for substr in {"Linear", "TST", "SegRNN"}
                        ):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                else:
                    if any(
                        substr in self.args.model
                        for substr in {"Linear", "TST", "SegRNN"}
                    ):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                if (
                    self.args.d == 1 and self.args.k == 1
                ):  # TDOS: 1-order differencing between y_t and y_{t+1}
                    tmp = torch.cat([batch_x[:, -1:, :], batch_y], dim=1)
                    d_true = torch.diff(tmp, n=1, dim=1)
                    tmp = torch.cat([batch_x[:, -1:, :], outputs], dim=1)
                    d_pred = torch.diff(tmp, n=1, dim=1)
                elif self.args.d > 1 and self.args.k == 1:  # TDOS: d-order differencing
                    tmp = torch.cat([batch_x[:, -self.args.d :, :], batch_y], dim=1)
                    d_true = torch.diff(tmp, n=self.args.d, dim=1)
                    tmp = torch.cat([batch_x[:, -self.args.d :, :], outputs], dim=1)
                    d_pred = torch.diff(tmp, n=self.args.d, dim=1)
                elif (
                    self.args.d == 1 and self.args.k > 1
                ):  # difference between y_t and y_{t+k}
                    tmp = torch.cat([batch_x[:, -self.args.k :, :], batch_y], dim=1)
                    d_true = tmp[:, self.args.k :, :] - tmp[:, : -self.args.k, :]
                    tmp = torch.cat([batch_x[:, -self.args.k :, :], outputs], dim=1)
                    d_pred = tmp[:, self.args.k :, :] - tmp[:, : -self.args.k, :]

                y_pred = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                d_true = d_true.detach().cpu().numpy()
                d_pred = d_pred.detach().cpu().numpy()
                batch_x = input_x.detach().cpu().numpy()

                y_preds.append(y_pred)
                y_trues.append(batch_y)
                d_preds.append(d_pred)
                d_trues.append(d_true)
                inputx.append(batch_x)

        del (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            input_x,
            dec_inp,
            dec_label,
            y_pred,
            d_pred,
        )
        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        d_preds = np.concatenate(d_preds, axis=0)
        d_trues = np.concatenate(d_trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        results_path = os.path.join(self.args.results, setting)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        y_mae, y_mse, y_rmse, y_mape, y_mspe, y_rse, y_corr = metric(y_preds, y_trues)
        print("y_mse:{}, y_mae:{}".format(y_mse, y_mae))
        np.save(
            results_path + "/" + "y_metrics.npy",
            np.array([y_mae, y_mse, y_rmse, y_mape, y_mspe, y_rse]),
        )
        np.save(results_path + "/" + "y_preds.npy", y_preds)
        del y_preds
        np.save(results_path + "/" + "y_trues.npy", y_trues)
        del y_trues

        (
            d_mae,
            d_mse,
            d_rmse,
            d_mape,
            d_mspe,
            d_rse,
            d_corr,
        ) = metric(d_preds, d_trues)
        loss_sgn = compute_loss_sgn(d_preds, d_trues)
        print("d_mse:{}, d_mae:{}".format(d_mse, d_mae))
        print("loss_sgn:{}".format(loss_sgn))
        np.save(results_path + "/" + "loss_sgn.npy", np.array([loss_sgn]))
        np.save(
            results_path + "/" + "d_metrics.npy",
            np.array([d_mae, d_mse, d_rmse, d_mape, d_mspe, d_rse]),
        )
        np.save(results_path + "/" + "d_preds.npy", d_preds)
        del d_preds
        np.save(results_path + "/" + "d_trues.npy", d_trues)
        del d_trues

        np.save(results_path + "/" + "inputx.npy", inputx)
        return y_mse, y_mae, d_mse, d_mae, loss_sgn

    def predict(self, setting):
        pred_data, pred_loader = self._get_data(flag="test")

        model_path = os.path.join(self.args.checkpoints, setting)
        best_model_path = model_path + "/" + "checkpoint.pth"
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=torch.device("cuda"))
        )

        y_preds = []
        y_trues = []
        d_preds = []
        d_trues = []
        inputx = []

        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                input_x = batch_x
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_label = batch_y[:, -self.args.label_len :, :]
                dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(
                            substr in self.args.model
                            for substr in {"Linear", "TST", "SegRNN"}
                        ):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                else:
                    if any(
                        substr in self.args.model
                        for substr in {"Linear", "TST", "SegRNN"}
                    ):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                if (
                    self.args.d == 1 and self.args.k == 1
                ):  # TDOS: 1-order differencing between y_t and y_{t+1}
                    tmp = torch.cat([batch_x[:, -1:, :], batch_y], dim=1)
                    d_true = torch.diff(tmp, n=1, dim=1)
                    tmp = torch.cat([batch_x[:, -1:, :], outputs], dim=1)
                    d_pred = torch.diff(tmp, n=1, dim=1)
                elif self.args.d > 1 and self.args.k == 1:  # TDOS: d-order differencing
                    tmp = torch.cat([batch_x[:, -self.args.d :, :], batch_y], dim=1)
                    d_true = torch.diff(tmp, n=self.args.d, dim=1)
                    tmp = torch.cat([batch_x[:, -self.args.d :, :], outputs], dim=1)
                    d_pred = torch.diff(tmp, n=self.args.d, dim=1)
                elif (
                    self.args.d == 1 and self.args.k > 1
                ):  # difference between y_t and y_{t+k}
                    tmp = torch.cat([batch_x[:, -self.args.k :, :], batch_y], dim=1)
                    d_true = tmp[:, self.args.k :, :] - tmp[:, : -self.args.k, :]
                    tmp = torch.cat([batch_x[:, -self.args.k :, :], outputs], dim=1)
                    d_pred = tmp[:, self.args.k :, :] - tmp[:, : -self.args.k, :]

                y_pred = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                d_true = d_true.detach().cpu().numpy()
                d_pred = d_pred.detach().cpu().numpy()
                batch_x = input_x.detach().cpu().numpy()

                y_preds.append(y_pred)
                y_trues.append(batch_y)
                d_preds.append(d_pred)
                d_trues.append(d_true)
                inputx.append(batch_x)

        del (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            input_x,
            dec_inp,
            dec_label,
            y_pred,
            d_pred,
        )
        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        d_preds = np.concatenate(d_preds, axis=0)
        d_trues = np.concatenate(d_trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        results_path = os.path.join(self.args.results, setting)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        y_mae, y_mse, y_rmse, y_mape, y_mspe, y_rse, y_corr = metric(y_preds, y_trues)
        print("y_mse:{}, y_mae:{}".format(y_mse, y_mae))
        np.save(
            results_path + "/" + "y_metrics.npy",
            np.array([y_mae, y_mse, y_rmse, y_mape, y_mspe, y_rse]),
        )
        np.save(results_path + "/" + "y_preds.npy", y_preds)
        del y_preds
        np.save(results_path + "/" + "y_trues.npy", y_trues)
        del y_trues

        (
            d_mae,
            d_mse,
            d_rmse,
            d_mape,
            d_mspe,
            d_rse,
            d_corr,
        ) = metric(d_preds, d_trues)
        loss_sgn = compute_loss_sgn(d_preds, d_trues)
        print("d_mse:{}, d_mae:{}".format(d_mse, d_mae))
        print("loss_sgn:{}".format(loss_sgn))
        np.save(results_path + "/" + "loss_sgn.npy", np.array([loss_sgn]))
        np.save(
            results_path + "/" + "d_metrics.npy",
            np.array([d_mae, d_mse, d_rmse, d_mape, d_mspe, d_rse]),
        )
        np.save(results_path + "/" + "d_preds.npy", d_preds)
        del d_preds
        np.save(results_path + "/" + "d_trues.npy", d_trues)
        del d_trues

        np.save(results_path + "/" + "inputx.npy", inputx)
