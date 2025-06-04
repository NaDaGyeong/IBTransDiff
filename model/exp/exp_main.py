from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

# from model.ns_models import ns_Transformer, ns_iTransformer, ns_Informer, ns_Autoformer
from model.exp.exp_basic import Exp_Basic
from model.diffusion_models.diffusion_utils import *
from model.diffusion_models import DDPM

from model.diffusion_models.DDPM_CNNNet import *
from model.diffusion_models.DDPM_diffusion_worker import *

from model.diffusion_models.samplers.dpm_sampler import DPMSolverSampler
from model.diffusion_models.samplers.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

from multiprocessing import Pool
import CRPS.CRPS as pscore

import warnings

import yaml
from tqdm import tqdm
import wandb
from layers.RevIN import *
from torch.nn.functional import mse_loss

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def ccc(id, pred, true):
    # print(id, datetime.datetime.now())
    res_box = np.zeros(len(true))
    for i in range(len(true)):
        try:
            res = pscore(pred[i], true[i]).compute()
            res_box[i] = res[0] if isinstance(res, (list, tuple, np.ndarray)) else res
        except Exception as e:
            print(f"[CCC ERROR] id={id}, i={i}, pred[i].shape={pred[i].shape}, error={e}")
            res_box[i] = np.nan  # or 0
    return res_box

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def visual_with_prediction_intervals(gt, pred_mean, p5=None, p25=None, p75=None, p95=None, name='./pic/test_interval_combined.pdf'):
    plt.figure()
    x = np.arange(len(gt))
    t_in = len(gt) - len(pred_mean)
    x_pred = np.arange(t_in, len(gt))

    plt.plot(x, gt, label='GroundTruth', linewidth=2, color='black')
    plt.plot(x_pred, pred_mean, label='Prediction', linewidth=2, color='darkgreen')

    if p5 is not None and p95 is not None:
        plt.fill_between(x_pred, p5, p95, color='mediumseagreen', alpha=0.2, label='90% Interval')
    if p25 is not None and p75 is not None:
        plt.fill_between(x_pred, p25, p75, color='seagreen', alpha=0.4, label='50% Interval')

    plt.legend(prop={'family': 'serif', 'weight': 'bold', 'size': 15})
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'DDPM': DDPM,
        }
        self.args.device = self.device
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode='Model'):
        if mode == 'Model':
            # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs)
        return model_optim, lr_scheduler

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def pretrain(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = optim.Adam(self.model.parameters(), lr=0.0001)

        best_train_loss = 10000000.0
        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                dec_inp = batch_y

                loss = self.model.pretrain_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                train_loss.append(loss.item())

                loss.backward()

                model_optim.step()

            print("PreTraining Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("PreTraining Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(epoch + 1, train_steps, train_loss))

            if train_loss < best_train_loss:
                print("-------------------------")
                best_train_loss = train_loss
                torch.save(self.model.cond_pred_model.state_dict(), path + '/' + 'cond_pred_model_checkpoint.pth')


    def train(self, setting):

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        best_model_path = path + '/' + 'cond_pred_model_checkpoint.pth'
        self.model.cond_pred_model.load_state_dict(torch.load(best_model_path), strict=False)
        print("Successfully loading pretrained model!")

        train_data, train_loader = self._get_data(flag='train')
        
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, lr_scheduler = self._select_optimizer()

        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # best_val_loss = float('inf')
        best_train_loss = 10000000.0
        training_process = {}
        training_process["train_loss"] = []
        training_process["val_loss"] = []
            
        for epoch in range(self.args.train_epochs):

            # Training the diffusion part
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device) 
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_dec_inp:  
                    loss = self.model.train_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:  
                    loss = self.model.train_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            training_process["train_loss"].append(train_loss)

            if epoch % 1 == 0:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss))
                wandb.log({
                            "Train Loss": train_loss,
                            "Val Loss": vali_loss,
                            "Epoch": epoch + 1
                        })
                training_process["val_loss"].append(vali_loss)
            
                if vali_loss < best_train_loss:
                        print("-------------------------")
                        best_train_loss = vali_loss
                        best_model_path = path + '/' + 'checkpoint.pth'
                        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            lr_scheduler.step()

            
            # save the model checkpoints
            early_stopping(vali_loss, self.model, path)

            if (math.isnan(train_loss)):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break



        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        inps = []    
        preds = []
        trues = []

        self.model.eval()

        # with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            if self.args.use_dec_inp: 
                outputs, batch_x, batch_y, mean, label_part = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=1)
            else: 
                outputs, batch_x, batch_y, mean, label_part = self.model.forward(batch_x, batch_x_mark, batch_y, batch_y_mark, sample_times=1)

            if len(np.shape(outputs)) == 4:
                outputs = outputs.mean(dim=1)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)
            inps.append(batch_x.detach().cpu().numpy())

            if i > 5:
                break

        inps = np.array(inps)
        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        wandb.log({
                        "Validation MSE": mse,
                        "Validation MAE": mae
                    })

        
        return mse


    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print("Successfully loading trained model!")

        def compute_true_coverage_by_gen_QI(dataset_object, all_true_y, all_generated_y):
            n_bins = 10
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)  

            assert all_generated_y.shape == all_true_y.shape, \
                f"Shape mismatch: generated {all_generated_y.shape}, true {all_true_y.shape}"

            y_pred_quantiles = np.percentile(all_generated_y, q=quantile_list, axis=0)

            y_true = all_true_y  

            diff = y_true[None, :, :] - y_pred_quantiles[:, None, :]
            quantile_membership_array = (diff > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0).flatten()  

            y_true_quantile_membership[y_true_quantile_membership == 0] = 1
            y_true_quantile_membership[y_true_quantile_membership == n_bins + 1] = n_bins

            y_true_quantile_bin_count = np.array([
                (y_true_quantile_membership == v).sum() for v in range(1, n_bins + 1)
            ])  

            y_true_ratio_by_bin = y_true_quantile_bin_count / y_true_quantile_membership.shape[0]
            
            assert np.abs(np.sum(y_true_ratio_by_bin) - 1) < 1e-5, \
                f"Sum of coverage ratios is {np.sum(y_true_ratio_by_bin):.6f}, not 1!"

            qice_coverage_ratio = np.abs(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()

            return y_true_ratio_by_bin, qice_coverage_ratio, y_true

        
        def compute_PICP(y_true, all_gen_y, return_CI=False):
            """
            Another coverage metric.
            """
            low, high = [2.5, 97.5]

            assert all_gen_y.shape == y_true.shape, \
            f"Shape mismatch: pred {all_gen_y.shape}, true {y_true.shape}"

            CI_y_pred = np.percentile(all_gen_y, q=[low, high], axis=0)
            # compute percentage of true y in the range of credible interval
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high
    

        inps = []   
        preds = []
        all_generated_samples = []
        trues = []
        
        return_mean = []
        return_label = []

        all_cond=[]

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                start_time = time.time()
                sample_times = self.args.sample_times

                if self.args.use_dec_inp:  
                    outputs, batch_x, dec_inp, cond, label_part = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=sample_times)
                else:  
                    outputs, batch_x, batch_y, cond, label_part = self.model.forward(batch_x, batch_x_mark, batch_y, batch_y_mark, sample_times=sample_times)
                
                end_time = time.time()
                elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x)[0]

                if i < 5:
                    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")

                cond = cond.detach().cpu().numpy()
                cond_mean = cond.mean(axis=1)  # shape: [B, D]
                all_cond.append(cond_mean)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()  # .squeeze()


                if len(np.shape(pred)) == 4:
                    preds.append(pred.mean(axis=1))
                    if self.args.sample_times > 1:
                        all_generated_samples.append(pred)
                else:
                    preds.append(pred)
                trues.append(true)

               
                if i%20 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    # gt = np.concatenate((input[0,:,-1], true[0,:,-1]), axis=0)
                    # pd = np.concatenate((input[0,:,-1], pred.mean(axis=1)[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    input = batch_x.detach().cpu().numpy()
                    gt_seq = true[0, :, -1]            
                    input_seq = input[0, :, -1]        
                    full_gt = np.concatenate((input_seq, gt_seq), axis=0)

                    samples = pred[:, 0, :, -1]        
                    p5 = np.percentile(samples, 5, axis=0)
                    p25 = np.percentile(samples, 25, axis=0)
                    p75 = np.percentile(samples, 75, axis=0)
                    p95 = np.percentile(samples, 95, axis=0)
                    pred_mean = np.mean(samples, axis=0)

                    full_pred = np.concatenate((input_seq, pred_mean), axis=0)

                    visual(full_gt, full_pred, os.path.join(folder_path, f"{i}_basic.pdf"))

                    visual_with_prediction_intervals(
                        gt=full_gt,
                        pred_mean=pred_mean,
                        p5=p5, p25=p25, p75=p75, p95=p95,
                        name=os.path.join(folder_path, f"{i}_interval_combined.pdf")
                    )

        inps = np.array(inps)
        preds = np.array(preds)
        trues = np.array(trues)

        preds_save = np.array(preds)
        trues_save = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        id_worst = None

        if self.args.sample_times > 1:
            all_generated_samples = np.array(all_generated_samples)

        preds = preds.reshape(-1, trues.shape[-2], trues.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        if self.args.sample_times > 1:
            all_generated_samples = all_generated_samples.reshape(-1, self.args.sample_times , trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        wandb.log({
                    "Test MSE": mse,
                    "Test MAE": mae
                })

        print('NT metrc: mse:{:.4f}, mae:{:.4f} , rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mse, mae, rmse, mape,
                                                                                                mspe))
        
        preds = preds.reshape(preds.shape[0], -1)

        trues = trues.reshape(trues.shape[0], -1)

        y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
            dataset_object=preds.shape[0],
            all_true_y=trues, all_generated_y=preds, )

        coverage, _, _ = compute_PICP(y_true=y_true, all_gen_y=preds)

        print('CARD metrc: QICE:{:.4f}%, PICP:{:.4f}%'.format(qice_coverage_ratio * 100, coverage * 100))

        print("preds_save.shape:", preds_save.shape)
        print("trues_save.shape:", trues_save.shape)

        B, S, T, D = preds_save.shape  

        pred = preds_save  
        true = trues_save[:, 0, :, :]  

        assert true.shape == (B, T, D), f"True shape mismatch: {true.shape}"


        pool = Pool(processes=32)
        all_res = []
        for i in range(D):
            p_in = pred[:, :, :, i]  # (B, S, T)
            p_in = p_in.transpose(0, 2, 1).reshape(B * T, S)
            t_in = true[:, :, i].reshape(B * T)
            
            assert p_in.shape[0] == t_in.shape[0], \
                f"Mismatch at dim {i}: pred {p_in.shape[0]} vs true {t_in.shape[0]}"
            
            all_res.append(pool.apply_async(ccc, args=(i, p_in, t_in)))

        p_sum = np.sum(pred, axis=-1).transpose(0, 2, 1).reshape(-1, pred.shape[1])
        t_sum = np.sum(true, axis=-1).reshape(-1)
        CRPS_sum = pool.apply_async(ccc, args=(999, p_sum, t_sum))


        pool.close()
        pool.join()

        all_res_get = []
        for i in range(len(all_res)):
            all_res_get.append(all_res[i].get())
        all_res_get = np.array(all_res_get)

        CRPS_0 = np.mean(all_res_get, axis=0).mean()
        CRPS_sum = CRPS_sum.get()
        CRPS_sum = CRPS_sum.mean()

        print('CRPS', CRPS_0, 'CRPS_sum', CRPS_sum)

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return
    

    def test_robust(self, setting, test=0, noise_std=0.1):
        """
        Evaluate model robustness under Gaussian noise perturbation.
        noise_std: standard deviation of Gaussian noise to add to input.
        """
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        print("Successfully loading trained model for robustness test!")

        self.model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Add Gaussian noise to inputs
                batch_x += torch.randn_like(batch_x) * noise_std
                batch_y_noised = batch_y + torch.randn_like(batch_y) * noise_std

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_dec_inp:
                    outputs, _, _, _, _ = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark, sample_times=1)
                else:
                    outputs, _, _, _, _ = self.model.forward(batch_x, batch_x_mark, batch_y, batch_y_mark, sample_times=1)

                if len(outputs.shape) == 4:
                    outputs = outputs.mean(dim=1)

                pred = outputs.detach().cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()  # Slice true to pred_len

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print(f"[Noise std = {noise_std}] Robustness Test Results:")
        print('MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}'.format(mae, mse, rmse))
        return mae, mse, rmse

    def test_cross(self, setting, cross_dataset_name='ETTm1', cross_data_path='ETTm1.csv'):
        original_data = self.args.data
        original_data_path = self.args.data_path
        original_root_path = self.args.root_path

        self.args.data = cross_dataset_name
        self.args.data_path = cross_data_path
        self.args.root_path = './dataset/ETT-small/'

        _, test_loader = self._get_data(flag='test')
        self.args.data = original_data
        self.args.data_path = original_data_path
        self.args.root_path = original_root_path

        print(f"Cross Test Dataset: {cross_dataset_name} / {cross_data_path}")
        print('>>>>>>>cross testing : {} on {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting, cross_dataset_name))
        self.test(setting)