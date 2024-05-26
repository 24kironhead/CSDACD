import torch
import torch.nn.functional as F
import os
import os.path as osp
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage import io
from tqdm import tqdm

from utils.data_utils.misc import (
    to_array, to_pseudo_color,
    normalize_minmax, normalize_8bit,
    quantize_8bit as quantize,
)
from utils.utils import HookHelper, FeatureContainer
from utils.metrics import (Meter, Precision, Recall, Accuracy, F1Score, IoU)

from core.trainer import Trainer
from  core.factories import model_factory, critn_factory, data_factory, optim_factory, optim_factory_gan
from utils.utils import build_schedulers, build_schedulers_gan, ReplayBuffer
from utils.losses import MixedLoss, CombinedLoss
from torch.autograd import Variable
import constants
import shutil
import cv2

class CSDACDTrainer(Trainer):
    def __init__(self, settings):
        super().__init__(settings['model'], settings['dataset'], settings['criterion'], settings['optimizer'], settings)

        # Whether to save network output
        self.save = self.ctx['save_on'] and not self.debug
        if self.save: 
            self._mt_pool = ThreadPoolExecutor(max_workers=2)
        self.out_dir = self.ctx['out_dir']

        # Build lr schedulers
        self.sched_on = self.ctx['sched_on'] and self.is_training

        self.lr_gan = settings['lr_gan']
        self.lambda_cyc = settings['lambda_cyc']
        self.lambda_id = settings['lambda_id']
        self.lambda_sca = settings['lambda_sca']
        self.checkpoint_G_SW = settings['resume_G_SW']
        self.checkpoint_G_WS = settings['resume_G_WS']
        self.checkpoint_D_S = settings['resume_D_S']
        self.checkpoint_D_W = settings['resume_D_W']

        self.batch_size = settings['batch_size']
        self.image_size = settings['crop_size']
        self.channel = settings['channel']

        (self.model, self.gan_SW, self.gan_WS, self.dis_S, self.dis_W) = model_factory(settings['model'], settings)
        self.model.to(self.device)
        self.gan_SW.to(self.device)
        self.gan_WS.to(self.device)
        self.dis_S.to(self.device)
        self.dis_W.to(self.device)

        (self.criterion, self.criterion_mse, self.criterion_l1, self.criterion_sca) = critn_factory(settings['criterion'], settings)
        self.criterion.to(self.device)
        self.criterion_mse.to(self.device)
        self.criterion_l1.to(self.device)
        self.criterion_sca.to(self.device)

        self._init_trainer()

        if self.is_training:
            self.train_loader = data_factory(settings['dataset'], 'train', settings)
            self.eval_loader = data_factory(settings['dataset'], 'eval', settings)
            self.optimizer = optim_factory(settings['optimizer'], self.model, settings)
            self.optimizer_gan = optim_factory_gan(settings['optimizer_gan'], self.gan_SW, self.gan_WS, settings)
            self.optimizer_dis_S = optim_factory(settings['optimizer_gan'], self.dis_S, settings)
            self.optimizer_dis_W = optim_factory(settings['optimizer_gan'], self.dis_W, settings)
        else:
            self.eval_loader = data_factory(settings['dataset'], 'eval', settings)
        
        if self.sched_on:
            self.schedulers = build_schedulers(self.ctx['schedulers'], self.optimizer)
            self.schedulers_gan = build_schedulers_gan(self.ctx['schedulers_gan'], self.optimizer_gan)
            self.schedulers_dis_S = build_schedulers_gan(self.ctx['schedulers_gan_DS'], self.optimizer_dis_S)
            self.schedulers_dis_W = build_schedulers_gan(self.ctx['schedulers_gan_DW'], self.optimizer_dis_W)
        
    def init_learning_rate(self):
        if not self.sched_on:
            self.lr = super().init_learning_rate()
        else:
            for idx, sched in enumerate([self.schedulers[0], self.schedulers_gan[0], self.schedulers_dis_S[0], self.schedulers_dis_W[0]]):
                if self.start_epoch > 0:
                    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.logger.warn("The old state of lr scheduler {} will not be restored.".format(idx))
                        continue
                    # Restore previous state
                    # FIXME: This will trigger pytorch warning "Detected call of `lr_scheduler.step()` 
                    # before `optimizer.step()`" in pytorch 1.1.0 and later.
                    # Perhaps I should store the state of scheduler to a checkpoint file and restore it from disk.
                    last_epoch = self.start_epoch
                    while sched.last_epoch < last_epoch:
                        sched.step()
            self.lr = self.optimizer.param_groups[0]['lr']
            self.lr_gan = self.optimizer_gan.param_groups[0]['lr']
        return self.lr, self.lr_gan

    def adjust_learning_rate(self, epoch, acc):
        if not self.sched_on:
            self.lr = super().adjust_learning_rate(epoch, acc)
        else:
            for sched in [self.schedulers[0], self.schedulers_gan[0], self.schedulers_dis_S[0], self.schedulers_dis_W[0]]:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(acc)
                else:
                    sched.step()
            self.lr = self.optimizer.param_groups[0]['lr']
            self.lr_gan = self.optimizer_gan.param_groups[0]['lr']
        return self.lr, self.lr_gan
    
    def train_epoch(self, epoch):
        losses = Meter()
        loss_gan = Meter()
        loss_Dis = Meter()
        fake_S_buffer = ReplayBuffer()
        recov_S_buffer = ReplayBuffer()
        fake_W_buffer = ReplayBuffer()
        recov_W_buffer = ReplayBuffer()
        len_train = len(self.train_loader)
        width = len(str(len_train))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.train_loader)
        Tensor = torch.cuda.FloatTensor

        for i, (t1, t2, tar) in enumerate(pb):
            t1, t2, tar = self._prepare_data(t1, t2, tar)
            
            fetch_dict = self._set_fetch_dict()
            out_dict = FeatureContainer()

            #- DA training -#
            # Refer to https://github.com/Perfect-You/SDACD

            # -----------------------
            #  Train Generator
            # -----------------------

            valid = Variable(Tensor(np.ones((t1.size(0), *self.dis_S.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((t1.size(0), *self.dis_S.output_shape))), requires_grad=False)

            self.gan_SW.train()
            self.gan_WS.train()
            self.optimizer_gan.zero_grad()

            # Identity loss
            loss_id_S = self.criterion_l1(self.gan_WS(t1), t1)
            loss_id_W = self.criterion_l1(self.gan_SW(t2), t2)

            loss_identity = (loss_id_S + loss_id_W) / 2

            # GAN loss
            fake_S = self.gan_WS(t2).detach()
            fake_W = self.gan_SW(t1).detach()
            loss_GAN_SW = self.criterion_mse(self.dis_W(fake_W), valid)
            loss_GAN_WS = self.criterion_mse(self.dis_S(fake_S), valid)

            # G_SCA loss
            labels_SCA = 0.5 - tar
            labels_SCA = labels_SCA.unsqueeze(1).expand(self.batch_size, self.channel, self.image_size, self.image_size)
            loss_SCA = torch.mean(labels_SCA * self.criterion_sca(fake_S, t1)) + torch.mean(labels_SCA * self.criterion_sca(fake_W, t2))

            # Cycle loss
            recov_S = self.gan_WS(fake_W)
            loss_cycle_S = self.criterion_l1(recov_S, t1)
            loss_GAN_cycle_S = self.criterion_mse(self.dis_S(recov_S),valid)
            recov_W = self.gan_SW(fake_S)
            loss_cycle_W = self.criterion_l1(recov_W, t2)
            loss_GAN_cycle_W = self.criterion_mse(self.dis_W(recov_W),valid)

            loss_GAN = (loss_GAN_SW + loss_GAN_WS + loss_GAN_cycle_S + loss_GAN_cycle_W) / 4

            loss_cycle = (loss_cycle_S + loss_cycle_W) / 2

            # Total loss
            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity + self.lambda_sca * loss_SCA

            loss_gan.update(loss_G.item(), n=tar.shape[0])

            loss_G.backward()
            self.optimizer_gan.step()

            # -----------------------
            #  Train Discriminator S
            # -----------------------

            self.optimizer_dis_S.zero_grad()

            # Real loss
            loss_real = self.criterion_mse(self.dis_S(t1), valid)
            # Fake loss (on batch of previously generated samples)
            fake_S_ = fake_S_buffer.push_and_pop(fake_S)
            loss_fake = self.criterion_mse(self.dis_S(fake_S_.detach()), fake)
            recov_S_ = recov_S_buffer.push_and_pop(recov_S)
            loss_fake_cycle = self.criterion_mse(self.dis_S(recov_S_.detach()),fake)
            # Total loss
            loss_D_S = (loss_real + loss_fake + loss_fake_cycle) / 3

            loss_D_S.backward()
            self.optimizer_dis_S.step()

            # -----------------------
            #  Train Discriminator W
            # -----------------------

            self.optimizer_dis_W.zero_grad()

            # Real loss
            loss_real = self.criterion_mse(self.dis_W(t2), valid)
            # Fake loss (on batch of previously generated samples)
            fake_W_ = fake_W_buffer.push_and_pop(fake_W)
            loss_fake = self.criterion_mse(self.dis_W(fake_W_.detach()), fake)
            recov_W_ = recov_W_buffer.push_and_pop(recov_W)
            loss_fake_cycle = self.criterion_mse(self.dis_W(recov_W_.detach()), fake)
            # Total loss
            loss_D_W = (loss_real + loss_fake + loss_fake_cycle) / 3

            loss_D_W.backward()
            self.optimizer_dis_W.step()

            loss_D = (loss_D_S + loss_D_W) / 2
            loss_Dis.update(loss_D.item(), n=tar.shape[0])

            #- CD training -#

            self.model.train()
            self.optimizer.zero_grad()

            with HookHelper(self.model, fetch_dict, out_dict, hook_type='forward_out'):
                out = self.model(t1, t2, fake_W, fake_S)

            pred = self._process_model_out(out)
            
            loss = self.criterion(pred, tar)
            losses.update(loss.item(), n=tar.shape[0])

            loss.backward()
            self.optimizer.step()

            
            desc = (start_pattern+" Loss_CD: {:.4f} ({:.4f})\n").format(i+1, len_train, losses.val, losses.avg)
            desc += ("Loss_GAN: {:.4f} ({:.4f})\n").format(loss_gan.val, loss_gan.avg)
            desc += ("Loss_D: {:.4f} ({:.4f})\n").format(loss_Dis.val, loss_Dis.avg)

            pb.set_description(desc)
            if i % max(1, len_train//10) == 0:
                self.logger.dump(desc)

    def evaluate_epoch(self, epoch):
        self.logger.show_nl("Epoch: [{0}]".format(epoch))
        losses = Meter()
        len_eval = len(self.eval_loader)
        width = len(str(len_eval))
        start_pattern = "[{{:>{0}}}/{{:>{0}}}]".format(width)
        pb = tqdm(self.eval_loader)

        # Construct metrics
        metrics = (Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), IoU(mode='accum'), Accuracy(mode='accum'))

        self.model.eval()
        self.gan_SW.eval()
        self.gan_WS.eval()

        with torch.no_grad():
            for i, (name, t1, t2, tar) in enumerate(pb):
                t1, t2, tar = self._prepare_data(t1, t2, tar)
                batch_size = tar.shape[0]

                fetch_dict = self._set_fetch_dict()
                out_dict = FeatureContainer()
                fake_W = self.gan_SW(t1)
                fake_S = self.gan_WS(t2)
                
                with HookHelper(self.model, fetch_dict, out_dict, hook_type='forward_out'):
                    out = self.model(t1, t2, fake_W, fake_S)

                pred = self._process_model_out(out)

                loss = self.criterion(pred, tar)
                losses.update(loss.item(), n=batch_size)

                # Convert to numpy arrays
                prob = self._pred_to_prob(pred)
                prob = prob.cpu().numpy()
                cm = (prob>0.5).astype('uint8')
                tar = tar.cpu().numpy().astype('uint8')

                for m in metrics:
                    m.update(cm, tar, n=batch_size)

                desc = (start_pattern+" Loss: {:.4f} ({:.4f})").format(i+1, len_eval, losses.val, losses.avg)
                for m in metrics:
                    desc += " {} {:.4f}".format(m.__name__, m.val)
                desc += " Params {}".format(self.count_parameters(self.model) + self.count_parameters(self.gan_SW) + self.count_parameters(self.gan_WS))
                desc += "\n"

                pb.set_description(desc)
                dump = not self.is_training or (i % max(1, len_eval//10) == 0)
                if dump:
                    self.logger.dump(desc)
                
                if self.save:
                    for j in range(batch_size):
                        self.save_image(name[j], quantize(cm[j]), epoch)
                        fake_S_out, fake_W_out = to_array(fake_S[j]), to_array(fake_W[j])
                        fake_S_out, fake_W_out = self._denorm_image(fake_S_out).astype('uint8'), self._denorm_image(fake_W_out).astype('uint8')
                        self.save_image_gan(str(name[j]), fake_W_out, str(epoch)+'_G_AB')
                        self.save_image_gan(str(name[j]), fake_S_out, str(epoch)+'_G_BA')
        return metrics[2].val   # F1-score
    
    def train(self):
        if self.load_checkpoint:
            self._resume_from_checkpoint()

        max_acc, best_epoch = self._init_acc_epoch
        lr, lr_gan = self.init_learning_rate()

        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.show_nl("Epoch: [{0}]\tlr {1:.06f}\tlr_gan {2:.06f}".format(epoch, lr, lr_gan))

            # Train for one epoch
            self.train_epoch(epoch)
            
            # Evaluate the model
            self.logger.show_nl("Evaluate")
            acc = self.evaluate_epoch(epoch=epoch)
            
            is_best = acc > max_acc
            if is_best:
                max_acc = acc
                best_epoch = epoch
            self.logger.show_nl("Current: {:.6f} ({:03d})\tBest: {:.6f} ({:03d})\t".format(
                                acc, epoch, max_acc, best_epoch))

            # Do not save checkpoints in debugging mode
            if not self.debug:
                self._save_checkpoint(
                    self.gan_SW.state_dict(), 
                    self.optimizer_gan.state_dict() if self.ctx['save_optim'] else {}, 
                    (max_acc, best_epoch), epoch, is_best, 'G_SW_'
                )
                self._save_checkpoint(
                    self.gan_WS.state_dict(), 
                    self.optimizer_gan.state_dict() if self.ctx['save_optim'] else {}, 
                    (max_acc, best_epoch), epoch, is_best, 'G_WS_'
                )
                self._save_checkpoint(
                    self.dis_S.state_dict(), 
                    self.optimizer_dis_A.state_dict() if self.ctx['save_optim'] else {}, 
                    (max_acc, best_epoch), epoch, is_best, 'D_S_'
                )
                self._save_checkpoint(
                    self.dis_W.state_dict(), 
                    self.optimizer_dis_B.state_dict() if self.ctx['save_optim'] else {}, 
                    (max_acc, best_epoch), epoch, is_best, 'D_W_'
                )
                self._save_checkpoint(
                    self.model.state_dict(), 
                    self.optimizer.state_dict() if self.ctx['save_optim'] else {}, 
                    (max_acc, best_epoch), epoch, is_best, 'CD_'
                )

            lr, lr_gan = self.adjust_learning_rate(epoch, acc)
        
    def evaluate(self):
        if self.checkpoint: 
            if self._resume_from_checkpoint():
                self.evaluate_epoch(self.start_epoch)
        else:
            self.logger.error("No checkpoint assigned.")
    
    def _resume_from_checkpoint(self):
        self._load_state_dict(self.gan_SW, self.checkpoint_G_SW)
        self._load_state_dict(self.gan_WS, self.checkpoint_G_WS)
        self._load_state_dict(self.dis_S, self.checkpoint_D_S)
        self._load_state_dict(self.dis_W, self.checkpoint_D_W)
        self._load_state_dict(self.model, self.checkpoint)
        return True
    
    def _load_state_dict(self, model, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            self.logger.error("=> No checkpoint was found at '{}'.".format(checkpoint_path))
            return False

        self.logger.show("=> Loading checkpoint '{}'...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = model.state_dict()
        ckp_dict = checkpoint.get('state_dict', checkpoint)
        update_dict = {
            k:v for k,v in ckp_dict.items() 
            if k in state_dict and state_dict[k].shape == v.shape and state_dict[k].dtype == v.dtype
        }
        
        num_to_update = len(update_dict)
        if (num_to_update < len(state_dict)) or (len(state_dict) < len(ckp_dict)):
            if not self.is_training and (num_to_update < len(state_dict)):
                self.logger.error("=> Mismatched checkpoint for evaluation")
                return False
            self.logger.warn("Trying to load a mismatched checkpoint.")
            if num_to_update == 0:
                self.logger.error("=> No parameter is to be loaded.")
                return False
            else:
                self.logger.warn("=> {} params are to be loaded.".format(num_to_update))
            ckp_epoch = -1
        else:
            ckp_epoch = checkpoint.get('epoch', -1)
            if not self.is_training:
                self.start_epoch = ckp_epoch
                self._init_acc_epoch = checkpoint.get('max_acc', (0.0, ckp_epoch))
            elif not self.ctx['anew']:
                self.start_epoch = ckp_epoch+1
                self._init_acc_epoch = checkpoint.get('max_acc', (0.0, ckp_epoch))

        state_dict.update(update_dict)
        model.load_state_dict(state_dict)

        if ckp_epoch == -1:
            self.logger.show("=> Loaded checkpoint '{}'".format(checkpoint_path))
        else:
            self.logger.show("=> Loaded checkpoint '{}' (epoch {}, max_acc {:.4f} at epoch {}).".format(
                checkpoint_path, ckp_epoch, *self._init_acc_epoch
                ))
        
    def _save_checkpoint(self, state_dict, optim_state, max_acc, epoch, is_best, which_model):
        state = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optim_state, 
            'max_acc': max_acc
        } 
        if (epoch+1) % self.track_intvl == 0:
            history_path = self.path(
                'weight', which_model + constants.CKP_COUNTED.format(e=epoch+1), 
                suffix=True
            )
            torch.save(state, history_path)
        # Save latest
        latest_path = self.path(
            'weight', which_model + constants.CKP_LATEST, 
            suffix=True
        )
        torch.save(state, latest_path)
        if is_best:
            shutil.copyfile(
                latest_path, self.path(
                    'weight', which_model + constants.CKP_BEST, 
                    suffix=True
                )
            )
    

    def save_image_gan(self, file_name, image, epoch):
        file_path = osp.join(
            'epoch_{}'.format(epoch),
            self.out_dir,
            file_name
        )
        out_path = self.path(
            'out', file_path,
            suffix=not self.ctx['suffix_off'],
            auto_make=True,
            underline=True
        )
        return cv2.imwrite(out_path, image)

    def _init_trainer(self):
        if self.ctx.get('mix_coeffs') is not None:
            self.criterion = MixedLoss(self.criterion, self.ctx['mix_coeffs'])
        if self.ctx.get('cmb_coeffs') is not None:
            self.criterion = CombinedLoss(self.criterion, self.ctx['cmb_coeffs'])

    def _process_model_out(self, out):
        size = out[0].shape[2:]
        return [F.interpolate(o, size=size).squeeze(1) for o in out]

    def _pred_to_prob(self, pred):
        return F.sigmoid(pred[0])
    
    def _prepare_data(self, t1, t2, tar):
        return t1.to(self.device), t2.to(self.device), tar.float().to(self.device)
    
    def save_image(self, file_name, image, epoch):
        file_path = osp.join(
            'epoch_{}'.format(epoch),
            self.out_dir,
            file_name
        )
        out_path = self.path(
            'out', file_path,
            suffix=not self.ctx['suffix_off'],
            auto_make=True,
            underline=True
        )
        return self._mt_pool.submit(partial(io.imsave, check_contrast=False), out_path, image)

    def _denorm_image(self, x):
        return x*np.asarray(self.ctx['sigma']) + np.asarray(self.ctx['mu'])

    def _process_input_pairs(self, t1, t2):
        vis_band_inds = self.ctx['tb_vis_bands']
        t1 = t1[...,vis_band_inds]
        t2 = t2[...,vis_band_inds]
        if self.ctx['tb_vis_norm'] == '8bit':
            t1 = normalize_8bit(t1)
            t2 = normalize_8bit(t2)
        else:
            t1 = normalize_minmax(t1)
            t2 = normalize_minmax(t2)
        t1 = np.clip(t1, 0.0, 1.0)
        t2 = np.clip(t2, 0.0, 1.0)
        return t1, t2

    def _process_fetched_feat(self, feat):
        feat = normalize_minmax(feat.mean(0))
        feat = quantize(to_array(feat))
        feat = to_pseudo_color(feat)
        return feat

    def _set_fetch_dict(self):
        return dict()

    def count_parameters(trainer, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)