import torch
import datetime
import torch.nn.functional as F
from argparse import ArgumentParser
import numpy as np
import torch.nn as nn
import os
from dateutil import tz
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from datamodule.dataset import ADNI_DataSet
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from datamodule.data_module import DataModule
from datamodule.transforms import DataTransforms
from models.loss import MultiTaskLossWrapper
from models.transformer import CrossModalTransformer
from utils.metric import return_all_metric
from mm_dura.pretrain import MM_DURA


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Finetuner(LightningModule):
    def __init__(self,
                 cfg,
                 transforms,
                 img_encoder: nn.Module,
                 img_feat_pool: nn.Module,
                 img_emb_map: nn.Module,
                 cli_encoder: nn.Module,
                 gen_encoder: nn.Module,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['img_encoder','cli_encoder','gen_encoder','img_feat_pool','img_emb_map'])
        self.seq_len = cfg.seq_len
        self.cognitive_col = cfg.cognitive_col
        num_labels = cfg.n_cog_feat

        # experiments configs
        self.modality = cfg.modality
        self.emb_dim = cfg.img_emb_dim
        self.fusion_mode = cfg.fusion_mode
        self.gen_emb_dim = cfg.gen_emb_dim
        self.clinical_col = cfg.clinical_col
        self.use_time = cfg.time
        self.cfg = cfg

        # modules
        self.img_encoder = img_encoder
        self.img_feat_pool = img_feat_pool
        self.img_emb_map = img_emb_map
        self.cli_encoder = cli_encoder
        self.gen_encoder = gen_encoder
        # froze
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        for param in self.cli_encoder.parameters():
            param.requires_grad = False
        for param in self.gen_encoder.parameters():
            param.requires_grad = False

        # fusion and regression head
        if self.fusion_mode == 'concate':
            self.linear_project = torch.nn.Linear(
                cfg.seq_len*2*cfg.img_emb_dim+cfg.gen_emb_dim, cfg.seq_len*cfg.img_emb_dim)
        elif self.fusion_mode == 'cross_attn':
            self.fusion_block = CrossModalTransformer(
                emb_dim=cfg.img_emb_dim, num_heads = 8, 
                dim_feedforward=2048, dropout=0.1
            )    
        else:
            raise NotImplementedError
        
        self.encoder_RNN = nn.LSTM(cfg.img_emb_dim + len(self.cognitive_col), cfg.img_emb_dim, num_layers=1, batch_first=True)
        # linear encoder blocks
        self.ff_layers = nn.ModuleList()
        for _ in range(2):
            self.ff_layers.append(nn.Linear(cfg.img_emb_dim, cfg.img_emb_dim))
        self.ff_layers.append(nn.Linear(cfg.img_emb_dim, cfg.img_emb_dim + len(self.cognitive_col)))
        
        self.loss = MultiTaskLossWrapper(num_tasks=num_labels)

        # results logging
        self.transforms = transforms
        self.training_step_logits = []
        self.training_step_targets = []
        self.training_step_next_label = []
        self.validation_step_logits = []
        self.validation_step_targets = []
        self.validation_step_next_label = []
    
    # freeze the img encoder
    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.img_encoder.eval()

    def forward(self, batch):
        img_seq = batch['image']
        gene = batch['gene']
        cli_seq = batch['clinical']
        next_cli = batch['next_clinical']
        next_label = batch['next_label']
        B, L, _ = cli_seq.shape
        next_cog = next_cli[:, self.cognitive_col]
        current_cog = cli_seq[:,:,self.cognitive_col]

        img_seq = img_seq.reshape(-1, 1, img_seq.shape[-3], img_seq.shape[-2], img_seq.shape[-1])
        img_emb = self.img_encoder(img_seq) # [4, 4, 128]
        img_feat = self.img_encoder(img_seq) # [4, 4, 128]
        img_feat = self.img_feat_pool(img_feat)
        img_emb = img_feat.reshape(img_seq.shape[0],-1)
        img_emb = self.img_emb_map(img_emb)
        img_emb = img_emb.reshape(B, L, -1)
        
        if 'gene' in self.modality:
            gen_emb = self.gen_encoder(gene) # [4, 3x128]
        else:
            gen_emb = torch.zeros(B, self.gen_emb_dim).cuda()
        if 'clinic' in self.modality:
            cli_seq = cli_seq[:,:,self.clinical_col]
            cli_seq = cli_seq.reshape(-1, cli_seq.shape[-1])
            cli_emb = self.cli_encoder(cli_seq) # [4, 4, 128]
            cli_emb = cli_emb.reshape(B, L, -1)
        else:
            cli_emb = torch.zeros(B, L, self.emb_dim).cuda()

        # fusion
        if self.fusion_mode == 'cross_attn':
            if 'image' in self.modality and 'clinic' in self.modality and 'gene' in self.modality:
                embeddings = self.fusion_block.fuse3(img_emb, cli_emb, gen_emb.view(B, L, -1))
            elif 'image' in self.modality and 'clinic' in self.modality:
                embeddings = self.fusion_block.fuse2(img_emb, cli_emb)
            elif 'clinic' in self.modality and 'gene' in self.modality:
                embeddings = self.fusion_block.fuse2(gen_emb.view(B, L, -1), cli_emb)
            elif 'image' in self.modality:
                embeddings = img_emb
            elif 'clinic' in self.modality:
                embeddings = cli_emb
            elif 'gene' in self.modality:
                embeddings = gen_emb.view(B, L, -1)
            else:
                raise NotImplementedError
            
        elif self.fusion_mode == 'concate':
            embeddings = torch.cat([img_emb.view(B, -1), cli_emb.view(B, -1), gen_emb], dim=-1)
            embeddings = self.linear_project(embeddings)
            embeddings = embeddings.view(B, L, -1)
            
        embeddings_input = torch.concat((embeddings,current_cog),dim=2)
        all_encoder_outputs = []
        input_temp = embeddings_input[:,0,:].unsqueeze(1)
        
        encoder_outputs, (encoder_state_h, encoder_state_c) = self.encoder_RNN(input_temp)
        states = (encoder_state_h, encoder_state_c) 
        
        for layer_num in range(2):
            encoder_outputs = self.ff_layers[layer_num](encoder_outputs)
        encoder_single_output = self.ff_layers[-1](encoder_outputs)+input_temp    # x'_t = W * h_t + x_{t-1}
        
        all_encoder_outputs.append(encoder_single_output)
        
        for i in range(1,self.seq_len):
            input_temp =  embeddings_input[:,i,:].unsqueeze(1)       # n x 1 x (latent_dim + score_dim)
            encoder_outputs, (encoder_state_h, encoder_state_c) = self.encoder_RNN(input_temp,states)   # h_t = LSTM(x_t,0)
            states = (encoder_state_h, encoder_state_c)
            for layer_num in range(2):
                encoder_outputs = self.ff_layers[layer_num](encoder_outputs)
            encoder_single_output = self.ff_layers[-1](encoder_outputs) + input_temp

            if i != self.seq_len-1:
                all_encoder_outputs.append(encoder_single_output)
        
        encoder_outputs = torch.cat(all_encoder_outputs, dim=1)
        inputs = encoder_single_output
        
        decoder_outputs, (decoder_state_h, decoder_state_c) = self.encoder_RNN(inputs,states)     # h_t = LSTM(x_t,0)
        states = (decoder_state_h, decoder_state_c) 
            
        for layer_num in range(2):
            decoder_outputs = self.ff_layers[layer_num](decoder_outputs)
        outputs = self.ff_layers[-1](decoder_outputs) + inputs
        logits = outputs.reshape(B, -1)[:,self.emb_dim:]
            
        
        targets = next_cog
        loss = self.loss(logits, targets)

        return targets, logits, loss, next_label

    def training_step(self, batch, batch_idx):
        targets, logits, loss, next_label = self(
            batch)

        log = {
            "train_loss": loss,
        }

        self.training_step_logits.append(logits) # [B,N]
        self.training_step_targets.append(targets)
        self.training_step_next_label.append(next_label)

        self.log_dict(log, batch_size=self.hparams.batch_size,
                        sync_dist=True, prog_bar=True)

        return loss
    
    
    def on_train_epoch_end(self):
        if len(self.training_step_targets) > 0:
            all_targets = torch.stack(self.training_step_targets, dim=0).cpu().detach().numpy()
            all_logits = torch.stack(self.training_step_logits, dim=0).cpu().detach().numpy()
            all_next_label = torch.stack(self.training_step_next_label, dim=0).cpu().detach().numpy()

            all_targets = all_targets.reshape(-1, all_targets.shape[-1])
            all_logits = all_logits.reshape(-1, all_logits.shape[-1])
            all_next_label = all_next_label.reshape(-1, 1)

            all_targets = self.transforms.denormalize_cog(all_targets)
            all_logits = self.transforms.denormalize_cog(all_logits)
            total_targets = np.transpose(all_targets, (1,0))
            total_logits = np.transpose(all_logits, (1,0))

            total_next_label = all_next_label.reshape(1,-1)
            
            N, _ = total_targets.shape
            log = {}
            for i in range(N):
                MAE, r2, pccs, rmse_value, wR, _, _, _, _, _ = return_all_metric(
                    total_targets[i], total_logits[i], total_next_label)
                log.update({
                        'train_'+str(i)+'_MAE': MAE,
                        'train_'+str(i)+'_r2': r2,
                        'train_'+str(i)+'_pccs': pccs,
                        'train_'+str(i)+'_rmse': rmse_value,
                        'train_'+str(i)+'_wR': wR
                })
        
            self.log_dict(log, batch_size=self.hparams.batch_size,
                        sync_dist=True)
            self.training_step_logits.clear()
            self.training_step_targets.clear()
            self.training_step_next_label.clear()

    def validation_step(self, batch, batch_idx):
        targets, logits, loss, next_label = self(
            batch)
        
        log = {
            "val_loss": loss,
        }

        self.validation_step_logits.append(logits) # [B,N]
        self.validation_step_targets.append(targets)
        self.validation_step_next_label.append(next_label)

        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.validation_step_targets) > 0:
            all_targets = torch.stack(self.validation_step_targets, dim=0).cpu().detach().numpy()
            all_logits = torch.stack(self.validation_step_logits, dim=0).cpu().detach().numpy()
            all_next_label = torch.stack(self.validation_step_next_label, dim=0).cpu().detach().numpy()

            all_targets = all_targets.reshape(-1, all_targets.shape[-1])
            all_logits = all_logits.reshape(-1, all_logits.shape[-1])
            all_next_label = all_next_label.reshape(-1, 1)

            all_targets = self.transforms.denormalize_cog(all_targets)
            all_logits = self.transforms.denormalize_cog(all_logits)
            total_targets = np.transpose(all_targets, (1,0))
            total_logits = np.transpose(all_logits, (1,0))
            total_next_label = np.array(all_next_label)

            total_next_label = all_next_label.reshape(1,-1)
            
            N, _ = total_targets.shape
            log = {}
            for i in range(N):
                MAE, r2, pccs, rmse_value, wR, MAE_CI95, R2_CI95, pcc_CI95, rmse_CI95, wR_CI95 = return_all_metric(
                    total_targets[i], total_logits[i], total_next_label)
                log.update({
                        'val_'+str(i)+'_MAE': MAE,
                        'val_'+str(i)+'_r2': r2,
                        'val_'+str(i)+'_pccs': pccs,
                        'val_'+str(i)+'_rmse': rmse_value,
                        'val_'+str(i)+'_wR': wR,
                        'val_'+str(i)+'_MAE_CI95l': MAE_CI95[0],
                        'val_'+str(i)+'_MAE_CI95h': MAE_CI95[1],
                        'val_'+str(i)+'_R2_CI95l': R2_CI95[0],
                        'val_'+str(i)+'_R2_CI95h': R2_CI95[1],
                        'val_'+str(i)+'_pcc_CI95l': pcc_CI95[0],
                        'val_'+str(i)+'_pcc_CI95h': pcc_CI95[1],
                        'val_'+str(i)+'_RMSE_CI95l': rmse_CI95[0],
                        'val_'+str(i)+'_RMSE_CI95h': rmse_CI95[1],
                        'val_'+str(i)+'_wR_CI95l': wR_CI95[0],
                        'val_'+str(i)+'_wR_CI95h': wR_CI95[1]
                })
        
            self.log_dict(log, batch_size=self.hparams.batch_size,
                        sync_dist=True)
            self.validation_step_logits.clear()
            self.validation_step_targets.clear()
            self.validation_step_next_label.clear()
    
    def test_step(self, batch, batch_idx):
        targets, logits, loss, next_label = self(
            batch)
        targets = self.transforms.denormalize_cog(targets)
        logits = self.transforms.denormalize_cog(logits)

        next_label = next_label.detach()
        if next_label.is_cuda:
            next_label = next_label.cpu()
        next_label = next_label.numpy().reshape(1,-1)
        
        log = {
            "test_loss": loss,
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        
        return targets, logits, next_label
    

    def test_epoch_end(self, all_outputs):
        # each
        total_targets = [[] for _ in range(len(self.cognitive_col))]
        total_logits = [[] for _ in range(len(self.cognitive_col))]
        log = {}
        for tar, logi, next_label in all_outputs:
            # tar is a numpy array with shape [B, cog_col]
            B, cog_col = tar.shape
            for b in range(B):
                for i in range(cog_col):
                    total_targets[i].append(tar[b,i])
                    total_logits[i].append(logi[b,i])

        total_next_label = [batch[2] for batch in all_outputs]
        total_targets = np.array(total_targets)
        total_logits = np.array(total_logits)
        total_next_label = np.hstack(total_next_label)
        N, _ = total_targets.shape
        all_ADAS_MAE = []
        all_ADAS_r2 = []
        all_ADAS_pccs = []
        all_RAVLT_MAE = []
        all_RAVLT_r2 = []
        all_RAVLT_pccs = []
        ADAS_ids = [1,2,3]
        RAVLT_ids = [5,6,7,8]
        for i in range(N):
            MAE, r2, pccs, rmse_value, wR, MAE_CI95, R2_CI95, pcc_CI95, rmse_CI95, wR_CI95 = return_all_metric(
                total_targets[i], total_logits[i], total_next_label)
            log.update({
                    str(i)+'_MAE': MAE,
                    str(i)+'_R2': r2,
                    str(i)+'_pccs': pccs,
                    str(i)+'_MAE_CI95l': MAE_CI95[0],
                    str(i)+'_MAE_CI95h': MAE_CI95[1],
                    str(i)+'_R2_CI95l': R2_CI95[0],
                    str(i)+'_R2_CI95h': R2_CI95[1],
                    str(i)+'_pcc_CI95l': pcc_CI95[0],
                    str(i)+'_pcc_CI95h': pcc_CI95[1],
                    str(i)+'_RMSE': rmse_value,
                    str(i)+'_RMSE_CI95l': rmse_CI95[0],
                    str(i)+'_RMSE_CI95h': rmse_CI95[1],
                    str(i)+'_wR': wR,
                    str(i)+'_wR_CI95l': wR_CI95[0],
                    str(i)+'_wR_CI95h': wR_CI95[1]
                    # str(i)+'_Tstatistic': t_statistic,
                    # str(i)+'_Pvalue_Ttest': p_value_t_test
            })

            if i in ADAS_ids:
                all_ADAS_MAE.append(MAE)
                all_ADAS_r2.append(r2)
                all_ADAS_pccs.append(pccs)
            if i in RAVLT_ids:
                all_RAVLT_MAE.append(MAE)
                all_RAVLT_r2.append(r2)
                all_RAVLT_pccs.append(pccs)  
        
        log.update({
                    'avg_ADAS_MAE': sum(all_ADAS_MAE) / len(all_ADAS_MAE),
                    'avg_ADAS_r2': sum(all_ADAS_r2) / len(all_ADAS_r2),
                    'avg_ADAS_pccs': sum(all_ADAS_pccs) / len(all_ADAS_pccs),
                    'avg_RAVLT_MAE': sum(all_RAVLT_MAE) / len(all_RAVLT_MAE),
                    'avg_RAVLT_r2': sum(all_RAVLT_r2) / len(all_RAVLT_r2),
                    'avg_RAVLT_pccs': sum(all_RAVLT_pccs) / len(all_RAVLT_pccs),
            })

        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True)
            
    def predict_step(self, batch, batch_idx):
        targets, _,  = self(
            batch, batch_idx, "predict")
        targets = self.transforms.denormalize_cog(targets)
        return targets
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.1)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return optimizer
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--learning_rate", type=float, default=2e-4)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="vanilla")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_pct", type=float, default=1)
        parser.add_argument("--cfg_path", type=str, default='finetune_config.yaml')
        parser.add_argument("--weight_path", type=str, default='')
        parser.add_argument("--finetuned_model_path", type=str, default='None')
    
        return parser
    
    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_devices)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs

def cli_main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Finetuner.add_model_specific_args(parser)
    args = parser.parse_args()

    config_path = os.path.join(
        BASE_DIR, args.cfg_path
    )
    config = OmegaConf.load(config_path)

    args.deterministic = True
    args.max_epochs = 30

    seed_everything(args.seed)

    datatransforms = DataTransforms(
        config.dataset.clinical_numerical_features, config.dataset.clinical_categorical_features,
        config.dataset.cognitive_scores)
    datamodule = DataModule(
        ADNI_DataSet, config.dataset, None, datatransforms, args.batch_size, args.num_workers)

    if args.weight_path:
        model = MM_DURA.load_from_checkpoint(args.weight_path, strict=False)
        print('checkpoint loaded!')
    else:
        model = MM_DURA(transforms=datatransforms, cfg=config.model, **args.__dict__)

    args.img_encoder = model.img_encoder
    args.img_feat_pool = model.img_feat_pool
    args.img_emb_map = model.img_emb_map
    args.gen_encoder = model.gen_encoder
    args.cli_encoder = model.cli_encoder
    
    tuner = Finetuner(transforms=datatransforms, cfg=config.model, **args.__dict__)

    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("_%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/finetune/{args.experiment_name}_{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=2)
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="MM_DURA", save_dir=logger_dir, name=args.experiment_name+extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger,
        precision=16)
    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    if args.finetuned_model_path == "None":
        trainer.fit(tuner, datamodule=datamodule)
        trainer.test(tuner, datamodule, ckpt_path="best")
    else:
        print("========================================================")
        print("Directly inference MM_DURA model.")
        print("========================================================")
        trainer.test(tuner, datamodule, ckpt_path=args.finetuned_model_path)

if __name__ == '__main__':
    cli_main()
