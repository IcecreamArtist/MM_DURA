import torch
import datetime
import torch.nn.functional as F
from argparse import ArgumentParser
import numpy as np
import os
from dateutil import tz
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
from models.MedicalNet import generate_medical_3DresNet
from models.gene_encoder import Gen_MLP
from models.clinic_encoder import MLP


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MM_DURA(LightningModule):
    def __init__(self,
                 cfg,
                 transforms,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = self.hparams.gpus
        self.seq_len = cfg.seq_len
        self.cognitive_col = cfg.cognitive_col
        num_labels = cfg.n_cog_feat

        # experiments configs
        self.softmax_temperature = cfg.softmax_temperature
        self.intra_ctr = cfg.intra_ctr
        self.inter_ctr = cfg.inter_ctr
        self.intra_ctr_weight = cfg.intra_ctr_weight
        self.inter_ctr_weight_imgcli = cfg.inter_ctr_weight_imgcli
        self.inter_ctr_weight_gencli = cfg.inter_ctr_weight_gencli
        self.reg_weight = cfg.get('reg_weight', 1)
        self.modality = cfg.modality
        self.emb_dim = cfg.img_emb_dim
        self.use_transformer_fusion = "transformer" == cfg.fusion_mode
        self.use_mamba_fusion = "mamba" == cfg.fusion_mode
        self.encode_time = cfg.encode_time

        # modules
        self.gen_encoder = Gen_MLP(n_feats=cfg.n_gen_feat, out_dim=cfg.gen_emb_dim)
        
        self.img_encoder, _ = generate_medical_3DresNet(cfg)
        self.img_encoder.load_state_dict(torch.load(cfg.resnet_pretrain_path)['state_dict'], strict=False)
        print("Load pretrained medical 3D-resnet successfully!")
        self.img_feat_pool = torch.nn.AdaptiveAvgPool3d(output_size=(4,4,4))
        self.img_emb_map = MLP(n_feats=2*4*4*4, out_dim=cfg.img_emb_dim)

        self.cli_encoder = MLP(n_feats=cfg.n_cli_feat, out_dim=cfg.img_emb_dim)
        
        if self.use_transformer_fusion or self.use_mamba_fusion:
            self.mlp = MLP(
                n_feats=cfg.n_cog_feat*self.seq_len+cfg.img_emb_dim, out_dim=cfg.n_cog_feat)
        else:
            self.mlp = MLP(
                n_feats=cfg.img_emb_dim*2, out_dim=cfg.n_cog_feat)
            
        self.loss = MultiTaskLossWrapper(num_tasks=num_labels)

        # results logging
        self.transforms = transforms
        self.training_step_logits = []
        self.training_step_targets = []
        self.validation_step_logits = []
        self.validation_step_targets = []

    def forward(self, batch, batch_idx, split='train'):
        img_seq = batch['image'] # [4, 3, 256, 256, 256]
        time_matrix = batch['Tmatrix']
        next_label = batch['next_label'] # [4,]
        gene = batch['gene']
        cli_seq = batch['clinical'] # [4, 3, 37]
        next_cli = batch['next_clinical'] # [4, 37]
        B, L, _ = cli_seq.shape
        
        cognitive = cli_seq[:, :, self.cognitive_col] # [4, 3, 13]
        cognitive = cognitive.reshape(-1, cognitive.shape[-1])
        
        if 'image' in self.modality:
            img_seq = img_seq.reshape(-1, 1, img_seq.shape[-3], img_seq.shape[-2], img_seq.shape[-1])
            img_feat = self.img_encoder(img_seq) # [4, 4, 128]
            img_feat = self.img_feat_pool(img_feat)
            img_emb = img_feat.reshape(img_seq.shape[0],-1)
            img_emb = self.img_emb_map(img_emb)
            img_emb = img_emb.reshape(B, L, -1)
        else:
            img_emb = torch.zeros(B, L, self.emb_dim).cuda()
        if 'gene' in self.modality:
            gen_emb = self.gen_encoder(gene) # [4, 128]
            gen_emb = gen_emb.view(B, L, -1)
        else:
            gen_emb = torch.zeros(B, L, self.emb_dim).cuda()
        if 'clinic' in self.modality:
            cli_seq = cli_seq.reshape(-1, cli_seq.shape[-1])
            cli_emb = self.cli_encoder(cli_seq) # [4, 4, 128]
            cli_emb = cli_emb.reshape(B, L, -1)
        else:
            cli_emb = torch.zeros(B, L, self.emb_dim).cuda()

        img_emb_2 = torch.nn.functional.normalize(img_emb, dim=-1)
        cli_emb_2 = torch.nn.functional.normalize(cli_emb, dim=-1)
        
        # intra-contrastive learning loss, add time matrix
        if 'image' in self.modality and 'clinic' in self.modality and self.intra_ctr:
            ctr_label = torch.arange(L).type_as(cli_emb_2).long().repeat(B, 1)
            scores = img_emb_2.bmm(cli_emb_2.transpose(1, 2))
            if self.encode_time:
                scores /= (time_matrix + 0.5) # TODO: check range time
            else:
                scores /= self.softmax_temperature
            scores1 = scores.transpose(1, 2)
            loss0 = 0
            loss1 = 0
            for i in range(B):
                loss0 += F.cross_entropy(scores[i], ctr_label[i])
                loss1 += F.cross_entropy(scores1[i], ctr_label[i])
            intra_ctr_loss = (loss0 + loss1) / (B * 2)
        else:
            intra_ctr_loss = 0.0

        # inter-contrastive learning loss (img cli_img_cols).
        img_emb_3 = img_emb.view(B, -1)
        cli_emb_3 = cli_emb.view(B, -1)
        img_emb_3 = torch.nn.functional.normalize(img_emb_3, dim=-1) # TODO: ?
        cli_emb_3 = torch.nn.functional.normalize(cli_emb_3, dim=-1)
        if 'image' in self.modality and 'clinic' in self.modality and self.inter_ctr:
            ctr_label = torch.arange(B).type_as(cli_emb_3).long()
            scores = img_emb_3.mm(cli_emb_3.t())
            scores /= self.softmax_temperature
            scores1 = scores.transpose(0, 1)
            loss0 = F.cross_entropy(scores, ctr_label)
            loss1 = F.cross_entropy(scores1, ctr_label)
            inter_ctr_loss_imgcli = (loss0 + loss1) / 2
        else:
            inter_ctr_loss_imgcli = 0.0
        
        # inter-contrastive learning loss (gen cli_gen_cols).
        cli_emb_4 = cli_emb.view(B, -1)
        gen_emb_4 = gen_emb.view(B, -1)
        cli_emb_4 = torch.nn.functional.normalize(cli_emb_4, dim=-1) # TODO: ?
        gen_emb_4 = torch.nn.functional.normalize(gen_emb_4, dim=-1)
        if 'gene' in self.modality and 'clinic' in self.modality and self.inter_ctr:
            ctr_label = torch.arange(B).type_as(cli_emb_4).long()
            scores = gen_emb_4.mm(cli_emb_4.t())
            scores /= self.softmax_temperature
            scores1 = scores.transpose(0, 1)
            loss0 = F.cross_entropy(scores, ctr_label)
            loss1 = F.cross_entropy(scores1, ctr_label)
            inter_ctr_loss_gencli = (loss0 + loss1) / 2
        else:
            inter_ctr_loss_gencli = 0.0
        
        embeddings = torch.cat((img_emb.reshape(-1, img_emb.shape[-1]),gen_emb.reshape(-1, gen_emb.shape[-1])),dim=1)
        
        logits = self.mlp(embeddings)
        targets = cognitive
        reg_loss = self.loss(logits, targets)

        loss = self.intra_ctr_weight * intra_ctr_loss \
                + self.inter_ctr_weight_imgcli * inter_ctr_loss_imgcli \
                + self.inter_ctr_weight_gencli * inter_ctr_loss_gencli \
                + self.reg_weight * reg_loss
        
        return loss, intra_ctr_loss, inter_ctr_loss_imgcli, inter_ctr_loss_gencli, reg_loss

    
    def training_step(self, batch, batch_idx):
        loss, intra_ctr_loss, inter_ctr_loss_imgcli, inter_ctr_loss_gencli, reg_loss = self(
            batch, batch_idx, "train")
        
        log = {
            "train_loss": loss,
            "train_intra_ctr_loss": intra_ctr_loss,
            "train_inter_ctr_loss_imgcli": inter_ctr_loss_imgcli,
            "train_inter_ctr_loss_gencli": inter_ctr_loss_gencli,
            "train_reg_loss": reg_loss
        }

        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, intra_ctr_loss, inter_ctr_loss_imgcli, inter_ctr_loss_gencli, reg_loss = self(
            batch, batch_idx, "valid")
        
        log = {
            "val_loss": loss,
            "val_intra_ctr_loss": intra_ctr_loss,
            "val_inter_ctr_loss_imgcli": inter_ctr_loss_imgcli,
            "val_inter_ctr_loss_gencli": inter_ctr_loss_gencli,
            "val_reg_loss": reg_loss
        }

        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        
        return loss
        
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
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
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
        parser.add_argument("--cfg_path", type=str, default='config.yaml')
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
    parser = MM_DURA.add_model_specific_args(parser)
    args = parser.parse_args()

    config_path = os.path.join(
        BASE_DIR, args.cfg_path)
    config = OmegaConf.load(config_path)

    args.deterministic = True
    args.max_epochs = 30

    # seed
    seed_everything(args.seed)

    datatransforms = DataTransforms(
        config.dataset.clinical_numerical_features, config.dataset.clinical_categorical_features,
        config.dataset.cognitive_scores)
    datamodule = DataModule(
        ADNI_DataSet, config.dataset, None, datatransforms, args.batch_size, args.num_workers)

    model = MM_DURA(transforms=datatransforms, cfg=config.model, **args.__dict__)

    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("_%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../data/ckpts/pretrain/{args.experiment_name}_{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=3),
        # EarlyStopping(monitor="val_loss", min_delta=0.,
        #               patience=5, verbose=False, mode="min")
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
    model.training_steps = model.num_training_steps(trainer, datamodule)
    
    trainer.fit(model, datamodule=datamodule)
    # with open(os.path.join(logger_dir,'preprocessor.pkl'), 'wb') as f:
    #     pickle.dump(datatransforms, f)
    # trainer.test(model, datamodule)

if __name__ == "__main__":
    cli_main()