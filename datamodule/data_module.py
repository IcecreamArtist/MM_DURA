import pytorch_lightning as pl
from torch.utils.data import DataLoader
import SimpleITK as sitk
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, cfg, collate_fn, transforms, batch_size, num_workers):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cfg = cfg

    def train_dataloader(self):
        
        dataset = self.dataset(
            cfg=self.cfg, mode="train", transform=self.transforms)
        
        self.transforms = dataset.transform # updated after fitting

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        dataset = self.dataset(
            cfg=self.cfg, mode="val", transform=self.transforms)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        dataset = self.dataset(
            cfg=self.cfg, mode="test", transform=self.transforms)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
