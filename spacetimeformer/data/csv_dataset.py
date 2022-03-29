import random
from typing import List
import os
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

import spacetimeformer as stf


class CSVTimeSeries:
    def __init__(
        self,
        data_path: str,
        target_cols: List[str],
        read_csv_kwargs={},
        val_split: float = 0.2,
        test_split: float = 0.15,
    ):
        self.data_path = data_path


        df = None


        # for i in range(1, 4):
        #     assert os.path.exists(f'{self.data_path}-{i}.csv')
        #     raw_df = pd.read_csv(
        #         f'{self.data_path}-{i}.csv', 
        #         delimiter=",",
        #         **read_csv_kwargs,
        #     )
            
        # read the file and do some timestamp conversions
        for root,dirs,files in os.walk(self.data_path):
            for file in files:
                assert os.path.exists(f'{self.data_path}/{file}')

                
                raw_df = pd.read_csv(
                    f'{self.data_path}/{file}', 
                    delimiter=",",
                    **read_csv_kwargs,
                )
                # format="%Y-%m-%d %H:%M"
                # 
            # 
                # 
            # 
                # 
            # 
                # 
                time_df = pd.to_datetime(raw_df["timestamp"], format='%Y-%m-%d %H:%M:%S.%f')
                
                time_feat_df = stf.data.timefeatures.time_features(time_df, raw_df)
                if df is None:
                    df = time_feat_df.copy(deep=True)
                    # df = pd.DataFrame(time_feat_df, columns=["timestamp"])
                else:
                    df = pd.concat([df, time_feat_df])

                assert (df["timestamp"] > pd.Timestamp.min.tz_localize('utc')).all()
                assert (df["timestamp"] < pd.Timestamp.max.tz_localize('utc')).all()

        # Train/Val/Test Split using holdout approach #

        def mask_intervals(mask, intervals, cond):
            for (interval_low, interval_high) in intervals:
                if interval_low is None:
                    interval_low = df["timestamp"].iloc[0].year
                if interval_high is None:
                    interval_high = df["timestamp"].iloc[-1].year
                mask[
                    (df["timestamp"] >= interval_low) & (df["timestamp"] <= interval_high)
                ] = cond
            return mask

        test_cutoff = len(time_df) - round(test_split * len(time_df))
        val_cutoff = test_cutoff - round(val_split * len(time_df))

        val_interval_low = time_df.iloc[val_cutoff]
        val_interval_high = time_df.iloc[test_cutoff - 1]
        val_intervals = [(val_interval_low, val_interval_high)]

        test_interval_low = time_df.iloc[test_cutoff]
        test_interval_high = time_df.iloc[-1]
        test_intervals = [(test_interval_low, test_interval_high)]

        train_mask = df["timestamp"] > pd.Timestamp.min.tz_localize('utc')
        val_mask = df["timestamp"] > pd.Timestamp.max.tz_localize('utc')
        test_mask = df["timestamp"] > pd.Timestamp.max.tz_localize('utc')
        train_mask = mask_intervals(train_mask, test_intervals, False)
        train_mask = mask_intervals(train_mask, val_intervals, False)
        val_mask = mask_intervals(val_mask, val_intervals, True)
        test_mask = mask_intervals(test_mask, test_intervals, True)

        if (train_mask == False).all():
            print(f"No training data detected for file {data_path}")

        self.scale_feats = ['val_0', 'val_1', 'val_2', 'val_3']

        self._train_data = df[train_mask]
        self._scaler_0 = StandardScaler()
        self._scaler_1 = StandardScaler()
        self._scaler_2 = StandardScaler()
        self._scaler_3 = StandardScaler()
        # self._scaler = StandardScaler()
        self.target_cols = target_cols

        # self._scaler = self._scaler.fit(self._train_data[target_cols].values)
        #self._scaler = self._scaler.fit(self._train_data[self.scale_feats].values)

        self._scaler_0 = self._scaler_0.fit(self._train_data['val_0'].values.reshape(-1, 1))
        self._scaler_1 = self._scaler_1.fit(self._train_data['val_1'].values.reshape(-1, 1))
        self._scaler_2 = self._scaler_2.fit(self._train_data['val_2'].values.reshape(-1, 1))
        self._scaler_3 = self._scaler_3.fit(self._train_data['val_3'].values.reshape(-1, 1))

        self._train_data = self.apply_scaling_df(df[train_mask])
        self._val_data = self.apply_scaling_df(df[val_mask])
        self._test_data = self.apply_scaling_df(df[test_mask])

    def get_slice(self, split, start, stop, skip):
        assert split in ["train", "val", "test"]
        if split == "train":
            return self.train_data.iloc[start:stop:skip]
        elif split == "val":
            return self.val_data.iloc[start:stop:skip]
        else:
            return self.test_data.iloc[start:stop:skip]

    def apply_scaling_df(self, df):
        scaled = df.copy(deep=True)

        # scaled[self.target_cols] = (
        #     df[self.target_cols].values - self._scaler.mean_
        # ) / self._scaler.scale_

        # scaled[self.scale_feats] = (
        #     df[self.scale_feats].values - self._scaler.mean_
        # ) / self._scaler.scale_


        scaled['val_0'] = (
            df['val_0'].values - self._scaler_0.mean_
        ) / self._scaler_0.scale_
        
        scaled['val_1'] = (
            df['val_1'].values - self._scaler_1.mean_
        ) / self._scaler_1.scale_

        scaled['val_2'] = (
            df['val_2'].values - self._scaler_2.mean_
        ) / self._scaler_2.scale_

        scaled['val_3'] = (
            df['val_3'].values - self._scaler_3.mean_
        ) / self._scaler_3.scale_

        return scaled

    def apply_scaling(self, array):
         return (array - self._scaler.mean_) / self._scaler.scale_

    def reverse_scaling_df(self, df):
        scaled = df.copy(deep=True)
        # scaled[self.target_cols] = (
        #     df[self.target_cols] * self._scaler.scale_
        # ) + self._scaler.mean_


        # scaled[self.scale_feats] = (
        #     df[self.scale_feats] * self._scaler.scale_
        # ) + self._scaler.mean_

        scaled['val_0'] = (
            df['val_0'] * self._scaler_0.scale_
        ) + self._scaler_0.mean_

        scaled['val_1'] = (
            df['val_1'] * self._scaler_1.scale_
        ) + self._scaler_1.mean_

        scaled['val_2'] = (
            df['val_2'] * self._scaler_2.scale_
        ) + self._scaler_2.mean_

        scaled['val_3'] = (
            df['val_3'] * self._scaler_3.scale_
        ) + self._scaler_3.mean_

        return scaled

    def reverse_scaling(self, array):
        val_0, val_1, val_2, val_3, sourceType, id, event = torch.split(array, 1, dim=-1) #.cpu().numpy()

        val_0 = (val_0.cpu().numpy() * self._scaler_0.scale_) + self._scaler_0.mean_
        val_1 = (val_1.cpu().numpy() * self._scaler_1.scale_) + self._scaler_1.mean_
        val_2 = (val_2.cpu().numpy() * self._scaler_2.scale_) + self._scaler_2.mean_
        val_3 = (val_3.cpu().numpy() * self._scaler_3.scale_) + self._scaler_3.mean_

        # return (array * self._scaler.scale_) + self._scaler.mean_
        c = np.concatenate([val_0, val_1, val_2, val_3, sourceType.cpu().numpy(), id.cpu().numpy(), event.cpu().numpy()], axis=-1)
        return c
        # return torch.cat((val_0, val_1, val_2, val_3, sourceType, id, event), dim=-1).numpy()

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    def length(self, split):
        return {
            "train": len(self.train_data),
            "val": len(self.val_data),
            "test": len(self.test_data),
        }[split]

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="auto")


class CSVTorchDset(Dataset):
    def __init__(
        self,
        csv_time_series: CSVTimeSeries,
        split: str = "train",
        context_points: int = 128,
        target_points: int = 32,
        time_resolution: int = 1,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.series = csv_time_series
        self.context_points = context_points
        self.target_points = target_points
        self.time_resolution = time_resolution

        self._slice_start_points = [
            i
            for i in range(
                0,
                self.series.length(split)
                + time_resolution * (-target_points - context_points)
                + 1,
            )
        ]
        random.shuffle(self._slice_start_points)
        self._slice_start_points = self._slice_start_points

    def __len__(self):
        return len(self._slice_start_points)

    def _torch(self, *np_arrays):
        t = []
        for x in np_arrays:
            t.append(torch.from_numpy(x).float())
        return tuple(t)

    def __getitem__(self, i):
        start = self._slice_start_points[i]
        series_slice = self.series.get_slice(
            self.split,
            start=start,
            stop=start
            + self.time_resolution * (self.context_points + self.target_points),
            skip=self.time_resolution,
        ).drop(columns=["timestamp"])
        ctxt_slice, trgt_slice = (
            series_slice.iloc[: self.context_points],
            series_slice.iloc[self.context_points :],
        )

        # "sourceType",
        # "ID",
        ctxt_x = ctxt_slice[
            ctxt_slice.columns.difference(self.series.target_cols)
        ].values
        ctxt_y = ctxt_slice[self.series.target_cols].values

        trgt_x = trgt_slice[
            trgt_slice.columns.difference(self.series.target_cols)
        ].values
        trgt_y = trgt_slice[self.series.target_cols].values

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)

    @classmethod
    def add_cli(self, parser):
        parser.add_argument(
            "--context_points",
            type=int,
            default=128,
            help="number of previous timesteps given to the model in order to make predictions",
        )
        parser.add_argument(
            "--target_points",
            type=int,
            default=32,
            help="number of future timesteps to predict",
        )
        parser.add_argument(
            "--time_resolution",
            type=int,
            default=1,
        )


if __name__ == "__main__":
    test = CSVTimeSeries(
        "/p/qdatatext/jcg6dn/asos/temperature-v1.csv",
        ["ABI", "AMA", "ACT", "ALB", "JFK", "LGA"],
    )
    breakpoint()
    dset = CSVTorchDset(test)
    base = dset[0][0]
    for i in range(len(dset)):
        assert base.shape == dset[i][0].shape
    breakpoint()
