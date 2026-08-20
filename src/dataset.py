import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CellLineDataset(Dataset):
    """ VAE training dataset: single images with their metadata.

    `data_root` is prepended to each row's `relpath` to build the image path,
    so the metadata CSV never has to contain machine-specific absolute paths.
    """
    def __init__(self, df, in_channels, transform, data_root):
        self.df = df
        self.in_channels = in_channels
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.data_root, row['relpath'])
        if self.in_channels == 1:
            image = Image.open(filepath).convert('L')
        else:
            image = Image.open(filepath).convert('RGB')

        image = self.transform(image)
        if self.in_channels == 2:
            image = image[:2, :, :]

        return {
            "image": image,
            "label": row['label'],
            "concentration": row['concentration']
        }


class MicroscopyCellDataset(CellLineDataset):
    """ VAE training dataset with optional cell-line / drug / concentration filtering. """
    def __init__(self, csv_path, cell_line, split, in_channels, transform, data_root,
                 target_concentrations=None, drug=None, filtered_only=False):
        full_df = pd.read_csv(csv_path)
        mask = (full_df['split'] == split)
        if filtered_only:
            mask = mask & (full_df['filtered'].astype(bool) == True)
        if cell_line is not None:
            mask = mask & (full_df['label'] == cell_line)
        if drug is not None:
            mask = mask & (full_df['drug'] == drug)
        if target_concentrations is not None:
            mask = mask & (full_df['concentration'].isin(target_concentrations))
        filtered_df = full_df[mask].reset_index(drop=True)

        super().__init__(df=filtered_df, in_channels=in_channels, transform=transform, data_root=data_root)

        if target_concentrations:
            print(f"[DATA] Filtered for concentrations: {target_concentrations} | Samples: {len(self.df)}")
        if drug is not None:
            print(f"[DATA] Filtered for drug: {drug} | Samples: {len(self.df)}")


class FlowCompoundDataset(Dataset):
    """
    Returns unpaired (source, target) pairs for joint flow matching on a single drug.

    `data_root` is prepended to each row's `relpath` to build the image path,
    so the metadata CSV never has to contain machine-specific absolute paths.

    Mode "cond" (include_dmso_targets=True):
      Source: DMSO cells (concentration == 0) — used for shape/inversion reference
      Target: drug cells at all concentrations, plus DMSO cells at c=0
      Balanced sampling across ALL levels including c=0, so the model learns
      c=0 -> DMSO morphology (required for noise -> drug generation).
    """
    def __init__(
        self,
        csv_path,
        cell_line,
        split,
        in_channels,
        transform,
        data_root,
        drug=None,                    # None = all drugs, str = single drug
        target_concentrations=None,   # None = all non-zero concs, list = explicit
        conc_transform=None,
        filtered_only=True,
        include_dmso_targets=True,    # True = add DMSO cells as targets at c=0
    ):
        df = pd.read_csv(csv_path)

        mask = (df['split'] == split) & (df['label'] == cell_line)
        if filtered_only:
            mask = mask & (df['filtered'].astype(bool) == True)
        df = df[mask].reset_index(drop=True)

        # Drug filter — keep selected drug + DMSO
        if drug is not None:
            drug_mask = (df['drug'] == drug) | (df['drug'].str.upper() == 'DMSO')
            df = df[drug_mask].reset_index(drop=True)

        if target_concentrations is not None:
            conc_mask = (
                df['concentration'].isin(target_concentrations) |
                (df['concentration'] == 0.0)   # always keep DMSO
            )
            df = df[conc_mask].reset_index(drop=True)

        # ── Source pool: always DMSO cells ────────────────────────────────────
        self.df_src = df[df['concentration'] == 0.0].reset_index(drop=True)

        # ── Target pool ───────────────────────────────────────────────────────
        if include_dmso_targets:
            self.df_tgt = df.copy().reset_index(drop=True)
        else:
            self.df_tgt = df[df['concentration'] > 0.0].reset_index(drop=True)

        assert len(self.df_src) > 0, \
            f"No DMSO source cells — check filters (cell_line={cell_line}, split={split})"
        assert len(self.df_tgt) > 0, \
            f"No target cells — check filters (drug={drug}, cell_line={cell_line}, split={split})"

        self.in_channels          = in_channels
        self.transform             = transform
        self.data_root              = data_root
        self.conc_transform        = conc_transform
        self.include_dmso_targets  = include_dmso_targets

        # ── Balanced concentration sampling ───────────────────────────────────
        self.conc_pools = {}
        for conc in sorted(self.df_tgt['concentration'].unique()):
            self.conc_pools[conc] = self.df_tgt[
                self.df_tgt['concentration'] == conc
            ].index.tolist()
        self.unique_concs = sorted(self.conc_pools.keys())

        ref_treated = sorted(c for c in df['concentration'].unique() if c > 0.0)
        if not ref_treated:
            ref_treated = [c for c in self.unique_concs if c > 0.0]
        self._ref_treated = ref_treated

        # ── Uniform concentration map (built from the FULL reference range) ────
        if conc_transform == 'uniform':
            ref_all = [0.0] + ref_treated
            n = len(ref_all)
            self.uniform_map = {
                c: i / (n - 1) if n > 1 else 0.0
                for i, c in enumerate(ref_all)
            }
        else:
            self.uniform_map = None

        # ── DMSO sentinel for log10 (from the FULL reference range) ───────────
        if conc_transform == 'log10':
            # DMSO must be the axis minimum (it is concentration 0). Place it a
            # margin below log10 of the smallest treated dose in the full range.
            min_treated_log = float(np.log10(min(ref_treated)))
            self.dmso_sentinel = min_treated_log - 0.5   # e.g. log10(0.037)-0.5 ~ -1.93
        else:
            self.dmso_sentinel = 0.0

        mode_str = "cond (noise->drug)" if include_dmso_targets else "i2i (DMSO->drug)"
        print(f"[DATA] FlowCompoundDataset | {cell_line} | {split} | "
              f"drug={drug if drug else 'all'} | "
              f"filtered={filtered_only} | mode={mode_str}")
        print(f"       DMSO source : {len(self.df_src):,} cells")
        print(f"       Target pool : {len(self.df_tgt):,} cells total")
        if conc_transform == 'log10':
            print(f"       Transform   : log10 | dmso_sentinel={self.dmso_sentinel:.4f} "
                  f"| ref_treated={ref_treated}")
        print(f"       Concentrations (balanced):")
        for c in self.unique_concs:
            label = "DMSO" if c == 0.0 else f"{c:.3f}uM"
            print(f"         {label:12s} : {len(self.conc_pools[c]):,} cells")
        print(f"       Effective epoch length: "
              f"{len(self.unique_concs)} concs x "
              f"{min(len(p) for p in self.conc_pools.values())} "
              f"= {len(self):,} samples")

    def get_model_conc(self, c_raw, drug=None):
        """Raw concentration -> model conditioning value, applying the active
        transform. drug is accepted for API symmetry but unused (single-drug
        dataset). DMSO (c<=0) maps to the transform's baseline."""
        return self._transform_conc(float(c_raw))

    def __len__(self):
        min_per_conc = min(len(p) for p in self.conc_pools.values())
        return len(self.unique_concs) * min_per_conc

    def _load_image(self, relpath):
        filepath = os.path.join(self.data_root, relpath)
        if self.in_channels == 1:
            img = Image.open(filepath).convert('L')
        else:
            img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        if self.in_channels == 2:
            img = img[:2, :, :]
        return img

    def _transform_conc(self, c):
        if self.conc_transform == 'sqrt':
            return float(c ** 0.5)
        elif self.conc_transform == 'log1p':
            return float(np.log1p(c))
        elif self.conc_transform == 'log10':
            return self.dmso_sentinel if c <= 0 else float(np.log10(c))
        elif self.conc_transform == 'uniform':
            if c in self.uniform_map:
                return float(self.uniform_map[c])
            ref = [0.0] + self._ref_treated
            lo = max([r for r in ref if r <= c], default=ref[0])
            hi = min([r for r in ref if r >= c], default=ref[-1])
            i_lo = ref.index(lo); i_hi = ref.index(hi)
            n = len(ref) - 1
            if hi == lo:
                return float(i_lo / n if n > 0 else 0.0)
            frac = (c - lo) / (hi - lo)
            return float((i_lo + frac * (i_hi - i_lo)) / n if n > 0 else 0.0)
        return float(c)

    def __getitem__(self, idx):
        # ── Balanced concentration selection ──────────────────────────────────
        conc_idx = idx % len(self.unique_concs)
        conc     = self.unique_concs[conc_idx]

        pool    = self.conc_pools[conc]
        tgt_idx = pool[np.random.randint(len(pool))]
        tgt_row = self.df_tgt.iloc[tgt_idx]

        # ── Random DMSO source — always unpaired, used for shape/inversion ────
        src_idx = np.random.randint(len(self.df_src))
        src_row = self.df_src.iloc[src_idx]

        img_src = self._load_image(src_row['relpath'])
        img_tgt = self._load_image(tgt_row['relpath'])

        c = self._transform_conc(float(tgt_row['concentration']))

        return {
            "image_src"    : img_src,
            "image_tgt"    : img_tgt,
            "concentration": torch.tensor(c, dtype=torch.float32),
            "drug"         : tgt_row['drug'],
            "filename_src" : src_row['filename'],
            "filename_tgt" : tgt_row['filename'],
        }
