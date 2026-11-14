# Solar Filament Segmentation Challenge 2026: U-Net ensemble

Divyansh Kumar, The Maharaja Sayajirao University of Baroda, Vadodara, India

Solution to the [Solar Filament Segmentation Challenge 2026](https://kaggle.com/competitions/filament-segmentation-2026): instance segmentation of solar filaments in GONG H-alpha images, scored with Panoptic Quality (PQ).

## Reproduce the submitted predictions

```
pip install -r requirements.txt
jupyter notebook pipeline.ipynb
```

`pipeline.ipynb` loads the 10 stored models from `weights/`, predicts the 180 test photos, applies the post-processing rule and writes `outputs/submission.csv`, then compares it with the submitted file `submission/ens10b18_r40.csv`. Set `TEST_DIR` in the notebook (or as an environment variable) to the competition's `test/test_images` folder; on Kaggle it is found automatically. A CUDA GPU is recommended (about 3 minutes); the CPU also works, more slowly. For a CUDA build of PyTorch, see [pytorch.org](https://pytorch.org/get-started/locally/).

## Method

| Step | Details |
|---|---|
| Pre-processing | Photos shrunk to 1024 x 1024; divided by the median disk brightness, clipped to 0..2.5, minus 1 |
| Models | Two U-Net groups, 5 models each (one per fold): ResNet-34 encoder with our decoder; EfficientNetV2-S encoder (segmentation-models-pytorch). ImageNet-pretrained |
| Training | Random 512 x 512 windows (70% on filaments), flips, 90-degree rotations, brightness jitter; soft target = mean of all annotators; BCE + Dice; AdamW 3e-4, one-cycle; batch 18, 8000 steps |
| Prediction | Mirror averaging; 5 fold models averaged per group; 2 groups averaged |
| Post-processing | Cut at 0.40 (2048 px scale); 8-connected blobs kept if area >= 300 px and mean probability >= 0.80 |

## Validation

5-fold cross-validation grouped by calendar day (no same-day photos on both sides) and stratified by station; the split is in `data/folds_photos.csv`. Out-of-fold PQ on all 707 training photos, pooled over every annotator as in the official scorer: see the technical report. Two human annotators agree with each other at PQ 0.340 on the 296 photos drawn by several annotators.

## Files

| Path | Content |
|---|---|
| `pipeline.ipynb` | End-to-end pipeline (pre-processing, models, prediction, post-processing, check, examples) |
| `weights/` | `resnet34_fold0..4.pt`, `effv2s_fold0..4.pt` |
| `src/train_resnet.py`, `src/train_effnet.py` | Training scripts used on Kaggle (settings via environment variables, see their headers) |
| `data/folds_photos.csv` | The fixed 5-fold split of the training photos |
| `submission/ens10b18_r40.csv` | The exact file submitted to Kaggle |
| `requirements.txt` | Packages and versions used |

Data: MAGFiLO, Kaggle 2026 edition (CC BY 4.0). The code was written with the help of an AI assistant (Claude); all experiments were run and checked by the author.
