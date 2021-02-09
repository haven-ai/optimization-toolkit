

```
python train.py [-h] [--arch [ARCH]] [--dataset [DATASET]]
                [--img_rows [IMG_ROWS]] [--img_cols [IMG_COLS]]
                [--n_epoch [N_EPOCH]] [--batch_size [BATCH_SIZE]]
                [--l_rate [L_RATE]] [--feature_scale [FEATURE_SCALE]]
  --arch           Architecture to use ['fcn8s, unet, segnet etc']
  --dataset        Dataset to use ['pascal, camvid, ade20k etc']
  --img_rows       Height of the input image
  --img_cols       Height of the input image
  --n_epoch        # of the epochs
  --batch_size     Batch Size
  --l_rate         Learning Rate
  --feature_scale  Divider for # of features to use
```
