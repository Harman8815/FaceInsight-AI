import pandas as pd
import tensorflow as tf

from . import config


def load_merged_dataframe(merged_csv=config.MERGED_CSV):
    return pd.read_csv(merged_csv)


def get_column_groups(df, landmarks_csv=config.LANDMARKS_CSV):
    landmarks = pd.read_csv(landmarks_csv)
    landmark_cols = [c for c in landmarks.columns if c != "image_id"]
    bbox_cols = config.BBOX_COLS
    excluded = {"image_id", "confidence", "target", "partition", *bbox_cols, *landmark_cols}
    attr_cols = [c for c in df.columns if c not in excluded]
    return attr_cols, bbox_cols, landmark_cols


def split_partitions(df):
    train_df = df[df.partition == 0].reset_index(drop=True)
    val_df = df[df.partition == 1].reset_index(drop=True)
    test_df = df[df.partition == 2].reset_index(drop=True)
    return train_df, val_df, test_df


def normalize_targets(d, bbox_cols, landmark_cols,
                       img_w=config.ORIG_IMG_WIDTH, img_h=config.ORIG_IMG_HEIGHT):
    """Same normalization as the original notebook (cell 32), parameterized."""
    d = d.copy()
    d[bbox_cols[0]] = d[bbox_cols[0]] / img_w
    d[bbox_cols[2]] = d[bbox_cols[2]] / img_w
    d[bbox_cols[1]] = d[bbox_cols[1]] / img_h
    d[bbox_cols[3]] = d[bbox_cols[3]] / img_h
    for c in landmark_cols:
        d[c] = d[c] / (img_w if c.endswith("_x") else img_h)
    return d


def _load_and_preprocess(path, attr, land, box, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = img / 255.0
    return img, {"attributes": attr, "landmarks": land, "bbox": box}


def make_dataset(d, attr_cols, bbox_cols, landmark_cols, img_dir=config.IMG_DIR,
                  img_size=config.IMG_SIZE, shuffle=False, batch_size=config.BATCH_SIZE):
    """Same tf.data pipeline as the original notebook (cell 33), parameterized."""
    paths = (img_dir + d["image_id"]).values
    attr = (d[attr_cols].values == 1).astype("float32")
    land = d[landmark_cols].values.astype("float32")
    box = d[bbox_cols].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, attr, land, box))
    ds = ds.map(lambda p, a, l, b: _load_and_preprocess(p, a, l, b, img_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)