```python
import os
import json
import platform
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
 
# ─── TensorFlow / Keras ───────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
 
# ─── Scikit-learn (pipeline classique) ───────────────────────────────────────
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats as sp_stats
from sklearn.pipeline import Pipeline
 
# ─── OpenCV (extraction de features) ─────────────────────────────────────────
import cv2
 
# ─── TensorFlow Datasets (EuroSAT) ───────────────────────────────────────────
import tensorflow_datasets as tfds
```


```python

```


```python
EUROSAT_DIR = r"C:\Users\HP\OneDrive\Desktop\DL_FAMIEN 2027\Seg_U-net_New\data"
 
# Détection automatique des classes depuis les sous-dossiers
_detected_classes = sorted([
    d for d in os.listdir(EUROSAT_DIR)
    if os.path.isdir(os.path.join(EUROSAT_DIR, d))
]) if os.path.isdir(EUROSAT_DIR) else []
 
PALETTE = [
    (139,107,79), (168,135,92), (111,143,78), (194,176,138),   # AnnualCrop      
    (31,77,46), (47,107,63), (79,138,90), (111,167,122),   # Forest          
    (63,107,60), (92,138,79), (122,163,95), (155,122,85),   # PermanentCrop   
    (184,181,176), (140,127,122), (208,203,197), (111,106,102),   # Residential     
]
 
CONFIG = {
    "img_size":       64,
    "n_classes":      len(_detected_classes) if _detected_classes else 4,
    "batch_size":     32,
    "epochs_unet":    20,      # [REMARQUE 4] entraînement plus long (arrêt anticipé actif)
    "learning_rate":  1e-4,
    "seed":           42,
    "output_dir":     "outputs",
    "class_names":    _detected_classes if _detected_classes else [
        "AnnualCrop", "Forest", "PermanentCrop", "Residential"
    ],
    # ↓ CORRECTION — clé manquante qui causait KeyError: 'palette'
    "palette":        PALETTE[:len(_detected_classes)] if _detected_classes else PALETTE[:4],
    "mosaic_grid":    4,

    # ── [REMARQUE 1/2] découpage stratifié 80 / 10 / 10 identique pour tous les modèles
    "split":          (0.8, 0.1, 0.1),

    # ── [REMARQUE 3] supervision plus riche : mosaïques synthétiques à frontières réelles
    "use_mosaic_supervision": True,
    "mosaic_ratio":           0.5,     # part des échantillons d'entraînement en mosaïque

    # ── [REMARQUE 5] validation de l'estimation de superficie
    "pixel_size_m":   10.0,            # résolution Sentinel-2 (bandes RVB + NIR)
    "agri_class_names": ["AnnualCrop", "PermanentCrop"],

    # ── [REMARQUE 6] bandes proche infrarouge (EuroSAT multispectral .tif, 13 bandes)
    #    False → images RVB (jpg/png) ; True → EUROSAT_DIR doit pointer vers EuroSAT_MS
    "use_multispectral": False,
    "ms_band_indices":   [3, 2, 1, 7],  # B04 (R), B03 (V), B02 (B), B08 (NIR) dans l'ordre à 13 bandes
    "bootstrap_n":       1000,
}
CONFIG["n_channels"] = 4 if CONFIG["use_multispectral"] else 3
 
os.makedirs(CONFIG["output_dir"], exist_ok=True)
tf.random.set_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
print(f"Classes détectées ({CONFIG['n_classes']}) : {CONFIG['class_names']}") 
```

    Classes détectées (4) : ['AnnualCrop', 'Forest', 'PermanentCrop', 'Residential']
    


```python

```


```python
def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        return None


def log_run_config(extra=None):
    """[REMARQUE 4] Enregistre matériel, versions et hyperparamètres dans outputs/run_config.json."""
    info = {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "processeur": platform.processor() or platform.machine(),
        "coeurs_cpu": os.cpu_count(),
        "ram_go": _ram_gb(),
        "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
        "versions": {"tensorflow": tf.__version__, "numpy": np.__version__,
                     "opencv": cv2.__version__,
                     "scikit-learn": __import__("sklearn").__version__},
        "config": {k: v for k, v in CONFIG.items()
                   if k not in ("palette",) and not isinstance(v, (np.ndarray,))},
        "hyperparametres": {
            "unet": {"etages": 4, "filtres": [64, 128, 256, 512], "bottleneck": 1024,
                     "batchnorm": True, "optimiseur": "Adam", "pertes": "focale (cls) ; focale+Dice (seg)",
                     "poids_pertes": {"cls_out": 1.0, "seg_out": 0.3},
                     "early_stopping_patience": 10, "reduce_lr": {"factor": 0.5, "patience": 4}},
            "svm": {"kernel": "rbf", "C": 10, "gamma": "scale"},
            "random_forest": {"n_estimators": 300, "max_depth": 20, "criterion": "gini"},
        },
    }
    if extra:
        info.update(extra)
    path = os.path.join(CONFIG["output_dir"], "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Configuration enregistrée → {path}")
    print(f"  Matériel : CPU={info['processeur']} ({info['coeurs_cpu']} cœurs) | RAM={info['ram_go']} Go | GPU={info['gpu'] or 'aucun'}")
    return info
```


```python

```


```python
_IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _read_image(path):
    """Lit une image RVB (jpg/png) ou multispectrale (.tif) → uint8 (H, W, C)."""
    IMG = CONFIG["img_size"]
    if CONFIG["use_multispectral"]:
        # EuroSAT_MS : GeoTIFF 13 bandes (réflectances 0-10000)
        try:
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read()                       # (13, H, W)
            arr = np.transpose(arr, (1, 2, 0))
        except ImportError:
            import tifffile
            arr = tifffile.imread(path)
            if arr.ndim == 3 and arr.shape[0] <= 13 and arr.shape[-1] > 13:
                arr = np.transpose(arr, (1, 2, 0))
        arr = arr[..., CONFIG["ms_band_indices"]].astype(np.float32)
        arr = np.clip(arr / 3000.0, 0.0, 1.0) * 255.0      # mise à l'échelle 0-255
        arr = cv2.resize(arr, (IMG, IMG), interpolation=cv2.INTER_AREA)
        return arr.astype(np.uint8)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (IMG, IMG):
        rgb = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return rgb


def _class_counts(y):
    return {CONFIG["class_names"][c]: int(n) for c, n in enumerate(np.bincount(y, minlength=CONFIG["n_classes"]))}


def step1_acquisition():
    """
    [REMARQUES 1 et 2] Découpage STRATIFIÉ, mélangé, produit en UNE seule opération
    (train / val / test disjoints). Toutes les classes sont présentes dans le test,
    et le même jeu de test sert à TOUS les modèles (comparaison appariée).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — Acquisition des données (EuroSAT local, découpage stratifié)")
    print("=" * 60)

    SEED, DIR = CONFIG["seed"], EUROSAT_DIR
    paths, labels = [], []
    for c, cname in enumerate(CONFIG["class_names"]):
        cdir = os.path.join(DIR, cname)
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(_IMG_EXT))
        paths += [os.path.join(cdir, f) for f in files]
        labels += [c] * len(files)
    labels = np.array(labels, dtype=np.int32)
    if len(paths) == 0:
        raise FileNotFoundError(f"Aucune image trouvée dans {DIR}")

    print(f"  • Lecture de {len(paths)} images…")
    X = np.stack([_read_image(p) for p in paths]).astype(np.uint8)   # (N, H, W, C)
    CONFIG["n_channels"] = X.shape[-1]

    f_tr, f_va, f_te = CONFIG["split"]
    idx = np.arange(len(labels))
    tr_idx, rest_idx = train_test_split(idx, test_size=(1 - f_tr), stratify=labels,
                                        random_state=SEED, shuffle=True)
    va_idx, te_idx = train_test_split(rest_idx, test_size=f_te / (f_va + f_te),
                                      stratify=labels[rest_idx], random_state=SEED, shuffle=True)

    # Garde-fous : ensembles disjoints, toutes les classes dans chaque ensemble
    assert not (set(tr_idx) & set(va_idx)) and not (set(tr_idx) & set(te_idx)) \
        and not (set(va_idx) & set(te_idx)), "Recouvrement entre ensembles !"
    for nm, ii in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        present = np.unique(labels[ii])
        if len(present) != CONFIG["n_classes"]:
            raise ValueError(f"L'ensemble {nm} ne contient pas toutes les classes : {present}")

    CONFIG["n_train"], CONFIG["n_val"], CONFIG["n_test"] = len(tr_idx), len(va_idx), len(te_idx)
    CONFIG["class_counts"] = {"train": _class_counts(labels[tr_idx]),
                              "val":   _class_counts(labels[va_idx]),
                              "test":  _class_counts(labels[te_idx])}

    def _make(ii):
        return tf.data.Dataset.from_tensor_slices((X[ii], labels[ii])).map(
            lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

    print(f"  • Dossier  : {DIR}")
    print(f"  • Classes  : {CONFIG['n_classes']} → {CONFIG['class_names']}  | canaux : {CONFIG['n_channels']}")
    print(f"  • Train    : {CONFIG['n_train']}  |  Val : {CONFIG['n_val']}  |  Test : {CONFIG['n_test']}")
    for nm in ("train", "val", "test"):
        print(f"    - {nm:<5} : {CONFIG['class_counts'][nm]}")
    return _make(tr_idx), _make(va_idx), _make(te_idx)
```


```python

```


```python
def preprocess_image(image, label):
    """Normalisation [0,1]. Le resize est déjà fait par image_dataset_from_directory."""
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

```


```python

```


```python
def augment_image(image, label):
    """Augmentations aléatoires (flip, rotation, luminosité, contraste)."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.rot90(
        image,
        k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label
 
 
def build_tf_datasets(ds_train, ds_val, ds_test):
    """
    Construit les tf.data.Dataset prétraités et augmentés.
 
    [CORRECTION 4/5] : Les datasets reçus sont NON-batchés (batch_size=None dans
    image_dataset_from_directory). On applique prétraitement + augmentation,
    puis on batche ici. Les datasets renvoyés sont BATCHÉS pour l'entraînement,
    mais on conserve aussi les versions non-batchées (ds_*_raw) pour SVM/RF
    et pour make_unet_ds().
    """
    print("\n" + "="*60)
    print("ÉTAPES 2-3 — Prétraitement & Augmentation")
    print("="*60)
 
    AUTOTUNE = tf.data.AUTOTUNE
    B = CONFIG["batch_size"]
 
    # Versions batchées pour l'entraînement U-Net
    train_ds = (ds_train
                .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                .map(augment_image,    num_parallel_calls=AUTOTUNE)
                .shuffle(6000)
                .batch(B)
                .prefetch(AUTOTUNE))
 
    val_ds = (ds_val
              .map(preprocess_image, num_parallel_calls=AUTOTUNE)
              .batch(B)
              .prefetch(AUTOTUNE))
 
    test_ds = (ds_test
               .map(preprocess_image, num_parallel_calls=AUTOTUNE)
               .batch(B)
               .prefetch(AUTOTUNE))

    # [REMARQUE 2] Train SANS augmentation, ordre fixe : utilisé par SVM / Random Forest
    train_plain_ds = (ds_train
                      .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                      .batch(B)
                      .prefetch(AUTOTUNE))
 
    print("  • Augmentations : flip H/V, rotation 90°, luminosité, contraste")
    return train_ds, val_ds, test_ds, train_plain_ds
 
 
```


```python

```


```python
def extract_sobel_canny(image_np):
    """
    Vecteur de descripteurs : 70 dims en RVB (Sobel 32+4, Canny 8+2, histogrammes couleur 24).
    [REMARQUE 6] Avec un canal NIR : +10 dims (histogramme NDVI 8 classes + moyenne + écart-type).
    """
    rgb       = image_np[..., :3]
    gray      = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    canny     = cv2.Canny(gray, 50, 150)
    sobel_hist, _ = np.histogram(sobel_mag.ravel(), bins=32, range=(0, 500))
    canny_hist, _ = np.histogram(canny.ravel(),     bins=8,  range=(0, 255))
    sobel_stats   = np.array([sobel_mag.mean(), sobel_mag.std(),
                               sobel_mag.max(), np.percentile(sobel_mag, 75)])
    canny_stats   = np.array([canny.mean(), canny.sum() / canny.size])
    color_hist    = np.concatenate([
        np.histogram(rgb[:, :, c].ravel(), bins=8, range=(0, 1))[0]
        for c in range(3)
    ])
    feats = [sobel_hist, canny_hist, sobel_stats, canny_stats, color_hist]
    if image_np.shape[-1] >= 4:
        red, nir = image_np[..., 0], image_np[..., 3]
        ndvi = (nir - red) / (nir + red + 1e-6)
        feats += [np.histogram(ndvi.ravel(), bins=8, range=(-1, 1))[0],
                  np.array([ndvi.mean(), ndvi.std()])]
    return np.concatenate(feats)


def collect_numpy_data(ds, max_samples=None, balance=False):
    """
    [REMARQUE 2] Par défaut : AUCUN sous-échantillonnage ni rééquilibrage, pour que les
    modèles classiques soient évalués sur exactement les mêmes images que le U-Net.
    """
    images, labels = [], []
    for imgs_b, lbls_b in ds:
        if imgs_b.ndim == 3:
            imgs_b = tf.expand_dims(imgs_b, 0)
            lbls_b = tf.expand_dims(lbls_b, 0)
        for img, lbl in zip(imgs_b.numpy(), lbls_b.numpy()):
            if max_samples is not None and len(images) >= max_samples:
                break
            img_f = img.astype(np.float32)
            if img_f.max() > 1.0:
                img_f /= 255.0
            images.append(img_f)
            labels.append(int(lbl))
        if max_samples is not None and len(images) >= max_samples:
            break

    images = np.array(images)
    labels = np.array(labels)
    if not balance:
        return images, labels

    n_cls  = CONFIG["n_classes"]
    counts = np.bincount(labels, minlength=n_cls)
    n_each = counts[counts > 0].min()
    idx = []
    for c in range(n_cls):
        cls_idx = np.where(labels == c)[0]
        if len(cls_idx) > 0:
            idx.extend(np.random.choice(cls_idx, size=n_each, replace=False).tolist())
    idx = np.array(idx)
    np.random.shuffle(idx)
    return images[idx], labels[idx]


def step4a_classical_pipeline(ds_train_plain, ds_test):
    print("\n" + "=" * 60)
    print("ÉTAPE 4A — Pipeline Classique (Sobel/Canny → SVM / Random Forest)")
    print("=" * 60)
    print("  Collecte des données (train complet, sans augmentation)…")
    X_tr, y_tr = collect_numpy_data(ds_train_plain)
    print("  Collecte des données (test complet — même jeu que le U-Net)…")
    X_te, y_te = collect_numpy_data(ds_test)
    print(f"  Train : {len(y_tr)} images | Test : {len(y_te)} images")
    print("  Extraction des descripteurs…")
    X_feat_tr = np.array([extract_sobel_canny(img) for img in X_tr])
    X_feat_te = np.array([extract_sobel_canny(img) for img in X_te])
    print(f"  Dimension feature vector : {X_feat_tr.shape[1]}")
    CONFIG["feature_dim"] = int(X_feat_tr.shape[1])

    results = {}
    for name, clf in [
        ("SVM",           SVC(kernel="rbf", C=10, gamma="scale",
                              random_state=CONFIG["seed"])),
        ("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=20,
                                                  n_jobs=-1, random_state=CONFIG["seed"])),
    ]:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_feat_tr, y_tr)
        y_pred = pipe.predict(X_feat_te)
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "f1_macro": f1,
                         "y_pred": y_pred, "y_true": y_te}
        print(f"\n  [{name}] Accuracy={acc:.4f}  F1-macro={f1:.4f}")
        print(classification_report(y_te, y_pred,
                             labels=list(range(CONFIG["n_classes"])),
                             target_names=CONFIG["class_names"], zero_division=0))
    return results
```


```python

```


```python
def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    return x

def encoder_block(x, filters, name):
    skip   = conv_block(x, filters, name)
    pooled = layers.MaxPooling2D(2, name=f"{name}_pool")(skip)
    return skip, pooled

def decoder_block(x, skip, filters, name):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_cat")([x, skip])
    x = conv_block(x, filters, name)
    return x

def build_unet_classifier(img_size=64, n_classes=4, n_channels=3):
    inputs = layers.Input(shape=(img_size, img_size, n_channels), name="input_img")

    # Encodeur
    s1, p1 = encoder_block(inputs, 64,  "enc1")
    s2, p2 = encoder_block(p1,    128,  "enc2")
    s3, p3 = encoder_block(p2,    256,  "enc3")
    s4, p4 = encoder_block(p3,    512,  "enc4")

    # Bottleneck
    bottleneck = conv_block(p4, 1024, "bottleneck")

    # ✅ Décodeur — ordre corrigé : (x, skip, filters, name)
    d1 = decoder_block(bottleneck, s4, 512, "dec1")
    d2 = decoder_block(d1,         s3, 256, "dec2")
    d3 = decoder_block(d2,         s2, 128, "dec3")
    d4 = decoder_block(d3,         s1,  64, "dec4")

    # Sortie segmentation
    seg_output = layers.Conv2D(n_classes, 1, activation="softmax", name="seg_out")(d4)

    # Tête de classification
    gap = layers.GlobalAveragePooling2D(name="gap")(bottleneck)
    x   = layers.Dense(256, activation="relu", name="fc1")(gap)
    x   = layers.Dropout(0.4, name="drop1")(x)
    x   = layers.Dense(128, activation="relu", name="fc2")(x)
    x   = layers.Dropout(0.3, name="drop2")(x)
    cls_output = layers.Dense(n_classes, activation="softmax", name="cls_out")(x)

    model = Model(inputs, [cls_output, seg_output], name="UNet_Classifier")
    return model
```


```python
# PERTES : Focal Loss + Dice Loss (décodeur)

def focal_loss(gamma=2.0, alpha=0.75):
    def _focal(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        fl     = alpha * tf.pow(1 - pt, gamma) * ce
        return tf.reduce_mean(fl)
    return _focal
 
 
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter    = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * inter + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
 
 
def focal_dice_loss(y_true, y_pred):
    """Combinaison Focal + Dice pour la tête de segmentation."""
    fl = focal_loss()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return 0.5 * fl + 0.5 * dl
```


```python
# PRÉPARATION DES DATASETS POUR U-Net (dual output)

def make_segmentation_ds(tf_ds):
    """
    Masque pseudo-sémantique UNIFORME (supervision faible) : tous les pixels d'une
    image reçoivent la classe de son étiquette globale.
    seg_out : (H, W, n_classes) one-hot par pixel  |  cls_out : (n_classes,)
    """
    N   = CONFIG["n_classes"]
    IMG = CONFIG["img_size"]

    def _transform(image, label):
        lbl_oh   = tf.one_hot(label, N)
        seg_mask = tf.ones([IMG, IMG, 1], dtype=tf.float32) * lbl_oh[tf.newaxis, tf.newaxis, :]
        return image, {"seg_out": seg_mask, "cls_out": lbl_oh}

    return tf_ds.map(_transform, num_parallel_calls=tf.data.AUTOTUNE)


def compose_two_images(img_a, lbl_a, img_b, lbl_b, kind, t):
    """
    [REMARQUE 3] Compose deux patchs de classes (éventuellement) différentes le long d'une
    frontière (verticale, horizontale ou diagonale). Le masque obtenu possède de VRAIES
    frontières entre classes, connues par construction → supervision plus riche qu'un
    masque uniforme, et vérité terrain exacte pour l'IoU / l'estimation de superficie.
    t ∈ [0.25, 0.75] × IMG : position de la frontière.
    """
    N   = CONFIG["n_classes"]
    IMG = CONFIG["img_size"]
    yy = tf.cast(tf.range(IMG), tf.float32)[:, None] * tf.ones([1, IMG])
    xx = tf.cast(tf.range(IMG), tf.float32)[None, :] * tf.ones([IMG, 1])
    m0, m1, m2 = xx < t, yy < t, (xx + yy) < 2.0 * t
    mask = tf.where(tf.equal(kind, 0), m0, tf.where(tf.equal(kind, 1), m1, m2))
    m = tf.cast(mask, tf.float32)[..., None]                           # (H, W, 1)
    img = m * img_a + (1.0 - m) * img_b
    seg = m * tf.one_hot(lbl_a, N)[None, None, :] + (1.0 - m) * tf.one_hot(lbl_b, N)[None, None, :]
    cls = tf.reduce_mean(seg, axis=[0, 1])                             # proportions de surface
    return img, {"seg_out": seg, "cls_out": cls}


def make_mosaic_ds(tf_ds):
    """tf_ds : (image, label) non batché → mosaïques à deux régions, masques à frontières réelles."""
    IMG = CONFIG["img_size"]

    def _compose(imgs, lbls):
        kind = tf.random.uniform([], 0, 3, dtype=tf.int32)
        t    = tf.random.uniform([], 0.25, 0.75) * IMG
        return compose_two_images(imgs[0], lbls[0], imgs[1], lbls[1], kind, t)

    return (tf_ds.batch(2, drop_remainder=True)
                 .map(_compose, num_parallel_calls=tf.data.AUTOTUNE))
```


```python
# ENTRAÎNEMENT U-Net

def step4b_unet(train_ds, val_ds):
    print("\n" + "=" * 60)
    print("ÉTAPE 4B — U-Net Segmentation (Focal + Dice)")
    print("=" * 60)

    B     = CONFIG["batch_size"]
    N_VAL = CONFIG["n_val"]
    N_TR  = CONFIG["n_train"]

    model = build_unet_classifier(CONFIG["img_size"], CONFIG["n_classes"], CONFIG["n_channels"])
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["learning_rate"]),
        loss={"cls_out": focal_loss(), "seg_out": focal_dice_loss},
        loss_weights={"cls_out": 1.0, "seg_out": 0.3},
        metrics={"cls_out": ["accuracy"], "seg_out": ["accuracy"]},
    )

    # ── Entraînement : mélange masques uniformes + mosaïques à frontières réelles
    base_tr  = train_ds.unbatch()
    uniform  = make_segmentation_ds(base_tr).repeat()
    if CONFIG["use_mosaic_supervision"]:
        r      = CONFIG["mosaic_ratio"]
        mosaic = make_mosaic_ds(base_tr.shuffle(2000, seed=CONFIG["seed"])).repeat()
        stream = tf.data.Dataset.sample_from_datasets([uniform, mosaic],
                                                      weights=[1.0 - r, r], seed=CONFIG["seed"])
        print(f"  • Supervision : {100*(1-r):.0f}% masques uniformes + {100*r:.0f}% mosaïques (frontières réelles)")
    else:
        stream = uniform
        print("  • Supervision : masques uniformes uniquement")
    train_unet = stream.shuffle(3000).batch(B).prefetch(tf.data.AUTOTUNE)

    # ── Validation : patchs uniformes + mosaïques (mêmes proportions de classes)
    base_va = val_ds.unbatch()
    val_set = make_segmentation_ds(base_va)
    n_val_samples = N_VAL
    if CONFIG["use_mosaic_supervision"]:
        val_set = val_set.concatenate(make_mosaic_ds(base_va))
        n_val_samples = N_VAL + N_VAL // 2
    val_unet = val_set.batch(B).repeat().prefetch(tf.data.AUTOTUNE)

    steps_per_epoch  = max(1, N_TR // B)
    validation_steps = max(1, int(np.ceil(n_val_samples / B)))

    ckpt_path = os.path.join(CONFIG["output_dir"], "unet_best.weights.h5")
    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True,
                        monitor="val_cls_out_accuracy", mode="max",
                        save_weights_only=True),
        EarlyStopping(patience=10, monitor="val_cls_out_accuracy",
                      mode="max", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_unet,
        validation_data=val_unet,
        epochs=CONFIG["epochs_unet"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    CONFIG["epochs_run"] = len(history.history["loss"])
    print(f"  • Époques effectuées : {CONFIG['epochs_run']} / {CONFIG['epochs_unet']}")
    return model, history
```


```python
def step5_evaluation(model, test_ds, classical_results):
    print("\n" + "="*60)
    print("ÉTAPE 5 — Évaluation sur l'ensemble de test")
    print("="*60)
 
    y_true_all, y_pred_all = [], []
 
    for images_batch, labels_batch in test_ds:
        # CORRECTION — appel eager → dict nommé, robuste Keras 2 et 3
        preds_dict = model(images_batch, training=False)
 
        if isinstance(preds_dict, dict):
            # Keras 3 : dict {"cls_out": tensor, "seg_out": tensor}
            cls_preds = preds_dict["cls_out"].numpy()
        elif isinstance(preds_dict, (list, tuple)):
            # Keras 2 : [cls_out_tensor, seg_out_tensor]
            # build_unet_classifier retourne Model(inputs, [cls_output, seg_output])
            # donc index 0 = cls_out, index 1 = seg_out
            cls_preds = np.array(preds_dict[0])
        else:
            cls_preds = np.array(preds_dict)
 
        y_pred_all.extend(np.argmax(cls_preds, axis=-1).tolist())
        y_true_all.extend(labels_batch.numpy().astype(int).tolist())
 
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
 
    # [REMARQUES 1/2] Garde-fou : le test doit contenir toutes les classes
    if len(np.unique(y_true)) < CONFIG["n_classes"]:
        print("  ⚠ ATTENTION : l'ensemble de test ne contient pas toutes les classes — "
              "les métriques ne sont PAS interprétables.")
    print(f"\n  Classes présentes dans y_true : {np.unique(y_true)}")
    print(f"  Classes présentes dans y_pred : {np.unique(y_pred)}")
 
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
 
    print(f"\n  [U-Net] Accuracy={acc:.4f}  F1-macro={f1:.4f}")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(CONFIG["n_classes"])),
        target_names=CONFIG["class_names"],
        zero_division=0
    ))
    return {"accuracy": acc, "f1_macro": f1, "y_true": y_true, "y_pred": y_pred}
 
```


```python
# Vérifier le contenu du dataset
#for images, labels in test_ds.take(1):
 #   print("Shape images :", images.shape)
  #  print("Shape labels :", labels.shape)
   # print("Valeurs labels :", np.unique(labels.numpy()))
    #print("N classes CONFIG :", len(CONFIG["class_names"]))
```


```python

```


```python
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

def _get_palette():
    """Robuste si CONFIG['palette'] absent (cellule CONFIG non ré-exécutée)."""
    return CONFIG.get("palette", PALETTE[:CONFIG.get("n_classes", 4)])
 
def get_colormap():
    colors_norm = [(r/255, g/255, b/255) for (r, g, b) in _get_palette()]
    return ListedColormap(colors_norm, name="eurosat_seg")
 
def label_to_color_mask(label_map):
    label_map = np.array(label_map)
    if label_map.ndim == 3 and label_map.shape[0] == 1:
        label_map = label_map[0]
    if label_map.ndim == 3:
        label_map = np.argmax(label_map, axis=-1)
    if label_map.ndim != 2:
        raise ValueError(f"label_to_color_mask : shape={label_map.shape}, attendu (H,W)")
    H, W = label_map.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(_get_palette()):
        color_mask[label_map == cls_idx] = color
    return color_mask
 
```


```python
def predict_patch_segmentation(model, image_np):
    """
    NIVEAU 1 — Segmentation d'une image individuelle (patch 64×64).
 
    Entrée  : image_np (H×W×3, float32 [0,1])
    Sortie  : seg_map   (H, W)            — label prédit par pixel
              seg_proba (H, W, n_classes) — probabilités softmax par pixel
 
    CORRECTION ValueError "not enough values to unpack" :
      model.predict() avec deux sorties retourne selon la version Keras :
        - une liste Python  [array(1,H,W,N), array(1,N)]  → preds[0][0]
        - un dict           {"seg_out":..., "cls_out":...} → preds["seg_out"][0]
        - un seul array fusionné (rare)
      On utilise model() directement (appel eager) qui retourne toujours
      un dict nommé par les noms de sorties → extraction robuste par nom.
    """
    img_tensor = tf.cast(
        tf.expand_dims(image_np, 0), tf.float32   # (1, H, W, 3)
    )
 
    # Appel eager : toujours un dict {nom_sortie: tensor}
    preds_dict = model(img_tensor, training=False)
 
    # Extraction de seg_out — robuste quel que soit le type retourné
    if isinstance(preds_dict, dict):
        seg_proba = preds_dict["seg_out"][0].numpy()       # (H, W, n_classes)
    elif isinstance(preds_dict, (list, tuple)):
        # preds[0] = seg_out (1,H,W,N), preds[1] = cls_out (1,N)
        seg_tensor = preds_dict[1]
        seg_proba  = (seg_tensor[0].numpy()
                      if hasattr(seg_tensor, "numpy")
                      else np.array(seg_tensor)[0])        # (H, W, n_classes)
    else:
        # Fallback : array numpy brut
        seg_proba = np.array(preds_dict)[0]
 
    # Sécurité : garantir shape (H, W, n_classes)
    if seg_proba.ndim == 0 or seg_proba.ndim == 1:
        raise ValueError(
            f"predict_patch_segmentation : seg_proba.shape={seg_proba.shape} "
            f"inattendu. Vérifiez que la sortie 'seg_out' est bien (H,W,n_classes)."
        )
    if seg_proba.ndim == 4:          # batch non retiré
        seg_proba = seg_proba[0]
 
    seg_map = np.argmax(seg_proba, axis=-1).astype(np.int32)  # (H, W)
    return seg_map, seg_proba
 
 
def build_mosaic(ds_raw, grid_size=4, n_patches=None):
    """
    Construit une grande image mosaïque en assemblant des patches EuroSAT.
    Retourne aussi les labels ground-truth de chaque patch.
 
    grid_size : nombre de patches par côté (grid_size×grid_size patches)
    Résultat  : mosaïque de taille (grid_size*64) × (grid_size*64) pixels
    """
    IMG = CONFIG["img_size"]
    G   = grid_size
    total = G * G
 
    images_list, labels_list = [], []
    count = 0
 
    for img_t, lbl_t in ds_raw.take(total * 2):   # take un peu plus pour avoir le compte
        img_np = img_t.numpy().astype(np.float32)
        if img_np.max() > 1.0:
            img_np /= 255.0
        images_list.append(img_np)
        labels_list.append(int(lbl_t.numpy()))
        count += 1
        if count >= total:
            break
 
    # Construction de la grille
    rows = []
    for r in range(G):
        row_imgs = [images_list[r * G + c] for c in range(G)]
        rows.append(np.concatenate(row_imgs, axis=1))          # (64, 64*G, 3)
    mosaic = np.concatenate(rows, axis=0)                       # (64*G, 64*G, 3)
 
    return mosaic, images_list, labels_list
 
 
def predict_mosaic_segmentation(model, mosaic, images_list):
    """
    NIVEAU 2 — Segmentation d'une grande image mosaïque.
 
    Stratégie : on prédit chaque patch 64×64 séparément puis on assemble.
    Retourne une carte de segmentation (H×W) à l'échelle de la mosaïque.
    """
    IMG  = CONFIG["img_size"]
    G    = CONFIG["mosaic_grid"]
    H, W = mosaic.shape[:2]
 
    seg_mosaic = np.zeros((H, W), dtype=np.int32)
    proba_mosaic = np.zeros((H, W, CONFIG["n_classes"]), dtype=np.float32)
 
    for idx, img_patch in enumerate(images_list):
        r = idx // G
        c = idx %  G
        y0, y1 = r * IMG, (r + 1) * IMG
        x0, x1 = c * IMG, (c + 1) * IMG
 
        seg_map, seg_proba = predict_patch_segmentation(model, img_patch)
        seg_mosaic[y0:y1, x0:x1]    = seg_map
        proba_mosaic[y0:y1, x0:x1]  = seg_proba
 
    return seg_mosaic, proba_mosaic
 
 
def visualize_patch_segmentation(model, test_ds, n_exemples=6):
    """
    NIVEAU 1 — Visualisation de la segmentation sur des images individuelles.
    Affiche : image originale | masque segmentation | overlay
    """
    print("\n" + "="*60)
    print("SEGMENTATION NIVEAU 1 — Patches individuels")
    print("="*60)
 
    cmap = get_colormap()
    fig, axes = plt.subplots(n_exemples, 3,
                             figsize=(12, n_exemples * 3.5))
    fig.suptitle("Segmentation pixel par pixel — Patches EuroSAT",
                 fontsize=14, fontweight="bold")
 
    collected = []
    for imgs_b, lbls_b in test_ds:
        for img, lbl in zip(imgs_b.numpy(), lbls_b.numpy()):
            collected.append((img, int(lbl)))
            if len(collected) >= n_exemples:
                break
        if len(collected) >= n_exemples:
            break
 
    for i, (img_np, true_label) in enumerate(collected):
        seg_map, seg_proba = predict_patch_segmentation(model, img_np)
        pred_label         = int(np.bincount(seg_map.ravel()).argmax())
        color_mask         = label_to_color_mask(seg_map)
 
        # Overlay : image + masque semi-transparent
        overlay = (img_np[..., :3] * 255).astype(np.uint8).copy()
        overlay = cv2.addWeighted(overlay, 0.55,
                                   color_mask, 0.45, 0)
 
        ax_img, ax_seg, ax_ov = axes[i]
 
        ax_img.imshow(img_np[..., :3])
        ax_img.set_title(f"Image originale\n(Vraie classe : {CONFIG['class_names'][true_label]})",
                         fontsize=8)
        ax_img.axis("off")
 
        ax_seg.imshow(seg_map, cmap=cmap,
                      vmin=0, vmax=CONFIG["n_classes"] - 1)
        ax_seg.set_title(f"Masque de segmentation\n(Classe dominante : {CONFIG['class_names'][pred_label]})",
                         fontsize=8)
        ax_seg.axis("off")
 
        ax_ov.imshow(overlay)
        ax_ov.set_title("Overlay (image + masque)", fontsize=8)
        ax_ov.axis("off")
 
    # Légende des classes
    legend_patches = [
        mpatches.Patch(
            facecolor=tuple(c/255 for c in CONFIG["palette"][i]),
            label=CONFIG["class_names"][i]
        )
        for i in range(CONFIG["n_classes"])
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=CONFIG["n_classes"], fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
 
    plt.tight_layout()
    out_path = os.path.join(CONFIG["output_dir"], "segmentation_patches.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Sauvegardé → {out_path}")
    plt.show()
 
 
def visualize_mosaic_segmentation(model, ds_raw):
    """
    NIVEAU 2 — Carte de segmentation d'une grande image mosaïque.
    Simule la segmentation à l'échelle du paysage / d'une parcelle agricole.
    """
    print("\n" + "="*60)
    print("SEGMENTATION NIVEAU 2 — Mosaïque (paysage complet)")
    print("="*60)
 
    G   = CONFIG["mosaic_grid"]
    IMG = CONFIG["img_size"]
 
    print(f"  Construction de la mosaïque {G}×{G} patches ({G*IMG}×{G*IMG} px)…")
    mosaic, images_list, labels_gt = build_mosaic(ds_raw, grid_size=G)
 
    print("  Prédiction de segmentation sur chaque patch…")
    seg_mosaic, proba_mosaic = predict_mosaic_segmentation(model, mosaic, images_list)
    color_map_mosaic         = label_to_color_mask(seg_mosaic)
 
    # Overlay mosaïque
    mosaic_uint8 = (mosaic[..., :3] * 255).astype(np.uint8)
    overlay_mosaic = cv2.addWeighted(mosaic_uint8, 0.5,
                                      color_map_mosaic, 0.5, 0)
 
    # Grille des labels ground-truth
    gt_grid = np.array(labels_gt).reshape(G, G)
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"Segmentation de parcelles — Mosaïque EuroSAT {G}×{G}",
                 fontsize=14, fontweight="bold")
 
    # Image originale
    axes[0].imshow(mosaic[..., :3])
    axes[0].set_title("Image satellite originale", fontweight="bold")
    axes[0].axis("off")
    # Grille des classes GT
    for r in range(G):
        for c in range(G):
            axes[0].text(
                c * IMG + IMG//2, r * IMG + IMG//2,
                CONFIG["class_names"][gt_grid[r, c]][:5],
                ha="center", va="center", fontsize=6,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5)
            )
 
    # Carte de segmentation
    cmap = get_colormap()
    im   = axes[1].imshow(seg_mosaic, cmap=cmap,
                           vmin=0, vmax=CONFIG["n_classes"] - 1)
    axes[1].set_title("Carte de segmentation prédite", fontweight="bold")
    axes[1].axis("off")
 
    # Lignes de grille pour délimiter les parcelles
    for k in range(1, G):
        axes[1].axhline(k * IMG - 0.5, color="white", linewidth=0.8, alpha=0.6)
        axes[1].axvline(k * IMG - 0.5, color="white", linewidth=0.8, alpha=0.6)
 
    # Overlay
    axes[2].imshow(overlay_mosaic)
    axes[2].set_title("Overlay (image + segmentation)", fontweight="bold")
    axes[2].axis("off")
    for k in range(1, G):
        axes[2].axhline(k * IMG - 0.5, color="white", linewidth=0.8, alpha=0.6)
        axes[2].axvline(k * IMG - 0.5, color="white", linewidth=0.8, alpha=0.6)
 
    # Légende
    legend_patches = [
        mpatches.Patch(
            facecolor=tuple(c/255 for c in CONFIG["palette"][i]),
            label=CONFIG["class_names"][i]
        )
        for i in range(CONFIG["n_classes"])
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=CONFIG["n_classes"], fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
 
    plt.tight_layout()
    out_path = os.path.join(CONFIG["output_dir"], "mosaique_segmentation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Sauvegardé → {out_path}")
    plt.show()
 
    # Statistiques des surfaces par classe
    print("\n  Répartition des surfaces prédites :")
    total_pixels = seg_mosaic.size
    for cls_idx, cls_name in enumerate(CONFIG["class_names"]):
        n_px  = np.sum(seg_mosaic == cls_idx)
        pct   = 100.0 * n_px / total_pixels
        bar   = "█" * int(pct / 2)
        print(f"    {cls_name:<20} {pct:5.1f}%  {bar}")
```


```python

```


```python
def plot_confusion_matrix(ax, y_true, y_pred, title, class_names):
    n  = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Prédit", fontsize=8)
    ax.set_ylabel("Réel",   fontsize=8)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
def step7_comparison(unet_result, classical_results, history=None):
    print("\n" + "="*60)
    print("ÉTAPE 7 — Comparaison des algorithmes")
    print("="*60)
    all_results = {**classical_results, "U-Net Seg": unet_result}
    for name, res in all_results.items():
        print(f"  {name:<20} Acc={res['accuracy']:.4f}   F1={res['f1_macro']:.4f}")
 
    fig = plt.figure(figsize=(20, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)
 
    # Barplot
    ax0   = fig.add_subplot(gs[0, 0])
    names = list(all_results.keys())
    accs  = [r["accuracy"] for r in all_results.values()]
    f1s   = [r["f1_macro"] for r in all_results.values()]
    clrs  = ["#4C72B0", "#DD8452", "#55A868"]
    x     = np.arange(len(names))
    ax0.bar(x - 0.2, accs, 0.35, label="Accuracy", color=clrs[:len(names)])
    ax0.bar(x + 0.2, f1s,  0.35, label="F1-macro",
            color=[c + "99" for c in clrs[:len(names)]])
    ax0.set_xticks(x)
    ax0.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax0.set_ylim(0, 1.1)
    ax0.set_title("Accuracy & F1-macro", fontweight="bold")
    ax0.legend(fontsize=8); ax0.grid(axis="y", alpha=0.3)
    for xi, (a, f) in enumerate(zip(accs, f1s)):
        ax0.text(xi - 0.2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
        ax0.text(xi + 0.2, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=7)
 
    # Courbes apprentissage
    if history:
        ax1 = fig.add_subplot(gs[0, 1])
        ep  = range(1, len(history.history.get("seg_out_accuracy", [])) + 1)
        ax1.plot(ep, history.history.get("seg_out_accuracy", []),
                 label="Train (seg)", color="#4C72B0")
        ax1.plot(ep, history.history.get("val_seg_out_accuracy", []),
                 label="Val (seg)",   color="#DD8452", linestyle="--")
        ax1.set_title("Accuracy — Segmentation", fontweight="bold")
        ax1.set_xlabel("Époque"); ax1.set_ylabel("Accuracy")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
 
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(ep, history.history.get("loss", []),
                 label="Train loss", color="#4C72B0")
        ax2.plot(ep, history.history.get("val_loss", []),
                 label="Val loss",   color="#DD8452", linestyle="--")
        ax2.set_title("Loss totale", fontweight="bold")
        ax2.set_xlabel("Époque"); ax2.set_ylabel("Loss")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
 
    # Matrices de confusion
    for i, (name, res) in enumerate(all_results.items()):
        ax = fig.add_subplot(gs[1, i])
        plot_confusion_matrix(ax, res["y_true"], res["y_pred"],
                              f"Conf. Matrix — {name}", CONFIG["class_names"])
 
    plt.suptitle("Comparaison complète des pipelines EuroSAT",
                 fontsize=14, fontweight="bold")
    out_path = os.path.join(CONFIG["output_dir"], "comparaison_finale.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure sauvegardée → {out_path}")
    plt.show()
 
```


```python

```


```python
def paired_statistical_comparison(unet_result, classical_results):
    """
    [REMARQUE 2] Comparaison appariée sur les MÊMES images de test :
      • vérification que y_true est identique pour tous les modèles ;
      • test exact de McNemar (désaccords entre deux modèles) ;
      • IC bootstrap à 95 % de l'accuracy et du F1-macro (rééchantillonnage commun).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5B — Comparaison appariée (même jeu de test)")
    print("=" * 60)
    all_res = {**classical_results, "U-Net": unet_result}
    names = list(all_res.keys())
    y_true = all_res[names[0]]["y_true"]
    for n in names[1:]:
        if not np.array_equal(all_res[n]["y_true"], y_true):
            raise ValueError(f"Jeux de test différents entre {names[0]} et {n} : comparaison appariée impossible.")
    print(f"  ✔ Jeu de test identique pour les {len(names)} modèles ({len(y_true)} images)")

    rng = np.random.RandomState(CONFIG["seed"])
    boots = [rng.randint(0, len(y_true), len(y_true)) for _ in range(CONFIG["bootstrap_n"])]
    summary = {}
    for n in names:
        yp = all_res[n]["y_pred"]
        accs = [accuracy_score(y_true[b], yp[b]) for b in boots]
        f1s  = [f1_score(y_true[b], yp[b], average="macro", zero_division=0) for b in boots]
        summary[n] = {"acc": all_res[n]["accuracy"], "acc_ci": (np.percentile(accs, 2.5), np.percentile(accs, 97.5)),
                      "f1": all_res[n]["f1_macro"],  "f1_ci":  (np.percentile(f1s, 2.5),  np.percentile(f1s, 97.5))}
        print(f"  {n:<14} Acc={summary[n]['acc']:.4f} [{summary[n]['acc_ci'][0]:.4f}, {summary[n]['acc_ci'][1]:.4f}]"
              f"   F1-macro={summary[n]['f1']:.4f} [{summary[n]['f1_ci'][0]:.4f}, {summary[n]['f1_ci'][1]:.4f}]")

    print("\n  Test exact de McNemar (H0 : même taux d'erreur) :")
    mcn = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ci = all_res[names[i]]["y_pred"] == y_true
            cj = all_res[names[j]]["y_pred"] == y_true
            b, c = int(np.sum(ci & ~cj)), int(np.sum(~ci & cj))
            p = 1.0 if (b + c) == 0 else sp_stats.binomtest(min(b, c), b + c, 0.5).pvalue
            mcn[f"{names[i]} vs {names[j]}"] = {"b": b, "c": c, "p": float(p)}
            print(f"    {names[i]:<14} vs {names[j]:<14} b={b:4d} c={c:4d}  p={p:.4g}")
    return {"summary": summary, "mcnemar": mcn}
```


```python

```


```python
def _collect_test_arrays(test_ds):
    X, y = [], []
    for xb, yb in test_ds:
        X.append(xb.numpy()); y.append(yb.numpy())
    return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int32)


def _seg_predict(model, X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        p = model(X[i:i + bs], training=False)
        seg = p["seg_out"] if isinstance(p, dict) else p[1]
        out.append(np.argmax(seg.numpy(), axis=-1))
    return np.concatenate(out)                                   # (N, H, W)


def _numpy_compose(img_a, la, img_b, lb, kind, t):
    IMG = CONFIG["img_size"]
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    mask = [xx < t, yy < t, (xx + yy) < 2 * t][kind]
    img = np.where(mask[..., None], img_a, img_b)
    gt = np.where(mask, la, lb).astype(np.int32)
    return img, gt


def _pixel_confusion(gt, pred, n):
    return np.bincount((gt.ravel() * n + pred.ravel()), minlength=n * n).reshape(n, n)


def _seg_scores(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou  = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
    present = cm.sum(1) > 0
    return iou, dice, float(np.nanmean(iou[present])), float(tp.sum() / cm.sum())


def step5c_segmentation_and_area(model, test_ds):
    """
    [REMARQUE 5]
      • IoU par classe, mIoU, Dice, exactitude pixel — sur (a) patchs de test à masque
        uniforme et (b) mosaïques de test à frontières réelles (vérité terrain exacte) ;
      • validation de l'estimation de superficie agricole (ha) : MAE, RMSE, biais,
        erreur relative, R², corrélation — par rapport à la surface de référence.
      ⚠ La référence est ici la composition connue des mosaïques ; une validation sur
        données réelles exige une couche cadastrale / parcellaire externe (RPG, etc.).
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 5C — Métriques de segmentation & estimation de superficie")
    print("=" * 60)
    n, names = CONFIG["n_classes"], CONFIG["class_names"]
    X, y = _collect_test_arrays(test_ds)
    IMG = CONFIG["img_size"]
    results = {}

    # (a) patchs uniformes
    pred_u = _seg_predict(model, X)
    gt_u   = np.broadcast_to(y[:, None, None], pred_u.shape)
    cm_u   = _pixel_confusion(gt_u, pred_u, n)
    iou_u, dice_u, miou_u, pa_u = _seg_scores(cm_u)

    # (b) mosaïques de test (graine fixe, chaque image utilisée comme A et comme B)
    rng  = np.random.RandomState(CONFIG["seed"])
    perm = rng.permutation(len(X))
    imgs_m, gts_m = [], []
    for a, b in zip(range(len(X)), perm):
        img, gt = _numpy_compose(X[a], y[a], X[b], y[b], int(rng.randint(0, 3)), rng.uniform(0.25, 0.75) * IMG)
        imgs_m.append(img); gts_m.append(gt)
    imgs_m, gts_m = np.stack(imgs_m).astype(np.float32), np.stack(gts_m)
    pred_m = _seg_predict(model, imgs_m)
    cm_m   = _pixel_confusion(gts_m, pred_m, n)
    iou_m, dice_m, miou_m, pa_m = _seg_scores(cm_m)

    print("\n  IoU par classe :")
    print(f"    {'Classe':<16}{'Patchs uniformes':>18}{'Mosaïques (frontières)':>26}")
    for c in range(n):
        print(f"    {names[c]:<16}{iou_u[c]:>18.4f}{iou_m[c]:>26.4f}")
    print(f"    {'mIoU':<16}{miou_u:>18.4f}{miou_m:>26.4f}")
    print(f"    {'Pixel accuracy':<16}{pa_u:>18.4f}{pa_m:>26.4f}")
    results["segmentation"] = {
        "uniform":  {"iou": iou_u.tolist(), "dice": dice_u.tolist(), "miou": miou_u, "pixel_acc": pa_u},
        "mosaic":   {"iou": iou_m.tolist(), "dice": dice_m.tolist(), "miou": miou_m, "pixel_acc": pa_m},
    }

    # (c) estimation de superficie agricole (ha) par mosaïque
    px_ha = (CONFIG["pixel_size_m"] ** 2) / 10000.0
    agri  = [names.index(c) for c in CONFIG["agri_class_names"] if c in names]
    if not agri:
        print("  ⚠ Aucune classe agricole reconnue dans CONFIG['agri_class_names'] — estimation de superficie ignorée.")
        return results
    ref  = np.isin(gts_m, agri).sum(axis=(1, 2)) * px_ha
    est  = np.isin(pred_m, agri).sum(axis=(1, 2)) * px_ha
    err  = est - ref
    mae, rmse, bias = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), float(np.mean(err))
    rel  = float(100.0 * err.sum() / max(ref.sum(), 1e-9))
    r2   = float(r2_score(ref, est)); rho = float(np.corrcoef(ref, est)[0, 1])
    print(f"\n  Superficie agricole par mosaïque 64×64 px ({64*64*px_ha:.2f} ha au total) — {len(ref)} mosaïques")
    print(f"    MAE={mae:.3f} ha | RMSE={rmse:.3f} ha | biais={bias:+.3f} ha | erreur relative globale={rel:+.2f} %")
    print(f"    R²={r2:.3f} | corrélation de Pearson={rho:.3f}")
    results["area"] = {"mae_ha": mae, "rmse_ha": rmse, "bias_ha": bias, "rel_err_pct": rel, "r2": r2, "pearson": rho}

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ref, est, s=6, alpha=0.4)
    lim = [0, max(ref.max(), est.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Superficie agricole de référence (ha)"); ax.set_ylabel("Superficie estimée (ha)")
    ax.set_title(f"Validation de l'estimation de superficie\nMAE={mae:.3f} ha, R²={r2:.3f}", fontsize=10)
    ax.grid(alpha=0.3)
    out = os.path.join(CONFIG["output_dir"], "validation_superficie.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  Figure sauvegardée → {out}")
    np.savetxt(os.path.join(CONFIG["output_dir"], "validation_superficie.csv"),
               np.c_[ref, est], delimiter=",", header="reference_ha,estimee_ha", comments="")
    return results


def estimate_areas_ha(model, image_np):
    """Superficie (ha) de chaque classe pour une image 64×64×C (float [0,1]) — utilisable sur toute donnée de référence externe."""
    seg = _seg_predict(model, image_np[None].astype(np.float32))[0]
    px_ha = (CONFIG["pixel_size_m"] ** 2) / 10000.0
    return {CONFIG["class_names"][c]: float(np.sum(seg == c) * px_ha) for c in range(CONFIG["n_classes"])}
```


```python

```


```python
# MAIN — Exécution du flowchart

def main():
    print("\n" + "█" * 60)
    print("  PIPELINE EUROSAT — SEGMENTATION DES PARCELLES")
    print("█" * 60)

    # 1. Acquisition (découpage stratifié, toutes classes dans train/val/test)
    ds_train_raw, ds_val_raw, ds_test_raw = step1_acquisition()
    hw = log_run_config()

    # 2-3. Prétraitement + Augmentation
    train_ds, val_ds, test_ds, train_plain_ds = build_tf_datasets(ds_train_raw, ds_val_raw, ds_test_raw)

    # 4A. Pipeline classique (même jeu de test complet que le U-Net)
    classical_results = step4a_classical_pipeline(train_plain_ds, test_ds)

    # 4B. U-Net (masques uniformes + mosaïques à frontières réelles)
    unet_model, history = step4b_unet(train_ds, val_ds)

    # 5. Évaluation classification + comparaison appariée
    unet_result = step5_evaluation(unet_model, test_ds, classical_results)
    paired = paired_statistical_comparison(unet_result, classical_results)

    # 5C. IoU / mIoU et validation de la superficie
    seg_area = step5c_segmentation_and_area(unet_model, test_ds)

    # 6A / 6B. Visualisations
    visualize_patch_segmentation(unet_model, test_ds, n_exemples=6)
    visualize_mosaic_segmentation(unet_model, ds_test_raw)

    # 7. Comparaison finale
    step7_comparison(unet_result, classical_results, history)

    # Sauvegarde du bilan chiffré
    summary = {"epochs_run": CONFIG.get("epochs_run"), "feature_dim": CONFIG.get("feature_dim"),
               "class_counts": CONFIG.get("class_counts"),
               "classification": {"U-Net": {"accuracy": unet_result["accuracy"], "f1_macro": unet_result["f1_macro"]},
                                  **{k: {"accuracy": v["accuracy"], "f1_macro": v["f1_macro"]}
                                     for k, v in classical_results.items()}},
               "paired": paired, "segmentation_et_superficie": seg_area}
    with open(os.path.join(CONFIG["output_dir"], "resultats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "█" * 60)
    print("  PIPELINE TERMINÉ — résultats dans outputs/ (resultats.json, run_config.json)")
    print("█" * 60)

    return unet_model, classical_results, unet_result


if __name__ == "__main__":
    unet_model, classical_results, unet_result = main()
```

    
    ████████████████████████████████████████████████████████████
      PIPELINE EUROSAT — SEGMENTATION DES PARCELLES
    ████████████████████████████████████████████████████████████
    
    ============================================================
    ÉTAPE 1 — Acquisition des données (EuroSAT local, découpage stratifié)
    ============================================================
      • Lecture de 6000 images…
      • Dossier  : C:\Users\HP\OneDrive\Desktop\DL_FAMIEN 2027\Seg_U-net_New\data
      • Classes  : 4 → ['AnnualCrop', 'Forest', 'PermanentCrop', 'Residential']  | canaux : 3
      • Train    : 4800  |  Val : 600  |  Test : 600
        - train : {'AnnualCrop': 1200, 'Forest': 1200, 'PermanentCrop': 1200, 'Residential': 1200}
        - val   : {'AnnualCrop': 150, 'Forest': 150, 'PermanentCrop': 150, 'Residential': 150}
        - test  : {'AnnualCrop': 150, 'Forest': 150, 'PermanentCrop': 150, 'Residential': 150}
    WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
      Configuration enregistrée → outputs\run_config.json
      Matériel : CPU=Intel64 Family 6 Model 165 Stepping 2, GenuineIntel (12 cœurs) | RAM=23.8 Go | GPU=aucun
    
    ============================================================
    ÉTAPES 2-3 — Prétraitement & Augmentation
    ============================================================
      • Augmentations : flip H/V, rotation 90°, luminosité, contraste
    
    ============================================================
    ÉTAPE 4A — Pipeline Classique (Sobel/Canny → SVM / Random Forest)
    ============================================================
      Collecte des données (train complet, sans augmentation)…
      Collecte des données (test complet — même jeu que le U-Net)…
      Train : 4800 images | Test : 600 images
      Extraction des descripteurs…
      Dimension feature vector : 70
    
      [SVM] Accuracy=0.9650  F1-macro=0.9648
                   precision    recall  f1-score   support
    
       AnnualCrop       0.97      0.91      0.94       150
           Forest       0.99      1.00      0.99       150
    PermanentCrop       0.92      0.96      0.94       150
      Residential       0.98      0.99      0.99       150
    
         accuracy                           0.96       600
        macro avg       0.97      0.96      0.96       600
     weighted avg       0.97      0.96      0.96       600
    
    
      [Random Forest] Accuracy=0.9517  F1-macro=0.9517
                   precision    recall  f1-score   support
    
       AnnualCrop       0.94      0.89      0.91       150
           Forest       0.99      1.00      0.99       150
    PermanentCrop       0.89      0.93      0.91       150
      Residential       0.99      0.98      0.99       150
    
         accuracy                           0.95       600
        macro avg       0.95      0.95      0.95       600
     weighted avg       0.95      0.95      0.95       600
    
    
    ============================================================
    ÉTAPE 4B — U-Net Segmentation (Focal + Dice)
    ============================================================
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "UNet_Classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                  </span>┃<span style="font-weight: bold"> Output Shape              </span>┃<span style="font-weight: bold">         Param # </span>┃<span style="font-weight: bold"> Connected to               </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ input_img (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)         │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc1_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">1,792</span> │ input_img[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc1_bn1 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ enc1_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc1_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ enc1_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc1_bn2 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ enc1_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc1_pool (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ enc1_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc2_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │ enc1_pool[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc2_bn1 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ enc2_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc2_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ enc2_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc2_bn2 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ enc2_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc2_pool (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ enc2_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc3_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │ enc2_pool[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc3_bn1 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │           <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ enc3_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc3_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ enc3_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc3_bn2 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │           <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ enc3_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc3_pool (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)         │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ enc3_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc4_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │       <span style="color: #00af00; text-decoration-color: #00af00">1,180,160</span> │ enc3_pool[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc4_bn1 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ enc4_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc4_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │       <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ enc4_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc4_bn2 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ enc4_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ enc4_pool (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ enc4_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bottleneck_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)        │       <span style="color: #00af00; text-decoration-color: #00af00">4,719,616</span> │ enc4_pool[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bottleneck_bn1                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">4,096</span> │ bottleneck_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]        │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bottleneck_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)        │       <span style="color: #00af00; text-decoration-color: #00af00">9,438,208</span> │ bottleneck_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ bottleneck_bn2                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)        │           <span style="color: #00af00; text-decoration-color: #00af00">4,096</span> │ bottleneck_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]        │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec1_up (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │       <span style="color: #00af00; text-decoration-color: #00af00">2,097,664</span> │ bottleneck_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec1_cat (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)        │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dec1_up[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],             │
│                               │                           │                 │ enc4_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec1_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │       <span style="color: #00af00; text-decoration-color: #00af00">4,719,104</span> │ dec1_cat[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec1_bn1 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ dec1_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec1_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │       <span style="color: #00af00; text-decoration-color: #00af00">2,359,808</span> │ dec1_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec1_bn2 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │ dec1_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec2_up (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">524,544</span> │ dec1_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec2_cat (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)       │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dec2_up[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],             │
│                               │                           │                 │ enc3_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec2_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │       <span style="color: #00af00; text-decoration-color: #00af00">1,179,904</span> │ dec2_cat[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec2_bn1 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │           <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ dec2_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec2_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">590,080</span> │ dec2_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec2_bn2 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │           <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ dec2_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec3_up (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">131,200</span> │ dec2_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec3_cat (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dec3_up[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],             │
│                               │                           │                 │ enc2_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec3_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">295,040</span> │ dec3_cat[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec3_bn1 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ dec3_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec3_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │         <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │ dec3_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec3_bn2 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ dec3_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec4_up (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2DTranspose</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">32,832</span> │ dec3_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ gap (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)              │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ bottleneck_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]       │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec4_cat (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dec4_up[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],             │
│                               │                           │                 │ enc1_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ fc1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)               │         <span style="color: #00af00; text-decoration-color: #00af00">262,400</span> │ gap[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec4_c1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">73,792</span> │ dec4_cat[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ drop1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ fc1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec4_bn1 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ dec4_c1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ fc2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)               │          <span style="color: #00af00; text-decoration-color: #00af00">32,896</span> │ drop1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec4_c2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │ dec4_bn1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ drop2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)               │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ fc2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dec4_bn2 (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │             <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ dec4_c2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ cls_out (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)                 │             <span style="color: #00af00; text-decoration-color: #00af00">516</span> │ drop2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ seg_out (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">260</span> │ dec4_bn2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]             │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">31,351,304</span> (119.60 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">31,339,528</span> (119.55 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">11,776</span> (46.00 KB)
</pre>



      • Supervision : 50% masques uniformes + 50% mosaïques (frontières réelles)
    Epoch 1/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1150s[0m 8s/step - cls_out_accuracy: 0.7392 - cls_out_loss: 0.0723 - loss: 0.1307 - seg_out_accuracy: 0.7536 - seg_out_loss: 0.1946 - val_cls_out_accuracy: 0.2433 - val_cls_out_loss: 0.4037 - val_loss: 0.5522 - val_seg_out_accuracy: 0.2913 - val_seg_out_loss: 0.4941 - learning_rate: 1.0000e-04
    Epoch 2/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1168s[0m 8s/step - cls_out_accuracy: 0.8244 - cls_out_loss: 0.0450 - loss: 0.0825 - seg_out_accuracy: 0.8509 - seg_out_loss: 0.1249 - val_cls_out_accuracy: 0.2533 - val_cls_out_loss: 0.4520 - val_loss: 0.7006 - val_seg_out_accuracy: 0.2501 - val_seg_out_loss: 0.8254 - learning_rate: 1.0000e-04
    Epoch 3/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1165s[0m 8s/step - cls_out_accuracy: 0.8540 - cls_out_loss: 0.0367 - loss: 0.0682 - seg_out_accuracy: 0.8762 - seg_out_loss: 0.1048 - val_cls_out_accuracy: 0.2567 - val_cls_out_loss: 0.5978 - val_loss: 0.8668 - val_seg_out_accuracy: 0.2528 - val_seg_out_loss: 0.8801 - learning_rate: 1.0000e-04
    Epoch 4/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1170s[0m 8s/step - cls_out_accuracy: 0.8565 - cls_out_loss: 0.0328 - loss: 0.0625 - seg_out_accuracy: 0.8826 - seg_out_loss: 0.0991 - val_cls_out_accuracy: 0.3644 - val_cls_out_loss: 0.2244 - val_loss: 0.4011 - val_seg_out_accuracy: 0.3505 - val_seg_out_loss: 0.5816 - learning_rate: 1.0000e-04
    Epoch 5/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1168s[0m 8s/step - cls_out_accuracy: 0.8579 - cls_out_loss: 0.0321 - loss: 0.0590 - seg_out_accuracy: 0.8940 - seg_out_loss: 0.0896 - val_cls_out_accuracy: 0.9189 - val_cls_out_loss: 0.0202 - val_loss: 0.0460 - val_seg_out_accuracy: 0.9002 - val_seg_out_loss: 0.0845 - learning_rate: 1.0000e-04
    Epoch 6/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1181s[0m 8s/step - cls_out_accuracy: 0.8533 - cls_out_loss: 0.0311 - loss: 0.0574 - seg_out_accuracy: 0.8953 - seg_out_loss: 0.0875 - val_cls_out_accuracy: 0.9289 - val_cls_out_loss: 0.0154 - val_loss: 0.0326 - val_seg_out_accuracy: 0.9324 - val_seg_out_loss: 0.0574 - learning_rate: 1.0000e-04
    Epoch 7/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1178s[0m 8s/step - cls_out_accuracy: 0.8623 - cls_out_loss: 0.0284 - loss: 0.0531 - seg_out_accuracy: 0.9011 - seg_out_loss: 0.0825 - val_cls_out_accuracy: 0.9044 - val_cls_out_loss: 0.0177 - val_loss: 0.0414 - val_seg_out_accuracy: 0.9039 - val_seg_out_loss: 0.0785 - learning_rate: 1.0000e-04
    Epoch 8/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1189s[0m 8s/step - cls_out_accuracy: 0.8731 - cls_out_loss: 0.0272 - loss: 0.0509 - seg_out_accuracy: 0.9063 - seg_out_loss: 0.0788 - val_cls_out_accuracy: 0.8989 - val_cls_out_loss: 0.0215 - val_loss: 0.0448 - val_seg_out_accuracy: 0.9062 - val_seg_out_loss: 0.0808 - learning_rate: 1.0000e-04
    Epoch 9/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1079s[0m 7s/step - cls_out_accuracy: 0.8723 - cls_out_loss: 0.0255 - loss: 0.0473 - seg_out_accuracy: 0.9129 - seg_out_loss: 0.0729 - val_cls_out_accuracy: 0.9344 - val_cls_out_loss: 0.0125 - val_loss: 0.0274 - val_seg_out_accuracy: 0.9427 - val_seg_out_loss: 0.0509 - learning_rate: 1.0000e-04
    Epoch 10/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m497s[0m 3s/step - cls_out_accuracy: 0.8846 - cls_out_loss: 0.0255 - loss: 0.0465 - seg_out_accuracy: 0.9164 - seg_out_loss: 0.0701 - val_cls_out_accuracy: 0.9211 - val_cls_out_loss: 0.0157 - val_loss: 0.0309 - val_seg_out_accuracy: 0.9372 - val_seg_out_loss: 0.0533 - learning_rate: 1.0000e-04
    Epoch 11/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m499s[0m 3s/step - cls_out_accuracy: 0.8869 - cls_out_loss: 0.0238 - loss: 0.0431 - seg_out_accuracy: 0.9236 - seg_out_loss: 0.0643 - val_cls_out_accuracy: 0.9478 - val_cls_out_loss: 0.0136 - val_loss: 0.0286 - val_seg_out_accuracy: 0.9427 - val_seg_out_loss: 0.0496 - learning_rate: 1.0000e-04
    Epoch 12/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m504s[0m 3s/step - cls_out_accuracy: 0.8800 - cls_out_loss: 0.0238 - loss: 0.0434 - seg_out_accuracy: 0.9208 - seg_out_loss: 0.0653 - val_cls_out_accuracy: 0.9278 - val_cls_out_loss: 0.0197 - val_loss: 0.0369 - val_seg_out_accuracy: 0.9395 - val_seg_out_loss: 0.0565 - learning_rate: 1.0000e-04
    Epoch 13/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3s/step - cls_out_accuracy: 0.8874 - cls_out_loss: 0.0244 - loss: 0.0436 - seg_out_accuracy: 0.9241 - seg_out_loss: 0.0640
    Epoch 13: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m498s[0m 3s/step - cls_out_accuracy: 0.8844 - cls_out_loss: 0.0244 - loss: 0.0441 - seg_out_accuracy: 0.9226 - seg_out_loss: 0.0656 - val_cls_out_accuracy: 0.9322 - val_cls_out_loss: 0.0140 - val_loss: 0.0281 - val_seg_out_accuracy: 0.9436 - val_seg_out_loss: 0.0458 - learning_rate: 1.0000e-04
    Epoch 14/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m633s[0m 4s/step - cls_out_accuracy: 0.8935 - cls_out_loss: 0.0201 - loss: 0.0359 - seg_out_accuracy: 0.9375 - seg_out_loss: 0.0526 - val_cls_out_accuracy: 0.9367 - val_cls_out_loss: 0.0111 - val_loss: 0.0236 - val_seg_out_accuracy: 0.9505 - val_seg_out_loss: 0.0411 - learning_rate: 5.0000e-05
    Epoch 15/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25827s[0m 173s/step - cls_out_accuracy: 0.8967 - cls_out_loss: 0.0200 - loss: 0.0350 - seg_out_accuracy: 0.9407 - seg_out_loss: 0.0500 - val_cls_out_accuracy: 0.9522 - val_cls_out_loss: 0.0103 - val_loss: 0.0208 - val_seg_out_accuracy: 0.9587 - val_seg_out_loss: 0.0348 - learning_rate: 5.0000e-05
    Epoch 16/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1216s[0m 8s/step - cls_out_accuracy: 0.9081 - cls_out_loss: 0.0182 - loss: 0.0318 - seg_out_accuracy: 0.9463 - seg_out_loss: 0.0453 - val_cls_out_accuracy: 0.9467 - val_cls_out_loss: 0.0111 - val_loss: 0.0216 - val_seg_out_accuracy: 0.9618 - val_seg_out_loss: 0.0346 - learning_rate: 5.0000e-05
    Epoch 17/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1211s[0m 8s/step - cls_out_accuracy: 0.9027 - cls_out_loss: 0.0193 - loss: 0.0333 - seg_out_accuracy: 0.9442 - seg_out_loss: 0.0469 - val_cls_out_accuracy: 0.9100 - val_cls_out_loss: 0.0163 - val_loss: 0.0341 - val_seg_out_accuracy: 0.9329 - val_seg_out_loss: 0.0586 - learning_rate: 5.0000e-05
    Epoch 18/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1070s[0m 7s/step - cls_out_accuracy: 0.9035 - cls_out_loss: 0.0187 - loss: 0.0323 - seg_out_accuracy: 0.9466 - seg_out_loss: 0.0453 - val_cls_out_accuracy: 0.9511 - val_cls_out_loss: 0.0106 - val_loss: 0.0215 - val_seg_out_accuracy: 0.9560 - val_seg_out_loss: 0.0356 - learning_rate: 5.0000e-05
    Epoch 19/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3s/step - cls_out_accuracy: 0.9082 - cls_out_loss: 0.0175 - loss: 0.0307 - seg_out_accuracy: 0.9480 - seg_out_loss: 0.0439
    Epoch 19: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m508s[0m 3s/step - cls_out_accuracy: 0.9090 - cls_out_loss: 0.0178 - loss: 0.0308 - seg_out_accuracy: 0.9487 - seg_out_loss: 0.0433 - val_cls_out_accuracy: 0.9233 - val_cls_out_loss: 0.0138 - val_loss: 0.0271 - val_seg_out_accuracy: 0.9476 - val_seg_out_loss: 0.0443 - learning_rate: 5.0000e-05
    Epoch 20/20
    [1m150/150[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m507s[0m 3s/step - cls_out_accuracy: 0.9054 - cls_out_loss: 0.0182 - loss: 0.0308 - seg_out_accuracy: 0.9500 - seg_out_loss: 0.0421 - val_cls_out_accuracy: 0.9433 - val_cls_out_loss: 0.0112 - val_loss: 0.0207 - val_seg_out_accuracy: 0.9617 - val_seg_out_loss: 0.0316 - learning_rate: 2.5000e-05
    Restoring model weights from the end of the best epoch: 15.
      • Époques effectuées : 20 / 20
    
    ============================================================
    ÉTAPE 5 — Évaluation sur l'ensemble de test
    ============================================================
    
      Classes présentes dans y_true : [0 1 2 3]
      Classes présentes dans y_pred : [0 1 2 3]
    
      [U-Net] Accuracy=0.9800  F1-macro=0.9799
                   precision    recall  f1-score   support
    
       AnnualCrop       0.99      0.93      0.96       150
           Forest       0.99      0.99      0.99       150
    PermanentCrop       0.94      0.99      0.97       150
      Residential       1.00      1.00      1.00       150
    
         accuracy                           0.98       600
        macro avg       0.98      0.98      0.98       600
     weighted avg       0.98      0.98      0.98       600
    
    
    ============================================================
    ÉTAPE 5B — Comparaison appariée (même jeu de test)
    ============================================================
      ✔ Jeu de test identique pour les 3 modèles (600 images)
      SVM            Acc=0.9650 [0.9500, 0.9783]   F1-macro=0.9648 [0.9484, 0.9783]
      Random Forest  Acc=0.9517 [0.9350, 0.9667]   F1-macro=0.9517 [0.9344, 0.9674]
      U-Net          Acc=0.9800 [0.9683, 0.9900]   F1-macro=0.9799 [0.9677, 0.9901]
    
      Test exact de McNemar (H0 : même taux d'erreur) :
        SVM            vs Random Forest  b=  12 c=   4  p=0.07681
        SVM            vs U-Net          b=   6 c=  15  p=0.07835
        Random Forest  vs U-Net          b=   6 c=  23  p=0.002316
    
    ============================================================
    ÉTAPE 5C — Métriques de segmentation & estimation de superficie
    ============================================================
    
      IoU par classe :
        Classe            Patchs uniformes    Mosaïques (frontières)
        AnnualCrop                  0.9135                    0.8426
        Forest                      0.9802                    0.9439
        PermanentCrop               0.9267                    0.8418
        Residential                 0.9942                    0.9432
        mIoU                        0.9537                    0.8929
        Pixel accuracy              0.9760                    0.9426
    
      Superficie agricole par mosaïque 64×64 px (40.96 ha au total) — 600 mosaïques
        MAE=1.059 ha | RMSE=3.212 ha | biais=-0.235 ha | erreur relative globale=-1.14 %
        R²=0.953 | corrélation de Pearson=0.977
    


    
![png](output_30_7.png)
    


      Figure sauvegardée → outputs\validation_superficie.png
    
    ============================================================
    SEGMENTATION NIVEAU 1 — Patches individuels
    ============================================================
      Sauvegardé → outputs\segmentation_patches.png
    


    
![png](output_30_9.png)
    


    
    ============================================================
    SEGMENTATION NIVEAU 2 — Mosaïque (paysage complet)
    ============================================================
      Construction de la mosaïque 4×4 patches (256×256 px)…
      Prédiction de segmentation sur chaque patch…
      Sauvegardé → outputs\mosaique_segmentation.png
    


    
![png](output_30_11.png)
    


    
      Répartition des surfaces prédites :
        AnnualCrop             6.2%  ███
        Forest                25.0%  ████████████
        PermanentCrop         31.3%  ███████████████
        Residential           37.5%  ██████████████████
    
    ============================================================
    ÉTAPE 7 — Comparaison des algorithmes
    ============================================================
      SVM                  Acc=0.9650   F1=0.9648
      Random Forest        Acc=0.9517   F1=0.9517
      U-Net Seg            Acc=0.9800   F1=0.9799
    
      Figure sauvegardée → outputs\comparaison_finale.png
    


    
![png](output_30_13.png)
    


    
    ████████████████████████████████████████████████████████████
      PIPELINE TERMINÉ — résultats dans outputs/ (resultats.json, run_config.json)
    ████████████████████████████████████████████████████████████
    


```python

```
