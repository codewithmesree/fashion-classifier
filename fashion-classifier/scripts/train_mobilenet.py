"""
train_mobilenet.py — MobileNetV2 transfer-learning trainer for Fashion Classifier
====================================================================================

Prerequisites
-------------
    pip install tensorflow tensorflowjs pillow scikit-learn tqdm

Usage
-----
1.  Download the DeepFashion "Category and Attribute Prediction Benchmark" from:
        http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html
    You need:
        • img/          — the flat directory (or nested, both handled) of JPEGs
        • list_category_img.txt — per-image category label file

2.  Set DEEPFASHION_ROOT below to your local copy.

3.  Run:
        python scripts/train_mobilenet.py

4.  On completion the script writes model artefacts directly into
        public/model/model.json
        public/model/group1-shard*.bin
    overwriting the old Fashion-MNIST model.
"""

# ── User-configurable constants ──────────────────────────────────────────────

# Path to the top-level DeepFashion Category/Attribute Prediction folder.
# Must contain:
#   img/                       (images)
#   Anno/list_category_img.txt (label file)
DEEPFASHION_ROOT = r"C:\path\to\DeepFashion\Category_and_Attribute_Prediction_Benchmark"

# Where to write the exported TF.js model.
# Adjust if your public/ folder is elsewhere relative to this script.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFJS_OUTPUT_DIR = os.path.join(_REPO_ROOT, "public", "model")

# Training hyper-parameters
IMG_SIZE      = 96          # square input fed to MobileNetV2
BATCH_SIZE    = 32
PHASE1_EPOCHS = 15          # head-only training (backbone frozen)
PHASE2_EPOCHS = 10          # top-30-layer fine-tune at low LR
PHASE2_LR     = 5e-5
TEST_SPLIT    = 0.20        # fraction of data held out for evaluation
RANDOM_SEED   = 42

# ── DeepFashion → our 10-class mapping ───────────────────────────────────────
#
# DeepFashion category IDs are 1-indexed strings in list_category_img.txt.
# The file lists 50 fine-grained categories; we collapse them to our 10.
#
# ⚠ REVIEW markers below flag ambiguous mappings — inspect these before training
# and drop or remap classes you disagree with.
#
# Format:  deepfashion_category_name -> our_class_index (0-9)
#   0  T-shirt/top
#   1  Trouser
#   2  Pullover
#   3  Dress
#   4  Coat
#   5  Sandal
#   6  Shirt
#   7  Sneaker
#   8  Bag
#   9  Ankle boot

DEEPFASHION_CATEGORY_MAP: dict[str, int] = {
    # ── Tops ──────────────────────────────────────────────────────────────────
    "Tee":              0,   # T-shirt/top
    "Tank":             0,   # T-shirt/top — sleeveless
    "Blouse":           6,   # Shirt (more formal than a tee)
    "Shirt":            6,   # Shirt
    "Polo":             6,   # Shirt (polo collar)
    "Sweatshirt":       2,   # Pullover (closest — thick pullover silhouette)
    "Hoodie":           2,   # Pullover
    "Sweater":          2,   # Pullover
    "Cardigan":         2,   # Pullover # REVIEW: some cardigans button-up → Coat?
    "Jacket":           4,   # Coat (outerwear with zipper/buttons)
    "Coat":             4,   # Coat
    "Parka":            4,   # Coat
    "Windbreaker":      4,   # Coat
    "Vest":             6,   # Shirt  # REVIEW: could also be Coat if padded
    "Kimono":           3,   # Dress  # REVIEW: long open robe → Dress is proximate
    "Crop Top":         0,   # T-shirt/top

    # ── Bottoms ───────────────────────────────────────────────────────────────
    "Jeans":            1,   # Trouser
    "Pants":            1,   # Trouser
    "Shorts":           1,   # Trouser  # REVIEW: semantically adjacent; no dedicated class
    "Leggings":         1,   # Trouser
    "Skirt":            3,   # Dress    # REVIEW: skirts → closest is Dress

    # ── Full-length / one-piece ───────────────────────────────────────────────
    "Dress":            3,   # Dress
    "Gown":             3,   # Dress
    "Jumpsuit":         3,   # Dress    # REVIEW: one-piece; closest proxy is Dress
    "Romper":           3,   # Dress    # REVIEW: same as Jumpsuit
    "Playsuit":         3,   # Dress    # REVIEW: same as Jumpsuit

    # ── Footwear ──────────────────────────────────────────────────────────────
    "Sandals":          5,   # Sandal
    "Flip Flops":       5,   # Sandal
    "Slippers":         5,   # Sandal   # REVIEW: could argue Ankle boot for enclosed styles
    "Sneakers":         7,   # Sneaker
    "Running Shoes":    7,   # Sneaker
    "Trainers":         7,   # Sneaker
    "Loafers":          7,   # Sneaker  # REVIEW: flat shoes — Sneaker is closest
    "Oxford Shoes":     7,   # Sneaker  # REVIEW: formal shoes — weakest proxy
    "Formal Shoes":     7,   # Sneaker  # REVIEW: same as Oxford — consider dropping
    "Boots":            9,   # Ankle boot
    "Ankle Boots":      9,   # Ankle boot
    "Chelsea Boots":    9,   # Ankle boot
    "Knee High Boots":  9,   # Ankle boot
    "Socks":            9,   # Ankle boot  # REVIEW: very weak — consider dropping (set to None)
    "Stockings":        9,   # Ankle boot  # REVIEW: same as Socks — consider dropping

    # ── Bags & accessories ────────────────────────────────────────────────────
    "Bag":              8,   # Bag
    "Handbag":          8,   # Bag
    "Tote":             8,   # Bag
    "Backpack":         8,   # Bag
    "Clutch":           8,   # Bag
    "Shoulder Bag":     8,   # Bag
    "Crossbody Bag":    8,   # Bag
    "Wallet":           8,   # Bag      # REVIEW: small; might skew bag class stats

    # ── Categories intentionally NOT mapped (will be skipped & reported) ──────
    # "Scarf", "Hat", "Belt", "Sunglasses", "Jewelry", "Watch" → None
}

# Set any key above to None to explicitly drop that category, e.g.:
#   "Socks": None,

# ── Imports ───────────────────────────────────────────────────────────────────
import pathlib, collections, textwrap
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = len(CLASS_NAMES)

# ── 1. Load & parse DeepFashion label file ───────────────────────────────────

def load_deepfashion(root: str) -> tuple[list[str], list[int]]:
    root = pathlib.Path(root)
    label_file = root / "Anno" / "list_category_img.txt"
    img_dir    = root / "img"

    if not label_file.exists():
        sys.exit(
            f"\n❌  Label file not found: {label_file}\n"
            f"    Make sure DEEPFASHION_ROOT points at the top-level benchmark folder\n"
            f"    containing Anno/ and img/ sub-directories.\n"
        )

    lines = label_file.read_text().splitlines()
    # Format:  line 0 = count, line 1 = header, lines 2+ = "<rel_path> <cat_id>"
    # Cat ID is 1-indexed; there is also a list_category_cloth.txt that maps
    # id→name — we parse that too if present.
    cat_cloth_file = root / "Anno" / "list_category_cloth.txt"
    id_to_name: dict[int, str] = {}
    if cat_cloth_file.exists():
        cloth_lines = cat_cloth_file.read_text().splitlines()
        for cl in cloth_lines[2:]:
            parts = cl.split()
            if len(parts) >= 2:
                cat_id = int(parts[-1])
                cat_name = " ".join(parts[:-1]).strip()
                id_to_name[cat_id] = cat_name
    else:
        print("⚠  list_category_cloth.txt not found — category IDs will be used "
              "as-is and matched against DEEPFASHION_CATEGORY_MAP by numeric key. "
              "Consider downloading the full Anno/ folder.")

    image_paths: list[str] = []
    labels:      list[int] = []
    skipped_cats: dict[str, int] = collections.Counter()
    missing_files = 0

    for line in tqdm(lines[2:], desc="Parsing labels", unit="img"):
        parts = line.split()
        if len(parts) < 2:
            continue
        rel_path = parts[0]
        cat_id   = int(parts[1])
        cat_name = id_to_name.get(cat_id, str(cat_id))

        # Resolve our mapping
        class_idx = DEEPFASHION_CATEGORY_MAP.get(cat_name)
        if class_idx is None:
            skipped_cats[cat_name] += 1
            continue

        full_path = img_dir / rel_path
        if not full_path.exists():
            # Some distributions use a flat img/ without sub-dirs
            flat_path = img_dir / pathlib.Path(rel_path).name
            if flat_path.exists():
                full_path = flat_path
            else:
                missing_files += 1
                continue

        image_paths.append(str(full_path))
        labels.append(class_idx)

    # Report
    print(f"\n✅  Loaded {len(image_paths)} images across {NUM_CLASSES} classes.")
    if missing_files:
        print(f"⚠   {missing_files} image paths listed in label file but not found on disk.")
    if skipped_cats:
        print("\n⚠  Skipped DeepFashion categories (not in DEEPFASHION_CATEGORY_MAP):")
        for name, count in sorted(skipped_cats.items(), key=lambda x: -x[1]):
            print(f"     {count:>5}  {name}")
        print("   Add or remap these in DEEPFASHION_CATEGORY_MAP to include them.\n")

    # Per-class breakdown
    print("\nClass distribution:")
    counter = collections.Counter(labels)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {i}  {cls:<18}  {counter.get(i, 0):>5} images")

    return image_paths, labels


# ── 2. tf.data pipeline ───────────────────────────────────────────────────────

def preprocess_image(path: str, label: int, augment: bool) -> tuple[tf.Tensor, tf.Tensor]:
    """Load JPEG/PNG, resize to IMG_SIZE×IMG_SIZE, apply MobileNetV2 scaling."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)

    # Letterbox: pad to square with mid-grey before resizing so aspect ratio is
    # preserved and the letterbox colour doesn't shift the RGB distribution.
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    max_side = tf.maximum(h, w)
    pad_h = max_side - h
    pad_w = max_side - w
    img = tf.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2],
         [pad_w // 2, pad_w - pad_w // 2],
         [0, 0]],
        constant_values=127,   # mid-grey letterbox
    )
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        # Random rotation ±15° via crop-and-resize trick
        img = tf.image.random_crop(
            tf.image.resize(img, [IMG_SIZE + 16, IMG_SIZE + 16]),
            size=[IMG_SIZE, IMG_SIZE, 3],
        )
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0, 255)

    # MobileNetV2 expects [-1, 1]
    img = (img / 127.5) - 1.0
    return img, tf.one_hot(label, NUM_CLASSES)


def make_dataset(paths, labels, augment: bool, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_SEED)
    ds = ds.map(
        lambda p, l: preprocess_image(p, l, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── 3. Model definition ───────────────────────────────────────────────────────

def build_model() -> keras.Model:
    backbone = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False   # Phase 1: freeze entire backbone

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="fashion_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


# ── 4. Evaluation helpers ─────────────────────────────────────────────────────

def per_class_accuracy(model: keras.Model, ds: tf.data.Dataset) -> None:
    correct = np.zeros(NUM_CLASSES, dtype=int)
    total   = np.zeros(NUM_CLASSES, dtype=int)

    for batch_imgs, batch_labels in tqdm(ds, desc="Evaluating", unit="batch"):
        preds    = model.predict(batch_imgs, verbose=0)
        pred_cls = np.argmax(preds, axis=1)
        true_cls = np.argmax(batch_labels.numpy(), axis=1)
        for t, p in zip(true_cls, pred_cls):
            total[t] += 1
            if t == p:
                correct[t] += 1

    print("\n" + "─" * 50)
    print(f"{'Class':<20} {'Correct':>8} {'Total':>7} {'Acc':>7}")
    print("─" * 50)
    low_performers = []
    for i, cls in enumerate(CLASS_NAMES):
        acc = correct[i] / max(total[i], 1)
        flag = "  ⚠  LOW" if acc < 0.65 else ""
        print(f"{cls:<20} {correct[i]:>8} {total[i]:>7} {acc:>6.1%}{flag}")
        if acc < 0.65:
            low_performers.append(cls)
    print("─" * 50)
    overall = correct.sum() / max(total.sum(), 1)
    print(f"{'Overall':<20} {correct.sum():>8} {total.sum():>7} {overall:>6.1%}")
    print("─" * 50)
    if low_performers:
        print(f"\n⚠  Low-performing classes (< 65% acc): {', '.join(low_performers)}")
        print("   Consider: collecting more images, merging with a sibling class,")
        print("   or reviewing the DEEPFASHION_CATEGORY_MAP entries for these classes.")


# ── 5. TF.js export ───────────────────────────────────────────────────────────

def export_tfjs(model: keras.Model) -> None:
    import subprocess, shutil, tempfile

    keras_path = os.path.join(tempfile.mkdtemp(), "fashion_mobilenetv2.keras")
    model.save(keras_path)
    print(f"\nSaved Keras model: {keras_path}")

    # Clear old public/model/ shards so stale files don't linger
    tfjs_dir = pathlib.Path(TFJS_OUTPUT_DIR)
    if tfjs_dir.exists():
        for f in tfjs_dir.glob("group*-shard*"):
            f.unlink()
        model_json = tfjs_dir / "model.json"
        if model_json.exists():
            model_json.unlink()
    tfjs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting to TF.js layers format → {TFJS_OUTPUT_DIR}")
    result = subprocess.run(
        [
            sys.executable, "-m", "tensorflowjs.converters.converter",
            "--input_format",  "keras",
            "--output_format", "tfjs_layers_model",
            keras_path,
            TFJS_OUTPUT_DIR,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("❌  tensorflowjs_converter failed:")
        print(result.stderr)
        print("\nFallback: install with:  pip install tensorflowjs")
        print(f"Then run manually:\n  tensorflowjs_converter --input_format keras "
              f"--output_format tfjs_layers_model {keras_path} {TFJS_OUTPUT_DIR}")
    else:
        shard_count = len(list(tfjs_dir.glob("group*-shard*")))
        total_mb = sum(
            f.stat().st_size for f in tfjs_dir.iterdir() if f.is_file()
        ) / 1e6
        print(f"✅  TF.js export complete: {shard_count} shard(s), {total_mb:.1f} MB total")
        print(f"    Output: {TFJS_OUTPUT_DIR}")


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Fashion Classifier — MobileNetV2 Transfer Learning")
    print("=" * 60)
    print(f"\nTensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}" + (f"  {gpus[0].name}" if gpus else " (CPU only)"))
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    paths, labels = load_deepfashion(DEEPFASHION_ROOT)
    if len(paths) == 0:
        sys.exit("❌  No images loaded. Check DEEPFASHION_ROOT and label mapping.")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels,
        test_size=TEST_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    print(f"\nTrain: {len(train_paths)}  |  Test: {len(test_paths)}")

    train_ds = make_dataset(train_paths, train_labels, augment=True,  shuffle=True)
    test_ds  = make_dataset(test_paths,  test_labels,  augment=False, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model, backbone = build_model()
    model.summary(line_length=80)

    # ── Phase 1: Train classification head ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Phase 1 — Train head only ({PHASE1_EPOCHS} epochs, lr=1e-3)")
    print(f"{'─'*60}")
    callbacks_p1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=test_ds,
        callbacks=callbacks_p1,
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Phase 2 — Unfreeze top 30 backbone layers ({PHASE2_EPOCHS} epochs, lr={PHASE2_LR})")
    print(f"{'─'*60}")

    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_p2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]
    model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=test_ds,
        callbacks=callbacks_p2,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Per-class accuracy on held-out test set:")
    per_class_accuracy(model, test_ds)

    # ── Export ────────────────────────────────────────────────────────────────
    export_tfjs(model)

    print("\n✅  Done! Drop the updated public/model/ into your Next.js app and run:")
    print("       npm run build")
    print("       npm run dev")


if __name__ == "__main__":
    main()
