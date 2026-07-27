import json
import os
from datetime import datetime

import tensorflow as tf

from . import config


def _state_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, "state.json")


def _load_state(checkpoint_dir):
    path = _state_path(checkpoint_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_epochs": 0, "done": False}


def _save_state(checkpoint_dir, state):
    with open(_state_path(checkpoint_dir), "w") as f:
        json.dump(state, f, indent=2)


def _append_history(checkpoint_dir, record):
    with open(os.path.join(checkpoint_dir, "history.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def train_in_sessions(model, train_ds, val_ds,
                       total_epochs=config.TOTAL_EPOCHS,
                       epochs_per_session=config.EPOCHS_PER_SESSION,
                       checkpoint_dir=config.CHECKPOINT_DIR,
                       final_model_dir=config.FINAL_MODEL_DIR,
                       callbacks=None,
                       stage_name="normal"):
    """
    Trains `model` on the full `train_ds` toward `total_epochs`, running only
    `epochs_per_session` epochs per call. Call this once per session (e.g.
    once per notebook run) — it resumes automatically from wherever it left
    off, and on reaching `total_epochs` it dumps the model into
    `final_model_dir` and marks itself done.
    """
    if callbacks is None:
        callbacks = []
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = _load_state(checkpoint_dir)

    if state["done"]:
        print(f"[{stage_name}] Already completed {state['completed_epochs']}/{total_epochs} epochs. "
              f"Final model at {config.FINAL_MODEL_PATH}")
        return model

    latest_ckpt = os.path.join(checkpoint_dir, "latest.keras")
    if state["completed_epochs"] > 0 and os.path.exists(latest_ckpt):
        print(f"[{stage_name}] Resuming from epoch {state['completed_epochs']}/{total_epochs}")
        model = tf.keras.models.load_model(latest_ckpt)
    else:
        print(f"[{stage_name}] Starting fresh training (target: {total_epochs} epochs)")

    initial_epoch = state["completed_epochs"]
    target_epoch = min(initial_epoch + epochs_per_session, total_epochs)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=target_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    record = {
        "stage": stage_name,
        "epochs_run": list(range(initial_epoch + 1, target_epoch + 1)),
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {k: v[-1] for k, v in history.history.items()},
    }
    _append_history(checkpoint_dir, record)
    print(f"[{stage_name}] Completed epochs {initial_epoch + 1}-{target_epoch}/{total_epochs} — "
          f"train loss: {record['metrics'].get('loss'):.4f}, "
          f"val loss: {record['metrics'].get('val_loss'):.4f}")

    model.save(latest_ckpt)
    state["completed_epochs"] = target_epoch

    if target_epoch >= total_epochs:
        os.makedirs(final_model_dir, exist_ok=True)
        model.save(config.FINAL_MODEL_PATH)
        state["done"] = True
        print(f"[{stage_name}] Reached {total_epochs} epochs — final model saved to {config.FINAL_MODEL_PATH}")

    _save_state(checkpoint_dir, state)
    return model