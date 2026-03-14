import os
import json
import tensorflow as tf
from vision_desktop_app.training.model_templates import classification_model, detection_model_simple
from vision_desktop_app.training.dataset_utils import parse_dataset_from_folder


class Trainer:
    def __init__(self, model_type='classification', model_dir='models/default', input_shape=(128,128,3), num_classes=2, logger=None):
        self.model_type = model_type
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.logger = logger

    def build_model(self):
        if self.model_type == 'classification':
            self.model = classification_model(input_shape=self.input_shape, num_classes=self.num_classes)
        elif self.model_type == 'detection':
            self.model = detection_model_simple(input_shape=self.input_shape, num_classes=self.num_classes)
        else:
            raise ValueError('Unknown model type')

    def train(self, data_dir, epochs=10, batch_size=16):
        if self.model is None:
            self.build_model()
        ds = parse_dataset_from_folder(data_dir, image_size=self.input_shape[:2], classification=(self.model_type=='classification'))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # simple training loop
        if self.model_type == 'classification':
            callbacks = []
            if self.logger:
                class LoggerCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        self._logger = self._logger if hasattr(self, '_logger') else None
                cb = CallbackLogger(self.logger)
                callbacks.append(cb)
            try:
                self.model.fit(ds, epochs=epochs, callbacks=callbacks)
            except Exception as e:
                if self.logger:
                    self.logger(f'Training failed: {e}')
                raise
        else:
            # detection: ds elements are (img, (label, bbox)) but Keras won't know how to unpack easily; transform
            def map_fn(img, t):
                label, bbox = t
                return img, {'class_out': label, 'bbox_out': bbox}
            ds_mapped = ds.map(map_fn)
            callbacks = []
            if self.logger:
                cb = CallbackLogger(self.logger)
                callbacks.append(cb)
            try:
                self.model.fit(ds_mapped, epochs=epochs, callbacks=callbacks)
            except Exception as e:
                if self.logger:
                    self.logger(f'Training failed: {e}')
                raise
        # save model
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            save_path = os.path.join(self.model_dir, 'saved_model')
            try:
                # Try the high-level Keras save first
                self.model.save(save_path)
                if self.logger:
                    self.logger(f'Model saved to: {save_path} (via model.save)')
            except Exception as e_save:
                # If Keras complains about filepath extension, try saving as SavedModel directory
                if self.logger:
                    self.logger(f'Primary save failed: {e_save} — attempting tf.saved_model.save fallback')
                try:
                    tf.saved_model.save(self.model, save_path)
                    if self.logger:
                        self.logger(f'Model saved to: {save_path} (via tf.saved_model.save)')
                except Exception as e_savedmodel:
                    # As a last resort, try saving in Keras single-file format with .keras extension
                    alt_path = save_path + '.keras'
                    try:
                        self.model.save(alt_path)
                        if self.logger:
                            self.logger(f'Model saved to: {alt_path} (via model.save .keras)')
                    except Exception as e_alt:
                        # log full chain of errors and re-raise the last
                        if self.logger:
                            self.logger(f'Failed to save model: primary error: {e_save}; saved_model fallback error: {e_savedmodel}; .keras fallback error: {e_alt}')
                        raise
        except Exception:
            # propagate to caller after logging handled above
            raise


class CallbackLogger(tf.keras.callbacks.Callback):
    def __init__(self, cb_fn):
        super().__init__()
        self.cb_fn = cb_fn

    def on_epoch_end(self, epoch, logs=None):
        self.cb_fn(f"Epoch {epoch+1}: {logs}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='dataset folder'
    )
    p.add_argument('--model_type', default='classification', choices=['classification', 'detection'])
    p.add_argument('--model_dir', default='models/default')
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()

    t = Trainer(model_type=args.model_type, model_dir=args.model_dir)
    t.train(args.data, epochs=args.epochs)
