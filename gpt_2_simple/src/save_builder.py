import tensorflow as tf
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.ops import io_ops

class CastFromFloat32SaverBuilder(BaseSaverBuilder):
  # Based on tensorflow.python.training.saver.BulkSaverBuilder.bulk_restore
  # Code from https://stackoverflow.com/a/56664852
  def bulk_restore(self, filename_tensor, saveables, preferred_shard, restore_sequentially):
    restore_specs = []
    for saveable in saveables:
      for spec in saveable.specs:
        restore_specs.append((spec.name, spec.slice_spec, spec.dtype))
    names, slices, dtypes = zip(*restore_specs)
    restore_dtypes = [tf.float32 for _ in dtypes]
    with tf.device("cpu:0"):
      restored = io_ops.restore_v2(filename_tensor, names, slices, restore_dtypes)
      return [tf.cast(r, dt) for r, dt in zip(restored, dtypes)]
