import gpt_2_simple as gpt_2
import tensorflow as tf

model_name = "124M"
gpt_2.download_gpt2(model_name=model_name)

sess = gpt_2.start_tf_sess()
gpt_2.finetune(sess, 
              '/mnt/m.2/calvin-terminator/romans-com.txt', 
              model_name=model_name, 
              steps=300,
              dtype=tf.compat.v1.dtypes.float32)