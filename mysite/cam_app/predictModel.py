import tensorflow as tf

model=None
def load_model():
    global model
    model = tf.keras.models.load_model('mysite/model.h5')
    print(("* Loading Keras model and Flask starting server..."
                    "please wait until server has fully started"))
