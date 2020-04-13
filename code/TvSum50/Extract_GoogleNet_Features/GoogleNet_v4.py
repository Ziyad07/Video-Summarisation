from keras.models import Model
from keras.models import model_from_json
import os

class GoogleNet_v4(object):

        def __init__(self):
                model_dir = ''
                model_weight_filename = os.path.join(model_dir, '../../../models/GoogleLeNet_v4/inception-v4_weights_tf_dim_ordering_tf_kernels.h5')
                model_json_filename = os.path.join(model_dir, '../../../models/GoogleLeNet_v4/model.json')

                self.model = model_from_json(open(model_json_filename, 'r').read())
                self.model.load_weights(model_weight_filename)
                self.model.compile(loss='mean_squared_error', optimizer='sgd')

        def infer(self, X):
                return self.model.predict_on_batch(X)

        def infer_layer(self, X, n=-2):
                f = Model(self.model.input, self.model.layers[n].output)
                return f.predict(X)