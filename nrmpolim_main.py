"""
Skeleton with prototype functions for polarimetric interferometry image reconstruction.
Methods here could be put in functions in separate files if preferred
"""

import numpy as np

class nrmpolim:
    def __init__(self, datadir):
        self.datadir = datadir


    def load_training_data(self, datafilename):
        """
        Load labelled simulated data from file(s)
        """

        # Cube of all simulated images for training, shape (num_simims, num_stokes, num_xpixels, num_ypixels),
        # where num_stokes = 4 (for Stokes I, Q, U and V images). V is currently 0 but may be populated later.
        self.ydata = np.array(None, dtype='float32')

        # Array of generation parameters of all simulated images, shape (num_simims, )
        self.ydata_genparams = None


    def get_diffv2cp_fromims(self, input_images=None):
        """
        Extract the polarised differential visibilities and closure phases from input_images
        """

        if input_images is None:
            input_images = self.ydata

        # Array of polarimetric differential visibilities of shape (num_simims, num_stokes, num_bls),
        # where num_stokes = 4 (for Stokes I, Q, U and V visbilities). V is currently None but may be populated later.
        Xdata_v2s = None

        # Array of polarimetric differential closure phases of shape (num_simims, num_stokes, num_cps),
        # where num_stokes = 4 (for Stokes I, Q, U and V closure phases). V is currently None but may be populated later.
        Xdata_cps = None

        self.Xdata = np.concatenate((Xdata_v2s,Xdata_cps), axis=2)


    def build_model(self, model_params):
        """
        Build a model based on model_params
        """

        model = model_constructor(model_params)
        self.model = model


    def train_model(self, training_hyperparams):
        """
        Train model using training_hyperparams
        """

        X_train, X_test, y_train, y_test = train_test_split(self.Xdata, self.ydata)
        self.model.train(X_train, X_test, y_train, y_test)




"""
Example workflow
"""

imr = nrmpolim(datadir)
imr.load_training_data(datafilename)
imr.get_diffv2cp_fromims()
imr.build_model(model_params)
imr.train_model(training_hyperparams)



