In this assignment, the task was to implement DAN (Deep Averaging Network) and GRU (Gated Recurrent Unit).

In both the models, we apply masking to the inputs. In addition to this, in case of DAN, I have implemented dropout.

In probing model, we load the pretrained model and extract a specific layer from it and learn a classifier on top of that. This way we 
are able to learn how each of the layer can perform.
