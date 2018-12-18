#!/usr/bin/python3
'''Copyright (c) 2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Train a LPCNet model (note not a Wavenet model)

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()

# use this option to reserve GPU memory, e.g. for running more than
# one thing at a time.  Best to disable for GPUs with small memory
config.gpu_options.per_process_gpu_memory_fraction = 0.44

set_session(tf.Session(config=config))

nb_epochs = 120

# Try reducing batch_size if you run out of memory on your GPU
batch_size = 64

model, _, _ = lpcnet.new_lpcnet_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

feature_file = sys.argv[1]
pcm_file = sys.argv[2]  # 16 bit unsigned short PCM samples
frame_size = 160
nb_features = 55
nb_used_features = model.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size * feature_chunk_size


class Dataset(object):
    def __init__(self, pcm_file, feature_file, batch_size):
        print("start loading data from files...")
        self._udata = np.fromfile(pcm_file, dtype='int16')
        self._features = np.fromfile(feature_file, dtype='float32')

        _nb_frames = len(self._udata) // pcm_chunk_size
        self.loopcount = _nb_frames // batch_size
        print("loopcount: ", self.loopcount)

        # limit to discrete number of frames
        self._udata = self._udata[:_nb_frames * pcm_chunk_size]
        self._features = self._features[:_nb_frames * feature_chunk_size * nb_features]

    def __len__(self):
        return self.loopcount

    def __iter__(self):
        return self

    def __next__(self):
        i = np.random.randint(0, self.loopcount)
        nb_frames = batch_size
        udata = self._udata[i * batch_size * pcm_chunk_size:
                            (i + 1) * batch_size * pcm_chunk_size]
        features = self._features[i * batch_size * feature_chunk_size * nb_features:
                                  (i + 1) * batch_size * feature_chunk_size * nb_features]

        data = lin2ulaw(udata)

        # Noise injection: the idea is that the real system is going to be
        # predicting samples based on previously predicted samples rather than
        # from the original. Since the previously predicted samples aren't
        # expected to be so good, I add noise to the training data.  Exactly
        # how the noise is added makes a huge difference

        in_data = np.concatenate([data[0:1], data[:-1]])
        noise = np.concatenate([np.zeros((len(data) * 1 // 5)), np.random.randint(-3, 3, len(data) * 1 // 5),
                                np.random.randint(-2, 2, len(data) * 1 // 5),
                                np.random.randint(-1, 1, len(data) * 2 // 5)])
        # noise = np.round(np.concatenate([np.zeros((len(data)*1//5)), np.random.laplace(0, 1.2, len(data)*1//5), np.random.laplace(0, .77, len(data)*1//5), np.random.laplace(0, .33, len(data)*1//5), np.random.randint(-1, 1, len(data)*1//5)]))
        in_data = in_data + noise
        in_data = np.clip(in_data, 0, 255)

        features = np.reshape(features, (nb_frames * feature_chunk_size, nb_features))

        # Note: the LPC predictor output is now calculated by the loop below, this code was
        # for an ealier version that implemented the prediction filter in C

        upred = np.zeros((nb_frames * pcm_chunk_size,), dtype='float32')

        # Use 16th order LPC to generate LPC prediction output upred[] and (in
        # mu-law form) pred[]

        pred_in = ulaw2lin(in_data)
        for i in range(2, nb_frames * feature_chunk_size):
            upred[i * frame_size:(i + 1) * frame_size] = 0
            for k in range(16):
                upred[i * frame_size:(i + 1) * frame_size] = upred[i * frame_size:(i + 1) * frame_size] - \
                                                             pred_in[i * frame_size - k:(i + 1) * frame_size - k] * \
                                                             features[
                                                                 i, nb_features - 16 + k]

        pred = lin2ulaw(upred)

        in_data = np.reshape(in_data, (nb_frames, pcm_chunk_size, 1))
        in_data = in_data.astype('uint8')

        # LPC residual, which is the difference between the input speech and
        # the predictor output, with a slight time shift this is also the
        # ideal excitation in_exc

        out_data = lin2ulaw(udata - upred)
        in_exc = np.concatenate([out_data[0:1], out_data[:-1]])

        out_data = np.reshape(out_data, (nb_frames, pcm_chunk_size, 1))
        out_data = out_data.astype('uint8')

        in_exc = np.reshape(in_exc, (nb_frames, pcm_chunk_size, 1))
        in_exc = in_exc.astype('uint8')

        features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
        features = features[:, :, :nb_used_features]
        features[:, :, 18:36] = 0
        pred = np.reshape(pred, (nb_frames, pcm_chunk_size, 1))
        pred = pred.astype('uint8')

        periods = (.1 + 50 * features[:, :, 36:37] + 100).astype('int16')

        in_data = np.concatenate([in_data, pred], axis=-1)

        return [in_data, in_exc, features, periods], out_data


# dump models to disk as we go
checkpoint = ModelCheckpoint('lpcnet15_384_10_G16_{epoch:02d}.h5')

# model.load_weights('lpcnet9b_384_10_G16_01.h5')
model.compile(optimizer=Adam(0.001, amsgrad=True, decay=5e-5), loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

g = Dataset(pcm_file, feature_file, batch_size)
model.fit_generator(g, epochs=nb_epochs,
                    steps_per_epoch=len(g),
                    callbacks=[checkpoint, lpcnet.Sparsify(2000, 40000, 400, (0.1, 0.1, 0.1))])
