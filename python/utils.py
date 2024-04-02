import numpy as np
from collections import deque
from typing import Union, List, Callable, Deque
import js

# Base class for computing audio features using Google's speech_embedding
# model (https://tfhub.dev/google/speech_embedding/1)
class AudioFeatures:
    """
    A class for creating audio features from audio data, including melspectograms and Google's
    `speech_embedding` features.
    """
    def __init__(self,
                 sr: int = 16000
                ):
        """
        Initialize the AudioFeatures object.

        Args:
            melspec_model_path (str): The path to the model for computing melspectograms from audio data
            embedding_model_path (str): The path to the model for Google's `speech_embedding` model
            sr (int): The sample rate of the audio (default: 16000 khz)
            ncpu (int): The number of CPUs to use when computing melspectrograms and audio features (default: 1)
            inference_framework (str): The inference framework to use when for model prediction. Options are
                                       "tflite" or "onnx". The default is "tflite" as this results in better
                                       efficiency on common platforms (x86, ARM64), but in some deployment
                                       scenarios ONNX models may be preferable.
            device (str): The device to use when running the models, either "cpu" or "gpu" (default is "cpu".)
                          Note that depending on the inference framework selected and system configuration,
                          this setting may not have an effect. For example, to use a GPU with the ONNX
                          framework the appropriate onnxruntime package must be installed.
        """
        # Initialize the models with the appropriate framework
        
        """
           
        self.melspec_model_predict = lambda x: self.melspec_model.run(None, {'input': x})

        self.embedding_model_predict = lambda x: self.embedding_model.run(None, {'input_1': x})[0].squeeze()
        
        """
            
        # Create databuffers with empty/random data
        self.raw_data_buffer: Deque = deque(maxlen=sr*10)
        self.melspectrogram_buffer = np.ones((76, 32))  # n_frames x num_features
        self.melspectrogram_max_len = 10*97  # 97 is the number of frames in 1 second of 16hz audio
        self.accumulated_samples = 0  # the samples added to the buffer since the audio preprocessor was last called
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer_max_len = 120  # ~10 seconds of feature buffer history

    async def init(self):
        get_embeddings = await self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))
        get_embeddings = np.array(get_embeddings)
        self.feature_buffer = get_embeddings
        
    async def reset(self):
        """Reset the internal buffers"""
        self.raw_data_buffer.clear()
        self.melspectrogram_buffer = np.ones((76, 32))
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = await self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))

    async def _get_melspectrogram(self, x: Union[np.ndarray, List], melspec_transform: Callable = lambda x: x/10 + 2):
        """
        Function to compute the mel-spectrogram of the provided audio samples.

        Args:
            x (Union[np.ndarray, List]): The input audio data to compute the melspectrogram from
            melspec_transform (Callable): A function to transform the computed melspectrogram. Defaults to a transform
                                          that makes the ONNX melspectrogram model closer to the native Tensorflow
                                          implementation from Google (https://tfhub.dev/google/speech_embedding/1).

        Return:
            np.ndarray: The computed melspectrogram of the input audio data
        """
    
        # Get input data and adjust type/shape as needed
        x = np.array(x).astype(np.int16) if isinstance(x, list) else x
        if x.dtype != np.int16:
            raise ValueError("Input data must be 16-bit integers (i.e., 16-bit PCM audio)."
                             f"You provided {x.dtype} data.")
        x = x[None, ] if len(x.shape) < 2 else x
        x = x.astype(np.float32) if x.dtype != np.float32 else x
        
        melspecModelPredict = js.melspecModelPredict
        
        js_outputs = await melspecModelPredict(x.flatten())
        
        outputs = np.array(js_outputs.to_py())

        outputs = outputs.reshape(-1, 32)
        
        spec = np.squeeze(outputs)

        # Arbitrary transform of melspectrogram
        spec = melspec_transform(spec)

        return spec

    async def _get_embeddings(self, x: np.ndarray, window_size: int = 76, step_size: int = 8):
        """Function to compute the embeddings of the provide audio samples."""
        spec = await self._get_melspectrogram(x)
                
        windows = []
        
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i+window_size]
            if window.shape[0] == window_size:  # truncate short windows
                windows.append(window)

        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
                        
        embeddingModelPredict = js.embeddingModelPredict

        batch_count = batch.shape[0]
        
        embedding = (await embeddingModelPredict(batch_count, batch.flatten())).to_py()
        
        embedding = np.array(embedding)
        
        embedding = embedding.reshape(-1, 96)
        
        return embedding

    async def _streaming_melspectrogram(self, n_samples):
        """Note! There seem to be some slight numerical issues depending on the underlying audio data
        such that the streaming method is not exactly the same as when the melspectrogram of the entire
        clip is calculated. It's unclear if this difference is significant and will impact model performance.
        In particular padding with 0 or very small values seems to demonstrate the differences well.
        """
        
        if len(self.raw_data_buffer) < 400:
            raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")

        melspec_input = list(self.raw_data_buffer)[-n_samples-160*3:]    
                
        melspec = await self._get_melspectrogram(melspec_input) 
                
        self.melspectrogram_buffer = np.vstack((self.melspectrogram_buffer, melspec ))
        
        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

    def _buffer_raw_data(self, x):
        """
        Adds raw audio data to the input buffer
        """
        self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)

        
    async def _streaming_features(self, x):
        
        # Add raw audio data to buffer, temporarily storing extra frames if not an even number of 80 ms chunks
        processed_samples = 0

        if self.raw_data_remainder.shape[0] != 0:
            x = np.concatenate((self.raw_data_remainder, x))
            self.raw_data_remainder = np.empty(0)

        if self.accumulated_samples + x.shape[0] >= 1280:
            remainder = (self.accumulated_samples + x.shape[0]) % 1280
            if remainder != 0:
                x_even_chunks = x[0:-remainder]
                self._buffer_raw_data(x_even_chunks)
                self.accumulated_samples += len(x_even_chunks)
                self.raw_data_remainder = x[-remainder:]
            elif remainder == 0:
                self._buffer_raw_data(x)
                self.accumulated_samples += x.shape[0]
                self.raw_data_remainder = np.empty(0)
        else:
            self.accumulated_samples += x.shape[0]
            self._buffer_raw_data(x)

        embeddingModelPredict = js.embeddingModelPredict
    
        # Only calculate melspectrogram once minimum samples are accumulated
        if self.accumulated_samples >= 1280 and self.accumulated_samples % 1280 == 0:
            await self._streaming_melspectrogram(self.accumulated_samples)

            # Calculate new audio embeddings/features based on update melspectrograms
            for i in np.arange(self.accumulated_samples//1280-1, -1, -1):
                ndx = -8*i
                ndx = ndx if ndx != 0 else len(self.melspectrogram_buffer)
                x = self.melspectrogram_buffer[-76 + ndx:ndx].astype(np.float32)[None, :, :, None]
                if x.shape[1] == 76:
                    # js.console.log('embedding shape: ' + str(x.shape))
                    embed_prediction_js = await embeddingModelPredict(1, x.flatten())
                    embed_prediction = embed_prediction_js.to_py()
                    embed_prediction_array = np.array(embed_prediction)
                    
                    self.feature_buffer = np.vstack((self.feature_buffer, embed_prediction_array))

            # Reset raw data buffer counter
            processed_samples = self.accumulated_samples
            self.accumulated_samples = 0

        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]

        return processed_samples if processed_samples != 0 else self.accumulated_samples
        
    def get_features(self, n_feature_frames: int = 16, start_ndx: int = -1):
        if start_ndx != -1:
            end_ndx = start_ndx + int(n_feature_frames) \
                if start_ndx + n_feature_frames != 0 else len(self.feature_buffer)
            return self.feature_buffer[start_ndx:end_ndx, :][None, ].astype(np.float32)
        else:
            return self.feature_buffer[int(-1*n_feature_frames):, :][None, ].astype(np.float32)

    async def streaming_features(self, x):
        return await self._streaming_features(x)
        