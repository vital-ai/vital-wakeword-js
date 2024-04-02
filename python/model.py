import numpy as np
import functools
import pickle
from collections import deque, defaultdict
from functools import partial
import time
from typing import List, Union, DefaultDict, Dict
import js
# from utils import AudioFeatures

# Define main model class
class Model():
    def __init__(
            self,
            wakeword_models: List[str] = [],
            class_mapping_dicts: List[dict] = []
    ):
        """Initialize the openWakeWord model object.

        Args:
            wakeword_models (List[str]): A list of paths of ONNX/tflite models to load into the openWakeWord model object.
                                              If not provided, will load all of the pre-trained models. Alternatively,
                                              just the names of pre-trained models can be provided to select a subset of models.
            class_mapping_dicts (List[dict]): A list of dictionaries with integer to string class mappings for
                                              each model in the `wakeword_models` arguments
                                              (e.g., {"0": "class_1", "1": "class_2"})
            enable_speex_noise_suppression (bool): Whether to use the noise suppresion from the SpeexDSP
                                                   library to pre-process all incoming audio. May increase
                                                   model performance when reasonably stationary background noise
                                                   is present in the environment where openWakeWord will be used.
                                                   It is very lightweight, so enabling it doesn't significantly
                                                   impact efficiency.
            vad_threshold (float): Whether to use a voice activity detection model (VAD) from Silero
                                   (https://github.com/snakers4/silero-vad) to filter predictions.
                                   For every input audio frame, a VAD score is obtained and only those model predictions
                                   with VAD scores above the threshold will be returned. The default value (0),
                                   disables voice activity detection entirely.
            custom_verifier_models (dict): A dictionary of paths to custom verifier models, where
                                           the keys are the model names (corresponding to the openwakeword.MODELS
                                           attribute) and the values are the filepaths of the
                                           custom verifier models.
            custom_verifier_threshold (float): The score threshold to use a custom verifier model. If the score
                                               from a model for a given frame is greater than this value, the
                                               associated custom verifier model will also predict on that frame, and
                                               the verifier score will be returned.
            inference_framework (str): The inference framework to use when for model prediction. Options are
                                       "tflite" or "onnx". The default is "tflite" as this results in better
                                       efficiency on common platforms (x86, ARM64), but in some deployment
                                       scenarios ONNX models may be preferable.
            kwargs (dict): Any other keyword arguments to pass the the preprocessor instance
        """
        # Create attributes to store models and metadata
        self.model_inputs = {}
        self.model_outputs = {}
        self.class_mapping = {}

        self.model_inputs['hey_haley'] = 16 # 'onnx::Flatten_0'
        
        self.model_outputs['hey_haley'] = 1 # '39' 
        
        self.class_mapping['hey_haley'] = 'hey_haley' 
        
        """
        for mdl_path, mdl_name in zip(wakeword_models, wakeword_model_names):
            # Load openwakeword models
            if inference_framework == "onnx":
                
                sessionOptions = ort.SessionOptions()
                sessionOptions.inter_op_num_threads = 1
                sessionOptions.intra_op_num_threads = 1

                self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions,
                                                             providers=["CPUExecutionProvider"])

                self.model_inputs[mdl_name] = self.models[mdl_name].get_inputs()[0].shape[1]
                self.model_outputs[mdl_name] = self.models[mdl_name].get_outputs()[0].shape[1]
                pred_function = functools.partial(onnx_predict, self.models[mdl_name])
                self.model_prediction_function[mdl_name] = pred_function

            if class_mapping_dicts and class_mapping_dicts[wakeword_models.index(mdl_path)].get(mdl_name, None):
                self.class_mapping[mdl_name] = class_mapping_dicts[wakeword_models.index(mdl_path)]
            elif openwakeword.model_class_mappings.get(mdl_name, None):
                self.class_mapping[mdl_name] = openwakeword.model_class_mappings[mdl_name]
            else:
                self.class_mapping[mdl_name] = {str(i): str(i) for i in range(0, self.model_outputs[mdl_name])}
        """
        
        # Create buffer to store frame predictions
        self.prediction_buffer: DefaultDict[str, deque] = defaultdict(partial(deque, maxlen=30))
        
        # Initialize Silero VAD
        # self.vad_threshold = vad_threshold
        # self.vad = VAD()
        # if vad_threshold > 0:
        #    self.vad = VAD()
        # Create AudioFeatures object
        self.preprocessor = AudioFeatures()

    async def init(self):
        await self.preprocessor.init()

        
    def get_parent_model_from_label(self, label):
        """Gets the parent model associated with a given prediction label"""
        parent_model = ""
        for mdl in self.class_mapping.keys():
            if label in self.class_mapping[mdl].values():
                parent_model = mdl
            elif label in self.class_mapping.keys() and label == mdl:
                parent_model = mdl

        return parent_model

    async def reset(self):
        """Reset the prediction and audio feature buffers. Useful for re-initializing the model, though may not be efficient
        when called too frequently."""
        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))
        await self.preprocessor.reset()

    async def predict_js(self, x_js):
        x_np = np.array(x_js.to_py())
        return await self.predict(x_np)
        
    async def predict(self, x: np.ndarray):
        """Predict with all of the wakeword models on the input audio frames

        Args:
            x (ndarray): The input audio data to predict on with the models. Ideally should be multiples of 80 ms
                                (1280 samples), with longer lengths reducing overall CPU usage
                                but decreasing detection latency. Input audio with durations greater than or less
                                than 80 ms is also supported, though this will add a detection delay of up to 80 ms
                                as the appropriate number of samples are accumulated.
            patience (dict): How many consecutive frames (of 1280 samples or 80 ms) above the threshold that must
                             be observed before the current frame will be returned as non-zero.
                             Must be provided as an a dictionary where the keys are the
                             model names and the values are the number of frames. Can reduce false-positive
                             detections at the cost of a lower true-positive rate.
                             By default, this behavior is disabled.
            threshold (dict): The threshold values to use when the `patience` or `debounce_time` behavior is enabled.
                              Must be provided as an a dictionary where the keys are the
                              model names and the values are the thresholds.
            debounce_time (float): The time (in seconds) to wait before returning another non-zero prediction
                                   after a non-zero prediction. Can preven multiple detections of the same wake-word.
            timing (bool): Whether to return timing information of the models. Can be useful to debug and
                           assess how efficiently models are running on the current hardware.

        Returns:
            dict: A dictionary of scores between 0 and 1 for each model, where 0 indicates no
                  wake-word/wake-phrase detected. If the `timing` argument is true, returns a
                  tuple of dicts containing model predictions and timing information, respectively.
        """
        # Check input data type
        if not isinstance(x, np.ndarray):
            raise ValueError(f"The input audio data (x) must by a Numpy array, instead received an object of type {type(x)}.")

        n_prepared_samples = await self.preprocessor.streaming_features(x)

        # Get predictions from model(s)
        predictions = {}
        
        wakeWordPredict = js.wakeWordPredict

        mdl = 'hey_haley'
        
        # Run model to get predictions
        if n_prepared_samples > 1280:
            group_predictions = []
            for i in np.arange(n_prepared_samples//1280-1, -1, -1):
                prediction = np.array((await wakeWordPredict(
                        self.preprocessor.get_features(
                                self.model_inputs[mdl],
                                start_ndx=-self.model_inputs[mdl] - i
                        ).flatten()
                    ) ).to_py())
                
                if prediction[0] > 0.01:
                    js.console.log('hey haley score group: ' + str(prediction[0]))

                group_predictions.extend(
                    [[prediction]]
                )
            prediction = np.array(group_predictions).max(axis=0)[None, ]
        elif n_prepared_samples == 1280:
            prediction = np.array((await wakeWordPredict(
                self.preprocessor.get_features(self.model_inputs[mdl]).flatten()
            )).to_py())
            
            if prediction[0] > 0.01:
                js.console.log('hey haley score: ' + str(prediction[0]))
            
            prediction = [[prediction]]
        elif n_prepared_samples < 1280:  # get previous prediction if there aren't enough samples
            if self.model_outputs[mdl] == 1:
                if len(self.prediction_buffer[mdl]) > 0:
                    prediction = [[[self.prediction_buffer[mdl][-1]]]]
                else:
                    prediction = [[[0]]]
            elif self.model_outputs[mdl] != 1:
                n_classes = max([int(i) for i in self.class_mapping[mdl].keys()])
                prediction = [[[0]*(n_classes+1)]]

        # note: need to clean up return of score value
        if len(self.prediction_buffer[mdl]) < 5:
            predictions[mdl] = 0
        else:
            predictions[mdl] = prediction # [0][0][0]
        
        self.prediction_buffer[mdl].append(predictions[mdl])

        return predictions