import numpy as np
from collections import deque
import js


class VAD():
    def __init__(self):
        # Create buffer
        self.prediction_buffer: deque = deque(maxlen=125)  # buffer length of 10 seconds

        
    def append(self, p):
        self.prediction_buffer.append(p)
        # js.window.console.log('appending: ', p);

