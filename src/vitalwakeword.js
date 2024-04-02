import Mic from './vitalwakeword/nodes/mic.js'
import Recorder from './vitalwakeword/nodes/recorder.js'
import Vad from './vitalwakeword/nodes/vad.js'
import DownSampler from './vitalwakeword/nodes/downsampler.js'
import SpeechPreemphaser from './vitalwakeword/nodes/speechpreemphasis.js'
import FeaturesExtractor from './vitalwakeword/nodes/features.js'
import Hotword from './vitalwakeword/nodes/hotword.js'
import WakeWord from './vitalwakeword/nodes/wakeword.js'


const vitalWakeWord = {
    DownSampler,
    Mic,
    SpeechPreemphaser,
    Vad,
    FeaturesExtractor,
    Hotword,
    Recorder,
    
    WakeWord
}

window.vitalWakeWord = vitalWakeWord
module.exports = vitalWakeWord