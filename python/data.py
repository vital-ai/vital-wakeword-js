import os
import re
import logging
from functools import partial
from pathlib import Path
import random
from typing import List, Tuple
import numpy as np
import itertools


# Load audio clips and structure into clips of the same length
def stack_clips(audio_data, clip_size=16000*2):
    """
    Takes an input list of 1D arrays (of different lengths), concatenates them together,
    and then extracts clips of a uniform size by dividing the combined array.

    Args:
        audio_data (List[ndarray]): A list of 1D numpy arrays to combine and stack
        clip_size (int): The desired total length of the uniform clip size (in samples)

    Returns:
        ndarray: A N by `clip_size` array with the audio data, converted to 16-bit PCM
    """

    # Combine all clips into single clip
    combined_data = np.hstack((audio_data))

    # Get chunks of the specified size
    new_examples = []
    for i in range(0, combined_data.shape[0], clip_size):
        chunk = combined_data[i:i+clip_size]
        if chunk.shape[0] != clip_size:
            chunk = np.hstack((chunk, np.zeros(clip_size - chunk.shape[0])))
        new_examples.append(chunk)

    return np.array(new_examples)




# Convert clips with sox
def _convert_clip(input_file, output_file, backend="ffmpeg"):
    if backend == "sox":
        cmd = f"sox \"{input_file}\" -G -r 16000 -c 1 -b 16 \"{output_file}\""
    elif backend == "ffmpeg":
        cmd = f"ffmpeg -y -i \"{input_file}\" -ar 16000 \"{output_file}\""
    os.system(cmd)
    return None

def get_frame_labels(combined_size, start, end, buffer=1):
    sequence_label = np.zeros(np.ceil((combined_size-12400)/1280).astype(int))
    frame_positions = np.arange(12400, combined_size, 1280)
    start_frame = np.argmin(abs(frame_positions - start))
    end_frame = np.argmin(abs(frame_positions - end))
    sequence_label[start_frame:start_frame+2] = 1
    sequence_label[end_frame-1:end_frame+1] = 1
    return sequence_label


def mix_clip(fg, bg, snr, start):
    fg_rms, bg_rms = fg.norm(p=2), bg.norm(p=2)
    snr = 10 ** (snr / 20)
    scale = snr * bg_rms / fg_rms
    bg[start:start + fg.shape[0]] = bg[start:start + fg.shape[0]] + scale*fg
    return bg / 2


def truncate_clip(x, max_size, method="truncate_start"):
    """
    Truncates and audio clip with the specified method

    Args:
        x (nd.array): An array of audio data
        max_size (int): The maximum size (in samples)
        method (str): Can be one of four options:
            - "truncate_start": Truncate the start of the clip
            - "truncate_end": Truncate the end of the clip
            - "truncate_both": Truncate both the start and end of the clip
            - "random": Randomly select a segment of the right size from the clip

    Returns:
        nd.array: The truncated audio data
    """
    if x.shape[0] > max_size:
        if method == "truncate_start":
            x = x[x.shape[0] - max_size:]
        if method == "truncate_end":
            x = x[0:max_size]
        if method == "truncate_both":
            n = int(np.ceil(x.shape[0] - max_size)/2)
            x = x[n:-n][0:max_size]
        if method == "random":
            rn = np.random.randint(0, x.shape[0] - max_size)
            x = x[rn:rn + max_size]

    return x



def create_fixed_size_clip(x, n_samples, sr=16000, start=None, end_jitter=.200):
    """
    Create a fixed-length clip of the specified size by padding an input clip with zeros
    Optionally specify the start/end position of the input clip, or let it be chosen randomly.

    Args:
        x (ndarray): The input audio to pad to a fixed size
        n_samples (int): The total number of samples for the fixed length clip
        sr (int): The sample rate of the audio
        start (int): The start position of the clip in the fixed length output, in samples (default: None)
        end_jitter (float): The time (in seconds) from the end of the fixed length output
                            that the input clip should end, if `start` is None.

    Returns:
        ndarray: A new array of audio data of the specified length
    """
    dat = np.zeros(n_samples)
    end_jitter = int(np.random.uniform(0, end_jitter)*sr)
    if start is None:
        start = max(0, n_samples - (int(len(x))+end_jitter))

    if len(x) > n_samples:
        if np.random.random() >= 0.5:
            dat = x[0:n_samples].numpy()
        else:
            dat = x[-n_samples:].numpy()
    else:
        dat[start:start+len(x)] = x

    return dat


# Load batches of data from mmaped numpy arrays
class mmap_batch_generator:
    """
    A generator class designed to dynamically build batches from mmaped numpy arrays.

    The generator will return tuples of (data, labels) with a batch size determined
    by the `n_per_class` initialization argument. When a mmaped numpy array has been
    fully interated over, it will restart at the zeroth index automatically.
    """
    def __init__(self,
                 data_files: dict,
                 label_files: dict = {},
                 batch_size: int = 128,
                 n_per_class: dict = {},
                 data_transform_funcs: dict = {},
                 label_transform_funcs: dict = {}
                 ):
        """
        Initialize the generator object

        Args:
            data_files (dict): A dictionary of labels (as keys) and on-disk numpy array paths (as values).
                               Keys should be integer strings representing class labels.
            label_files (dict): A dictionary where the keys are the class labels and the values are the per-example
                                labels. The values must be the same shape as the correponding numpy data arrays
                                from the `data_files` argument.
            batch_size (int): The number of samples per batch
            n_per_class (dict): A dictionary with integer string labels (as keys) and number of example per batch
                               (as values). If None (the default), batch sizes for each class will be
                               automatically calculated based on the the input dataframe shapes and transformation
                               functions.

            data_transform_funcs (dict): A dictionary of transformation functions to apply to each batch of per class
                                    data loaded from the mmaped array. For example, with an array of shape
                                    (batch, timesteps, features), if the goal is to half the timesteps per example,
                                    (effectively doubling the size of the batch) this function could be passed:

                                    lambda x: np.vstack(
                                        (x[:, 0:timesteps//2, :], x[:, timesteps//2:, :]
                                    ))

                                    The user should incorporate the effect of any transform on the values of the
                                    `n_per_class` argument accordingly, in order to end of with the desired
                                    total batch size for each iteration of the generator.
            label_transform_funcs (dict): A dictionary of transformation functions to apply to each batch of labels.
                                          For example, strings can be mapped to integers or one-hot encoded,
                                          groups of classes can be merged together into one, etc.
        """
        # inputs
        self.data_files = data_files
        self.label_files = label_files
        self.n_per_class = n_per_class
        self.data_transform_funcs = data_transform_funcs
        self.label_transform_funcs = label_transform_funcs

        # Get array mmaps and store their shapes (but load files < 1 GB total size into memory)
        self.data = {label: np.load(fl, mmap_mode='r') for label, fl in data_files.items()}
        self.labels = {label: np.load(fl) for label, fl in label_files.items()}
        self.data_counter = {label: 0 for label in data_files.keys()}
        self.original_shapes = {label: self.data[label].shape for label in self.data.keys()}
        self.shapes = {label: self.data[label].shape for label in self.data.keys()}

        # # Update effective shape of mmap array based on user-provided transforms (currently broken)
        # for lbl, f in self.data_transform_funcs.items():
        #     dummy_data = np.random.random((1, self.original_shapes[lbl][1], self.original_shapes[lbl][2]))
        #     new_shape = f(dummy_data).shape
        #     self.shapes[lbl] = (new_shape[0]*self.original_shapes[lbl][0], new_shape[1], new_shape[2])

        # Calculate batch sizes, if the user didn't specify them
        scale_factor = 1
        if not self.n_per_class:
            self.n_per_class = {}
            for lbl, shape in self.shapes.items():
                dummy_data = np.random.random((10, self.shapes[lbl][1], self.shapes[lbl][2]))
                if self.data_transform_funcs.get(lbl, None):
                    scale_factor = self.data_transform_funcs.get(lbl, None)(dummy_data).shape[0]/10

                ratio = self.shapes[lbl][0]/sum([i[0] for i in self.shapes.values()])
                self.n_per_class[lbl] = max(1, int(int(batch_size*ratio)/scale_factor))

            # Get estimated batches per epoch, including the effect of any user-provided transforms
            batch_size = sum([val*scale_factor for val in self.n_per_class.values()])
            batches_per_epoch = sum([i[0] for i in self.shapes.values()])//batch_size
            self.batch_per_epoch = batches_per_epoch
            print("Batches/steps per epoch:", batches_per_epoch)

    def __iter__(self):
        return self

    def __next__(self):
        # Build batch
        while True:
            X, y = [], []
            for label, n in self.n_per_class.items():
                # Restart at zeroth index if an array reaches the end
                if self.data_counter[label] >= self.shapes[label][0]:
                    self.data_counter[label] = 0

                # Get data from mmaped file
                x = self.data[label][self.data_counter[label]:self.data_counter[label]+n]
                self.data_counter[label] += x.shape[0]

                # Transform data
                if self.data_transform_funcs and self.data_transform_funcs.get(label):
                    x = self.data_transform_funcs[label](x)

                # Make labels for data (following whatever the current shape of `x` is)
                if self.label_files.get(label, None):
                    y_batch = self.labels[label][self.data_counter[label]:self.data_counter[label]+n]
                else:
                    y_batch = [label]*x.shape[0]

                # Transform labels
                if self.label_transform_funcs and self.label_transform_funcs.get(label):
                    y_batch = self.label_transform_funcs[label](y_batch)

                # Add data to batch
                X.append(x)
                y.extend(y_batch)

            return np.vstack(X), np.array(y)


# Generate words that sound similar ("adversarial") to the input phrase using phoneme overlap
def generate_adversarial_texts(input_text: str, N: int, include_partial_phrase: float = 0, include_input_words: float = 0):
    """
    Generate adversarial words and phrases based on phoneme overlap.
    Currently only works for english texts.
    Note that homophones are excluded, as this wouldn't actually be an adversarial example for the input text.

    Args:
        input_text (str): The target text for adversarial phrases
        N (int): The total number of adversarial texts to return. Uses sampling,
                 so not all possible combinations will be included and some duplicates
                 may be present.
        include_partial_phrase (float): The probability of returning a number of words less than the input
                                        text (but always between 1 and the number of input words)
        include_input_words (float): The probability of including individual input words in the adversarial
                                     texts when the input text consists of multiple words. For example,
                                     if the `input_text` was "ok google", then setting this value > 0.0
                                     will allow for adversarial texts like "ok noodle", versus the word "ok"
                                     never being present in the adversarial texts.

    Returns:
        list: A list of strings corresponding to words and phrases that are phonetically similar (but not identical)
              to the input text.
    """
    # Get phonemes for english vowels (CMUDICT labels)
    vowel_phones = ["AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "UX"]

    word_phones = []
    input_text_phones = [pronouncing.phones_for_word(i) for i in input_text.split()]

    # Download phonemizer model for OOV words, if needed
    if [] in input_text_phones:
        phonemizer_mdl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "en_us_cmudict_forward.pt")
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")):
            os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources"))
        if not os.path.exists(phonemizer_mdl_path):
            logging.warning("Downloading phonemizer model from DeepPhonemizer library...")
            import requests
            file_url = "https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt"
            r = requests.get(file_url, stream=True)
            with open(phonemizer_mdl_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2048):
                    if chunk:
                        f.write(chunk)

        # Create phonemizer object
        from dp.phonemizer import Phonemizer
        phonemizer = Phonemizer.from_checkpoint(phonemizer_mdl_path)

    for phones, word in zip(input_text_phones, input_text.split()):
        if phones != []:
            word_phones.extend(phones)
        elif phones == []:
            logging.warning(f"The word '{word}' was not found in the pronunciation dictionary! "
                            "Using the DeepPhonemizer library to predict the phonemes.")
            phones = phonemizer(word, lang='en_us')
            logging.warning(f"Phones for '{word}': {phones}")
            word_phones.append(re.sub(r"[\]|\[]", "", re.sub(r"\]\[", " ", phones)))
        elif isinstance(phones[0], list):
            logging.warning(f"There are multiple pronunciations for the word '{word}'.")
            word_phones.append(phones[0])

    # add all possible lexical stresses to vowels
    word_phones = [re.sub('|'.join(vowel_phones), lambda x: str(x.group(0)) + '[0|1|2]', re.sub(r'\d+', '', i)) for i in word_phones]

    adversarial_phrases = []
    for phones, word in zip(word_phones, input_text.split()):
        query_exps = []
        phones = phones.split()
        adversarial_words = []
        if len(phones) <= 2:
            query_exps.append(" ".join(phones))
        else:
            query_exps.extend(phoneme_replacement(phones, max_replace=max(0, len(phones)-2), replace_char="(.){1,3}"))

        for query in query_exps:
            matches = pronouncing.search(query)
            matches_phones = [pronouncing.phones_for_word(i)[0] for i in matches]
            allowed_matches = [i for i, j in zip(matches, matches_phones) if j != phones]
            adversarial_words.extend([i for i in allowed_matches if word.lower() != i])

        if adversarial_words != []:
            adversarial_phrases.append(adversarial_words)

    # Build combinations for final output
    adversarial_texts = []
    for i in range(N):
        txts = []
        for j, k in zip(adversarial_phrases, input_text.split()):
            if np.random.random() > (1 - include_input_words):
                txts.append(k)
            else:
                txts.append(np.random.choice(j))

        if include_partial_phrase is not None and len(input_text.split()) > 1 and np.random.random() <= include_partial_phrase:
            n_words = np.random.randint(1, len(input_text.split())+1)
            adversarial_texts.append(" ".join(np.random.choice(txts, size=n_words, replace=False)))
        else:
            adversarial_texts.append(" ".join(txts))

    # Remove any exact matches to input phrase
    adversarial_texts = [i for i in adversarial_texts if i != input_text]

    return adversarial_texts


def phoneme_replacement(input_chars, max_replace, replace_char='"(.){1,3}"'):
    results = []
    chars = list(input_chars)

    # iterate over the number of characters to replace (1 to max_replace)
    for r in range(1, max_replace+1):
        # get all combinations for a fixed r
        comb = itertools.combinations(range(len(chars)), r)
        for indices in comb:
            chars_copy = chars.copy()
            for i in indices:
                chars_copy[i] = replace_char
            results.append(' '.join(chars_copy))

    return results
