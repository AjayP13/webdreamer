import re

import numpy as np
from symai import Symbol


# extract observation and create a sequence of symbols
def to_symbol(sequence):
    symbol_sequence = []
    for t in sequence:
        # extract observation from prompt
        matches = re.findall("(^Now make prediction given the observation\n\nObservation\n:OBSERVATION:$)(.*)", t["prompt"], flags=re.DOTALL|re.MULTILINE)
        if len(matches) == 1:
            observation = matches[0][1].strip()
        else:
            raise ValueError("Observation not found in prompt")
        
        # concatenate observation and response (action) and create a Symbol
        symbol_sequence.append(Symbol(observation + "\n" + t["response"]))
    return symbol_sequence
   

# embed a sequence of symbols using the registered embedding model
def embed_sequence(sequence):
    return [np.array(s.embed().value[0]) for s in sequence]


# set the default measure
def measure(self, other, normalize=None):
    # use distance measure as the default measure
    res = self.distance(other, kernel='gaussian', normalize=normalize)
    # return a Symbol
    if not isinstance(res, Symbol):
        res = Symbol(res)
    return res