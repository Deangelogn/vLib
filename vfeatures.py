# Audio features
def zero_crossing(audio):
    import numpy as np
    return len(np.where(np.diff(np.sign(audio)))[0])

def voicedTime(signal, t, hold = 0):
    totalSamples = signal.size
    segments = findVoicedSemants(abs(signal),t,hold)
    numVoicedSamples = 0
    print(len(segments))
    for seg in segments:
        numVoicedSamples += seg[1]-seg[0]
    
    return (numVoicedSamples/totalSamples , abs((numVoicedSamples)/totalSamples-1))

def totalEnergy(signal):
    return (signal**2).sum()

def potency(signal, sr):
    t = len(signal)/sr
    return (signal**2).sum()/t
