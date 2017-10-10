#Audio manipulation
import numpy as np
    
def splitSignal(audio,chuckSize,overlapping=0):
    ss = []
    for i in range(0,len(audio),chuckSize-overlapping):
        ss.append(audio[i:i+chuckSize])
    return ss

def createAudioFrames(audio, sr, fr):
    return splitSignal(audio, int(sr//fr))

def displayAudioFrames(audio_frames, sr):
    for af in audio_frames:
        IPython.display.display(IPython.display.Audio(data=af, rate=sr) )
        
def groupedFrames(frames, csize):
    return list(zip(*[iter(video_frames_frames)]*csize))
    
def mergeAudioFrames(audio_frames, n):
    af = []
    for i in range(0, len(audio_frames), n):
        af.append(np.concatenate(audio_frames[i:i+n]))
    return af 

def mergeSignal(audio_list,chuckSize=0,overlapping=0):
    y = np.hstack(audio_list)
    remove=[]
    for i in range(overlapping):
        remove = remove + list(np.arange(chuckSize+i,y.size,chuckSize))
    return np.delete(y,remove)

def removeDC(audio):
    return audio-audio.mean()

def normalize(array, normRange=[0,1]):
    return ( (normRange[1]-normRange[0]) *
            (array - array.min())/(array.max()-array.min())+normRange[0] )

def energyEnvelop(signal, window):
    signal = splitSignal(signal,window)
    output = np.zeros(len(signal))
    for i, subsingal in enumerate(signal):
        output[i]=  np.sqrt(np.mean(subsingal**2)) 
    return output

def STFT (signal, frameSize):
    output = np.zeros(len(signal),np.complex) 
    for i in range (0,len(signal),frameSize):
        output[i:i+frameSize] = np.fft.fft(signal[i:i+frameSize])
    return output

def MelFrequency(f):
    fc = 1000
    if f < 1000:
        return M
    else:
        return fc * (1 + np.log(f/fc) ) 

def energy(signal):
    return signal**2

def findVoicedSemants(signal, t, hold = 0):
    tVal = signal.max() * t
    signVariationArray = np.diff(np.sign(signal - tVal))
    (signVariationIdx,) = np.nonzero(signVariationArray)
    signVariationIdx = signVariationIdx.astype(float)
    
    signVariationIdx = np.insert(signVariationIdx,-1,float('inf'))
    signVariationIdx[-2:] = signVariationIdx[-1],signVariationIdx[-2]
    
    shiftIdx = np.zeros_like(signVariationIdx)
    shiftIdx[0] = float('-inf')
    shiftIdx[1:] = signVariationIdx[:-1]
    
    endSegments = shiftIdx[(abs(shiftIdx  - signVariationIdx)>hold)][1:]
    startSegments = signVariationIdx[(abs(shiftIdx  - signVariationIdx)>hold)][:-1]
    
    return np.vstack((startSegments, endSegments))

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
