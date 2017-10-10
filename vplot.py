def plotAudio(audio,sr=None,title=None,Yrange=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if sr == None:
        t = np.arange(0,len(audio),1)
    else:
        t = np.arange(0,len(audio),1)/sr
    plt.figure()
    plt.plot(t,audio)
    if title:
        plt.title(title)
    if Yrange:
        plt.ylim(Yrange)
    
    plt.show()
    
def plotAudioFrames(audio_frames, sr):
    import matplotlib.pyplot as plt
    prev = 0
    for i,a in enumerate(audio_frames):
        t = (np.arange(0,audio_frames[i].size,1) + prev)/sr 
        fig, (ax1) = plt.subplots(1,1)
        ax1.plot(t,a)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        plt.show()
        
def plotVideoAndAudioFrames(audio_frames,video_frames, sr):
    import matplotlib.pyplot as plt
    import numpy as np
    prev = 0
    for i ,(a, v) in enumerate (list(zip(audio_frames,video_frames))):
        t = (np.arange(0,audio_frames[i].size,1) + prev)/sr 
        np.set_printoptions(precision=6)
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
        ax1.imshow(v,aspect='auto')
        ax2.plot(t,a)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        fig.tight_layout()
        plt.show()
        prev = prev + audio_frames[i].size
