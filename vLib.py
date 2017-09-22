import numpy as np

# support functions
def getVideoResolution(filename):
    import re
    import subprocess as sp
    command_info = ['ffprobe', '-show_streams',
            '-i', filename]
    pipe = sp.Popen(command_info, stdout=sp.PIPE,stderr=sp.STDOUT)
    #H,W = (0,0)
    for x in pipe.stdout.readlines():
        if (re.findall('\\bwidth\\b', x.decode())):
            W = re.findall(r'\d+', x.decode())[0]
        if (re.findall('\\bheight\\b', x.decode())):
            H = re.findall(r'\d+', x.decode())[0]
    pipe.kill()        
    return (int(H),int(W))   


def getFrameRate(filename):
    import re
    import subprocess as sp
    command_info = ['ffprobe', '-show_streams',
            '-i', filename]
    pipe = sp.Popen(command_info, stdout=sp.PIPE,stderr=sp.STDOUT)

    fr = None
    for x in pipe.stdout.readlines():
        if (re.findall('\\bavg_frame_rate\\b', x.decode())):
            fr = re.findall(r'\d+', x.decode())
            fr = (float(fr[0])/float(fr[1]))
            pipe.kill()
            return fr
    pipe.kill()    
    return (fr)   

def getSampleRate(filename):
    import re
    import subprocess as sp
    command_info = ['ffprobe', '-show_streams',
            '-i', filename]
    pipe = sp.Popen(command_info, stdout=sp.PIPE,stderr=sp.STDOUT)

    for x in pipe.stdout.readlines():
        if (re.findall('\\bsample_rate\\b', x.decode())):
            sr = re.findall(r'\d+', x.decode())[0]
            pipe.kill()
            return int(sr)
    pipe.kill()    
    return (0)   

def getAudioCodecName(filename):
    import re
    import subprocess as sp
    command_info = ['ffprobe', '-show_streams',
            '-i', filename]
    pipe = sp.Popen(command_info, stdout=sp.PIPE,stderr=sp.STDOUT)
    index = 0
    for x in pipe.stdout.readlines():
        if (re.findall('\\bindex=1\\b', x.decode())):
            index=1
        if (index == 1):    
            if (re.findall('\\bcodec_name\\b', x.decode())):
                sr = re.findall(r'=(\w+)', x.decode())[0]
                pipe.kill()
                return sr
    pipe.kill()    
    return (0)  

def timeStringToFloat(timestr, timeFormat=[3600,60,1]):
    return sum([a*b for a,b in zip(timeFormat, map(float,timestr.split(':')))])
      
#Reading functions
def readFrames(filename, duration, start="00:00:00", framedrop=1, fd=None):
    import platform
    import subprocess as sp
    import numpy as np
    
    OS =  platform.system()
    if (OS == "Linux"):
        FFMPEG_BIN = "ffmpeg"
    elif (OS == "Windows"):
        FFMPEG_BIN = "ffmpeg.exe"
    else:
        raise Exception("OS not identified")   
    
    H,W = getVideoResolution(filename)    
    fr = getFrameRate(filename)
    nframes = int(fr * timeStringToFloat(duration,timeFormat=[60,1]))
    if fd == None:
        fd = [H,W]
    
    command = [ FFMPEG_BIN,
            '-ss', start, 
            '-i', filename,
            '-t', duration,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
    
    frameList=[]
    for f in range(nframes):
        raw_image = pipe.stdout.read(H * W * 3)
        image =  np.fromstring(raw_image, dtype=np.uint8)
        if (image.size != 0):
            if (f%framedrop==0):
                image = image.reshape((H,W,3))
                image = image[H//2-fd[0]//2:H//2+fd[0]//2, W//2-fd[1]//2:W//2+fd[1]//2,:]
                frameList.append(image)
    pipe.kill()            
    return frameList

def readAudio(filename, duration, start="00:00:00", mono=False, normRange=None):
    import numpy as np
    import platform
    import subprocess as sp
    
    OS =  platform.system()
    if (OS == "Linux"):
        FFMPEG_BIN = "ffmpeg"
    elif (OS == "Windows"):
        FFMPEG_BIN = "ffmpeg.exe"
    else:
        raise Exception("OS not identified")
    
    audioCodec = getAudioCodecName(filename)
    
    if mono:
        numberChannels = 1
    else:
        numberChannels = 2
        
    sr = getSampleRate(filename)
    
    command = [ FFMPEG_BIN,
            '-ss', start,   
            '-i', filename,
            '-t', duration,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(sr),
            '-ac', str(numberChannels), 
            '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    raw_audio = pipe.stdout.read(int(numberChannels*sr*timeStringToFloat(duration)))
    audio_array = np.fromstring(raw_audio, dtype="int16")
    if mono==False:
        audio_array = audio_array.reshape((len(audio_array)//2,2))
    pipe.kill()
    if normRange != None:
        audio_array = (audio_array).astype(np.float32)        
        return (normRange[1]-normRange[0])*(audio_array - audio_array.min())/(audio_array.max()-audio_array.min())+normRange[0]
    else:    
        return audio_array 

#Audio manipulation
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

#Audio plot functions
def plotAudio(audio,sr):
    import matplotlib.pyplot as plt
    t = np.arange(0,len(audio),1)/sr
    plt.figure()
    plt.plot(t,audio)
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



