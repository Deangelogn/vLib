def getVideoResolution(filename):
    import re
    import subprocess as sp
    command_info = ['ffprobe', '-show_streams',
            '-i', filename]
    pipe = sp.Popen(command_info, stdout=sp.PIPE,stderr=sp.STDOUT)
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

# Read videos
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

# Read audio
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
     
def saveAudio(audio_array, output_filename):
    import numpy as np
    import platform
    import subprocess as sp
    #print(filename)
    command = ['ffmpeg',
           '-y', # (optional) means overwrite the output file if it already exists.
           '-f', 's16le', # means 16bit input
           '-acodec', 'pcm_s16le', # means raw 16bit input
           '-ar', '48000', # the input will have 48000 Hz
           '-ac','1', # the input will have 1 channels (stereo)
           '-i', '-', # means that the input will arrive from the pipe
           '-vn', # means "don't expect any video input"
           '-acodec', 'aac', '-strict', '-2', # output audio codec
           '-b:a', '1536k', # output bitrate (=quality). Here, 3000kb/second  
           output_filename]

    pipe = sp.Popen(command, stdin=sp.PIPE,stdout=sp.PIPE, stderr=sp.STDOUT)
    
    pipe.communicate(input=audio_array.tobytes())
    pipe.kill()
