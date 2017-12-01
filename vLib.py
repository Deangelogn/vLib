import numpy as np

# Support Functions ---------------------------------------
def getffmpeg():
    import platform
    OS =  platform.system()
    
    # linux and MAC OS X
    if (OS == "Linux" or OS =="Darwin"):
        return "ffmpeg"
    # windows
    elif (OS == "Windows"):
        return "ffmpeg.exe"
    else:
        raise Exception("OS not identified") 
        
def getVideoInfo(filename):
    import re
    import subprocess as sp
    command_info = ['ffprobe', '-show_streams','-i', filename]
    pipe = sp.Popen(command_info, stdout=sp.PIPE,stderr=sp.STDOUT)
    stream = False
    videoInfo = {}
    currentChannel = None
    for x in pipe.stdout.readlines():
        if re.findall('\\bSTREAM\\b', x.decode()):
            stream = True
        if re.findall('\\b/STREAM\\b', x.decode()):
            stream = False
        if (stream) and ("STREAM" not in x.decode()):
            key, value = x.decode().strip('\n').split('=')
            if (key=="index"):
                videoInfo[int(value)] = {} #newchannel
                currentChannel = int(value)
            elif (currentChannel!= None):
                videoInfo[currentChannel][key] = value
    return videoInfo

def tofloat(string):
    try:
        return float(string)
    except ValueError:
        num, denom = string.split('/')
        try:
            return (float(num)/float(denom))
        except ValueError:
            print ("That was no valid number.")
            
def splitArray(array, numberOfBlocks, overlapping=0): 
    
    array_size = array.size
    step = int(np.ceil(array_size/numberOfBlocks))
    
    if isinstance(overlapping, float):
        overlapping = int(overlapping * step)
    
    splitArray = np.zeros((numberOfBlocks, step + overlapping))
    start = np.arange(0,array_size,step)
    stop = np.arange(step+overlapping,array_size,step)
    
    if len(start)!=len(stop):
        aux = np.ones(len(start)-len(stop)) * array_size
        stop = (np.append(stop,aux)).astype(int)
        
    for (i,(s,e)) in enumerate(zip(start, stop)):
        splitArray[i,:e-s] = array[s:e]
    return splitArray   
 

def splitList(alist, numberOfBlocks, overlapping=0): 
    
    list_size = len(alist)
    step = list_size/numberOfBlocks
    
    if isinstance(overlapping, float):
        overlapping = int(overlapping * step)
    
    start = (np.round(np.arange(0,list_size,step))).astype(int)
    stop = (np.round(np.arange(step+overlapping,list_size,step))).astype(int)
    splitArray = []
    
    if len(start)>numberOfBlocks:
        start = start[:numberOfBlocks]
    
    if len(start)!=len(stop):
        aux = np.ones(len(start)-len(stop)) * list_size
        stop = (np.append(stop,aux)).astype(int)
             
    for (s,e) in zip(start, stop):
        splitArray.append(alist[s:e])    
    return splitArray 

def timeStringToFloat(timestr, timeFormat=[60,1]):
    return sum([a*b for a,b in zip(timeFormat, map(float,timestr.split(':')))])

def rgb2gray(rgbImg):
    import numpy as np
    return (np.dot(rgbImg[...,:3], [0.299, 0.587, 0.114])).astype(np.uint8)

def normalize(array):
    return (array - array.min()) /(array.max() - array.min()) 

def zScore(array):
    return (array - array.mean())/array.std()

#-----------------------------------------------------------------------------
# VideoIO --------------------------------------------------------------------
def splitVideo(filename, numberOfParts=1, start=None, duration=None, mono=True, numberOfBlocks=50, overlapping=0):
    import subprocess as sp
    import numpy as np
    
    ffmpeg = getffmpeg()
    videoInfo = getVideoInfo(filename)
    
    if duration == None:
        duration = str(videoInfo[0]['duration'])
    else:
        duration = str(duration)
        
    if start == None:
        start = '0'
    else:
        start = str(start)
    
    H, W = int(videoInfo[0]['height']) , int(videoInfo[0]['width'])
    frameRate = tofloat(videoInfo[0]['avg_frame_rate'])
    
    numberOfFrames = int(np.round(frameRate * float(duration)))
    
    sampleRate = videoInfo[1]['sample_rate']
    numberChannels = videoInfo[1]['channels']
    
    commandVideo = [ffmpeg, 
               '-i', filename, 
               '-ss', start,
               '-t', duration,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo','-']
    
    commandAudio = [ffmpeg,
            '-i', filename,
            '-ss', start,   
            '-t', duration,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', sampleRate,
            '-ac', numberChannels, 
            '-']
    
    pipeVideo = sp.Popen(commandVideo, stdout = sp.PIPE, bufsize=10**8)
    pipeAudio = sp.Popen(commandAudio, stdout = sp.PIPE, bufsize=10**8)
    
    frames=[]
    
    for it in range(numberOfFrames):
        raw_image = pipeVideo.stdout.read(H * W * 3)
        image =  np.fromstring(raw_image, dtype=np.uint8)
        if (image.size != 0):
            image = image.reshape((H,W,3))
            frames.append(image)
    pipeVideo.kill() 
    
    num = int(int(numberChannels)*int(sampleRate)*float(duration))
    raw_audio = pipeAudio.stdout.read(num*2)
    audio_array = np.fromstring(raw_audio, dtype="int16")
    
    if int(numberChannels) > 1:
        audio_array = audio_array.reshape((len(audio_array)//int(numberChannels),int(numberChannels)))
    pipeAudio.kill()
    
    audioChunks = splitArray(audio_array[:,0],numberOfBlocks, overlapping) 
    videoChunks = splitList(frames, numberOfBlocks, overlapping)
    return videoChunks, audioChunks 
    
video, audio = splitVideo("../Database/579_0006_01.MP4", start=10,duration=10, numberOfBlocks=50,overlapping=0.4)

# ---------------------------------------------------------------------
# Face Landmark functions -------------------------------------------------
def getFace(image, plot=False):
    import dlib
    
    detector = dlib.get_frontal_face_detector()
    if image.ndim > 2:
        grayImage = rgb2gray(image[:,:,:3])
        faceLocation = detector(grayImage)
    else:
        faceLocation = detector(image)
    
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, ax = plt.subplots(1)
        fig.set_size_inches(24,32)
        plt.imshow(image)
        for region in faceLocation:
            left, top, width, height = region.left(), region.top(), region.width(), region.height()
            pat = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r',facecolor='none')
            ax.add_patch(pat)
        plt.show()
            
    return faceLocation 

def getFaceLandmarks(image, predictor,plot=False, array=True):
    import dlib
    import numpy as np

    grayImage = rgb2gray(image)
    faceLocation = getFace(grayImage)
    
    predictor = dlib.shape_predictor(predictor)

    for region in faceLocation:
        landmarks = predictor(grayImage, region)
    
    if plot:
        fig, ax = plt.subplots(1)
        plt.imshow(image)
        for points in list(landmarks.parts()):
            dots = patches.Circle( (points.x, points.y) , 5)
            ax.add_patch(dots)
        plt.show()    
    
    if array:
        landmarks_array = np.zeros((68,2), int)
        for i, points in enumerate (list(landmarks.parts())):
            landmarks_array[i,:] = np.array([points.x, points.y])
        return landmarks_array
    
    else:
        return landmarks
    
def splitShapes(faceLandmarks):
    
    jaw = faceLandmarks[:17]
    eyebrows = faceLandmarks[17:27]
    nose = faceLandmarks[27:36]
    eyes = faceLandmarks[36:48]
    mouth = faceLandmarks[48:]
    
    return jaw, eyebrows, nose, eyes, mouth

def isLandmarks(landmarks):
        try:
            if landmarks.shape[0] == 68:
                return True
            else:
                return False
        except:
            return False
        
#-----------------------------------------------------------------------
# Display Functions ----------------------------------------------------
def plotFaceLandMarks(image, faceLandmarks, extraDots=np.array([])):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1)
    fig.set_size_inches(24,32)
    plt.imshow(image)
    
    if faceLandmarks.ndim == 1:
        faceLandmarks = faceLandmarks[np.newaxis]
        
    for dot in faceLandmarks:
        dots = patches.Circle( dot , 5)
        ax.add_patch(dots)
    
    if extraDots.size:
        if extraDots.ndim == 1:
            extraDots = extraDots[np.newaxis]
        for dot in extraDots:
            dots = patches.Circle( dot , 10)
            patches.Circle.set_color(dots,'r')
            ax.add_patch(dots)
    plt.show()

def displayFrames(listOfFrames):
    import matplotlib.pyplot as plt
    numberofFrames = len(listOfFrames)
    plt.figure(1, figsize=(12,32))
    for i, frame in enumerate(listOfFrames):
        plt.subplot(numberofFrames//2 + 1,2,i+1)
        plt.imshow(frame)
    plt.show()

def drawLinesOnFace(image, dotArray, dotRef):
    x, y = dotRef[0], dotRef[1] 
    fig, ax = plt.subplots(1)
    fig.set_size_inches(24,32)
    plt.imshow(image)
    
    for dot in dotArray:
        dots = patches.Circle( dot , 5)
        ax.add_patch(dots)
    
    for dot in dotArray:
        dx = dot[0] - x 
        dy = dot[1] - y
        arrow = patches.Arrow(x=x, y=y, dx=dx,dy=dy,width=20)
        patches.Arrow.set_alpha(self=arrow,alpha=0.5)
        ax.add_patch(arrow)
    
    ref = patches.Circle( [x,y], 10)
    patches.Circle.set_color(ref,'r')
    ax.add_patch(ref)
    plt.show() 
    
def plotFace(faceLandmarks,h=None,w=None):
    import matplotlib.pyplot as plt
    #fig.set_size_inches(24,32)
    plt.figure(1,figsize=(8,6))
    plt.plot(faceLandmarks[:,0],faceLandmarks[:,1],'bo')
    xmin, ymin = fl.min(axis=0)
    xmax, ymax = fl.max(axis=0)
    w = [xmin-200,xmax+200]
    h = [ymin-50,ymax+50]
    if h: 
        plt.ylim(h)
    if w:
        plt.xlim(w)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()    
    
# ----------------------------------------------------
# Video descriptor functions   
def getMeanFace(faceFrames, split=False):
    nvls = 0 # Number of Valid Landmarks Set
    meanFace = np.empty((0,2),int)
    for aFrame in faceFrames:
        faceLandmarks = getFaceLandmarks(aFrame)
        if isLandmarks(faceLandmarks):
            nvls += 1
            if (meanFace.shape[0]==0):
                meanFace = np.append(meanFace,faceLandmarks, axis=0)
            else:
                meanFace += faceLandmarks
    meanFace = meanFace // nvls        
    if split:
        return splitShapes(meanFace)
    else:
        return meanFace

def getFaceFeatures(faceLandmarks):
    centerOfGravity = getCentreOfGravity(faceLandmarks)
    magnitude = np.sqrt( ((faceLandmarks - centerOfGravity)**2).sum(axis=1))
    magnitude = normalize(magnitude)
    delta = faceLandmarks - centerOfGravity
    dx, dy = delta[:,0], delta[:,1] 
    orientation = np.arctan2(dy,dx)
    orientation[orientation < 0] += 2 * np.pi
    orientation = normalize(orientation)
    featureVector = np.empty(magnitude.size + orientation.size)
    featureVector[::2] = magnitude
    featureVector[1::2]= orientation
    return featureVector    

def derivateFaces(videoChunks):
    featureMatrix = np.zeros((len(videoChunks), 136)) 
    progress = 0
    step = 100/len(videoChunks)
    for i, chunk in enumerate (videoChunks):
        meanFaces = getMeanFace(chunk)
        if isLandmarks(meanFaces):
            featureMatrix[i,:] = getFaceFeatures(meanFaces)
        else:
            featureMatrix[i,:] = featureMatrix[i-1,:]
            print('copy')
        progress += step
        print ("progress: {:.2f} %".format(progress))
    return (featureMatrix, featureMatrix[1:,:]-featureMatrix[:-1,:])

#---------------------------------------------------
#Audio manipulation ------------------------------
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



