import numpy as np

# Support Functions ---------------------------------------
def get_ffmpeg():
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
        
def file_information(filename):
    import re
    import subprocess as sp
    command_info = ['ffprobe', '-show_streams','-i', filename]
    pipe = sp.Popen(command_info, stdout=sp.PIPE,stderr=sp.STDOUT)
    stream = False
    file_info = {}
    currentChannel = None
    for x in pipe.stdout.readlines():
        if re.findall('\\bSTREAM\\b', x.decode()):
            stream = True
        if re.findall('\\b/STREAM\\b', x.decode()):
            stream = False
        if (stream) and ("STREAM" not in x.decode()):
            key, value = x.decode().strip('\n').split('=')
            if (key=="index"):
                file_info[int(value)] = {} #newchannel
                currentChannel = int(value)
            elif (currentChannel!= None):
                file_info[currentChannel][key] = value
    return file_info

def fraction2float(string):
    try:
        return float(string)
    except ValueError:
        num, denom = string.split('/')
        try:
            return (float(num)/float(denom))
        except ValueError:
            print ("That was no valid number.")
            
def split_array(array, numberOfBlocks, overlapping=0): 
    
    array_size = array.size
    step = int(np.ceil(array_size/numberOfBlocks))
    
    if isinstance(overlapping, float):
        overlapping = int(overlapping * step)
    
    split_array = np.zeros((numberOfBlocks, step + overlapping))
    start = np.arange(0,array_size,step)
    stop = np.arange(step+overlapping,array_size,step)
    
    if len(start)!=len(stop):
        aux = np.ones(len(start)-len(stop)) * array_size
        stop = (np.append(stop,aux)).astype(int)
        
    for (i,(s,e)) in enumerate(zip(start, stop)):
        split_array[i,:e-s] = array[s:e]
    return split_array

def split_list(alist, numberOfBlocks, overlapping=0): 
    
    list_size = len(alist)
    step = list_size/numberOfBlocks
    
    if isinstance(overlapping, float):
        overlapping = int(overlapping * step)
    
    start = (np.round(np.arange(0,list_size,step))).astype(int)
    stop = (np.round(np.arange(step+overlapping,list_size,step))).astype(int)
    split_array = []
    
    if len(start)>numberOfBlocks:
        start = start[:numberOfBlocks]
    
    if len(start)!=len(stop):
        aux = np.ones(len(start)-len(stop)) * list_size
        stop = (np.append(stop,aux)).astype(int)
             
    for (s,e) in zip(start, stop):
        split_array.append(alist[s:e])    
    return split_array 

def time2Float(timestr, timeFormat=[60,1]):
    return sum([a*b for a,b in zip(timeFormat, map(float,timestr.split(':')))])

def rgb2gray(rgbImg):
    import numpy as np
    return (np.dot(rgbImg[...,:3], [0.299, 0.587, 0.114])).astype(np.uint8)

def normalize(array):
    return (array - array.min()) /(array.max() - array.min()) 

def zScore(array):
    return (array - array.mean())/array.std()

def generate_dataframe_for_dataset(dataset_path):
    from os import listdir
    from os.path import isfile, join
    import pandas as pd
    
    files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
    df = pd.DataFrame(columns= ['Video_Id', 'Personality', 'Emotion', 'Individual_Id', 'File_path'])
    for i, f in enumerate (files):
        if '.mp4' not in f[-4:]:
            continue
        info = f[:-4] 
        Video_Id, Personality, Emotion, Individual_Id = info.split('_')
        File_path = dataset_path + '/' + f
        df.loc[i] = [Video_Id, Personality, Emotion, Individual_Id, File_path] 
    df = df.sort_values(by=['Video_Id'])  
    df = df.reset_index(drop=True)   
    return df

#-----------------------------------------------------------------------------
# VideoIO --------------------------------------------------------------------
def split_video(filename, start=None, duration=None, mono=True, numberOfBlocks=50, overlapping=0):
    import subprocess as sp
    import numpy as np
    
    ffmpeg = get_ffmpeg()
    videoInfo = file_information(filename)
    
    if duration == None:
        duration = str(videoInfo[0]['duration'])
    else:
        duration = str(duration)
        
    if start == None:
        start = '0'
    else:
        start = str(start)
     
    H, W = int(videoInfo[0]['height']) , int(videoInfo[0]['width'])
    frameRate = fraction2float(videoInfo[0]['avg_frame_rate'])
    
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
        if len(audio_array) % 2 != 0:
            audio_array = audio_array[:-1]
        audio_array = audio_array.reshape((len(audio_array)//int(numberChannels),int(numberChannels)))
    pipeAudio.kill()
    
    audioChunks = splitArray(audio_array[:,0],numberOfBlocks, overlapping) 
    videoChunks = splitList(frames, numberOfBlocks, overlapping)
    
    return videoChunks, audioChunks 
    
def read_video(filename, start=None, duration=None, mono=True):
    import subprocess as sp
    import numpy as np
    
    ffmpeg = get_ffmpeg()
    videoInfo = file_information(filename)
    
    if duration == None:
        duration = str(videoInfo[0]['duration'])
    else:
        duration = str(duration)
        
    if start == None:
        start = '0'
    else:
        start = str(start)
     
    H, W = int(videoInfo[0]['height']) , int(videoInfo[0]['width'])
    frameRate = fraction2float(videoInfo[0]['avg_frame_rate'])
    
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
    
    print('Converting video')
    pipeVideo = sp.Popen(commandVideo, stdout = sp.PIPE, bufsize=10**8)
    print('Converting audio')
    pipeAudio = sp.Popen(commandAudio, stdout = sp.PIPE, bufsize=10**8)
    
    frames=[]
    print('Generating frames')
    for it in range(numberOfFrames):
        print('progress: {} %'.format(int(it*100/(numberOfFrames-1))),end='\r')
        raw_image = pipeVideo.stdout.read(H * W * 3)
        image =  np.fromstring(raw_image, dtype=np.uint8)
        if (image.size != 0):
            image = image.reshape((H,W,3))
            frames.append(image)
    #pipeVideo.kill() 
    
    num = int(int(numberChannels)*int(sampleRate)*float(duration))
    raw_audio = pipeAudio.stdout.read(num*2)
    audio_array = np.fromstring(raw_audio, dtype="int16")
    
    if int(numberChannels) > 1:
        if len(audio_array) % 2 != 0:
            audio_array = audio_array[:-1]
        audio_array = audio_array.reshape((len(audio_array)//int(numberChannels),int(numberChannels)))
    #pipeAudio.kill()
    
    return frames, audio_array 
    

# ---------------------------------------------------------------------
# Face Landmark functions -------------------------------------------------
def localize_face(image, show=False):
    import dlib
    
    detector = dlib.get_frontal_face_detector()

    if image.ndim > 2:
        grayImage = rgb2gray(image[:,:,:3])
        faceLocation = detector(grayImage)
    else:
        faceLocation = detector(image)
    
    if show:
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

def localize_faceLandmarks(image, predictor,plot=False, array=True):
    import dlib
    import numpy as np

    grayImage = rgb2gray(image)
    faceLocation = localize_face(grayImage)
    
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
        faceLandmarks = localize_faceLandmarks(aFrame)
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

def localize_faceFeatures(faceLandmarks):
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
            featureMatrix[i,:] = localize_faceFeatures(meanFaces)
        else:
            featureMatrix[i,:] = featureMatrix[i-1,:]
            print('copy')
        progress += step
        print ("progress: {:.2f} %".format(progress))
    return (featureMatrix, featureMatrix[1:,:]-featureMatrix[:-1,:])

def processFaceDatabase(filename, predictor, sample_start=0, sample_end=None, checkpoint=False, checkpointDir='./'):
    df = pd.read_csv(filename)
    filesDir = '../Database/'
    df['File'] = df.File.str[-16:]
    df['File'] = filesDir + df['File']
    df = df.rename(columns={'Time_duration (mm:ss)': 'Time_duration','Time_start (mm:ss.ms)':'Time_start'})
    flpredictor = predictor
    
    if checkpoint:
        digitis = len(str(df.shape[0]))
    
    if sample_end == None:
        sample_end = df.shape[0]
    
    for idx, row in df.iterrows():
        
        if idx < sample_start:
            continue
        if idx >= sample_end:
            break
        if row['Split'] == 'Yes':
            continue
        
        fileDir = row['File']
        start = row['Time_start']
        duration = row['Time_duration']
        P = row['Personality']
        
        m, devm = derivateFaces(video)

        if P == 'Introvert':
            l = 0
        elif P == 'Balanced':
            l = 1
        elif P == 'Extrovert':
            l = 2
            
        labels = np.append(labels,l)
        featureMatrix = np.append(featureMatrix,m[np.newaxis,:], axis=0)
        devFeatureMatrix = np.append(devFeatureMatrix, devm[np.newaxis,:], axis=0)

        if checkpoint:
            np.save(checkpointDir + '/labels' + str(idx).zfill(digitis), labels)
            np.save(checkpointDir + '/fm' + str(idx).zfill(digitis), featureMatrix)
            np.save(checkpointDir + '/fmd' + str(idx).zfill(digitis), devFeatureMatrix)   

# ------------------------------------------------------------
# Audio Features ----------------------------------------------

#Opensmile Features
def generate_wav_file(input_file, output_file, offset = 0, duration = None, acodec = 'pcm_s16le'):
    import platform
    import subprocess
    from vLib import get_ffmpeg, file_information
    
    FFMPEG_BIN = get_ffmpeg()
    
    if  "wav" not in output_file[-3:]:
        output_file += ".wav"
    
    if duration == None:
        stream = file_information(input_file)
        channel = len(stream) - 1 
        duration = float(stream[channel]['duration_ts']) /float(stream[channel]['sample_rate']) 
        
    command = "{ffmpeg_bin} -y -i {input_file} -ss {offset} -t {duration} -acodec {acodec} {output_file}".format(
                                    ffmpeg_bin = FFMPEG_BIN,
                                    input_file = input_file,
                                    offset = offset,
                                    duration = duration,
                                    acodec = acodec,
                                    output_file = output_file)
    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT) 

def opensmile_features(opensmile_dir, input_file, output_file, overwrite=True):
    import subprocess
    from os import remove, path

    OPENSMILE_EXE = path.join(opensmile_dir, 'SMILExtract')
    OPENSMILE_CONF_PATH = path.join(opensmile_dir, 'config', 'gemaps', 'eGeMAPSv01a.conf')
    
    if not path.isfile(OPENSMILE_EXE):
        raise Exception("Can't find SMILExtract executable")
    
    if not path.isfile(OPENSMILE_CONF_PATH):
        raise Exception("Can't find eGeMAPSv01a.conf configure file")
    
    if overwrite:
        remove(output_file)
    
    try:
        command = "{exe_file} -I {input_file} -C {conf_file} --csvoutput {output_file}".format(
            exe_file = OPENSMILE_EXE,
            input_file = input_file, 
            conf_file = OPENSMILE_CONF_PATH,
            output_file = output_file)
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        raise Exception("Couldn't execute")    

def audio_features(input_file, opensmile_dir, offset = 0, duration = None, output_file = None):
    import pandas as pd
    from os import getcwd, remove
    
    delete_output_file = True
    work_dir = getcwd() + '/'
    
    if output_file == None:
        output_file = work_dir + 'temporary_file.csv'
    else:
        delete_output_file = False
    
    if duration == None:
        from vLib import file_information
        stream = file_information(input_file)
        channel = len(stream) - 1 
        duration = float(stream[channel]['duration_ts']) /float(stream[channel]['sample_rate'])  
    
    if 'csv' not in output_file[-3:]:
        output_file += '.csv'
    
    wav_file = work_dir + 'file.wav'
    generate_wav_file(input_file, wav_file, offset, duration)
    opensmile_features(opensmile_dir, wav_file, output_file)
    remove(wav_file)
    
    ##----------------------------------------------------------
    ## merge metadata and features
    ##----------------------------------------------------------
    features = pd.read_csv(output_file, sep=';', index_col=None)
    features.drop('name', axis=1, inplace=True)
    
    if delete_output_file:
        remove(output_file)
        
    return features

def generate_dataset_audio_features(csv_file, opensmile_dir):
    import pandas as pd
    from os import getcwd, remove, path
    from vLib import generate_wav_file, opensmile_features
    
    work_dir = getcwd() + '/'
    
    df = pd.read_csv(csv_file)
    output_file = work_dir + 'temporary_audio_features.csv'
    
    if path.isfile(output_file):
        remove(output_file) 

    total = df.shape[0]
    for i, row in df.iterrows():
        print("{} %".format(int((i+1)*100/total)), end='\r')
        wav_file = '/home/paula/Desktop/file.wav'
        generate_wav_file(row['File_path'], wav_file)
        opensmile_features(opensmile_dir, wav_file, output_file, overwrite=False)
    
    features = pd.read_csv(output_file, sep=';', index_col=None)
    features.drop('name', axis=1, inplace=True)
    
    remove(output_file) 
    remove('smile.log')
    return features

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



