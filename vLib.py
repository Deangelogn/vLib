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

def video_information(filename):
    import re
    import subprocess as sp
    command_info = ['ffprobe','-i', filename, '-hide_banner']
    pipe = sp.Popen(command_info, stdout=sp.PIPE,stderr=sp.STDOUT)
    file_info = {}
    for x in pipe.stdout.readlines():
        output_line = x.decode()
        if 'Duration' in output_line:
            for s in output_line.replace(" ","").strip('\n').split(','):
                idx = s.find(':')
                key, value = s[:idx], s[idx+1:] 
                file_info[key] = value
    
        elif 'Stream #0:0' in output_line:
            idx = output_line.find('Video') + len('Video:')
            output_line = output_line[idx:]
            values = output_line.replace(" ","").strip('\n').split(',')
            file_info['vcodec'] = values[0]
            file_info['pix_fmt'] = values[1]
            file_info['width'], file_info['height'] = map(int,(re.findall(r"\d+x\d+", values[2])[0]).split('x'))
            file_info['frame_rate'] = float(re.findall(r"[-+]?\d*\.\d+|\d+", values[4])[0])
            
        elif 'Stream #0:1' in output_line:
            idx = output_line.find('Audio') + len('Audio:')
            output_line = output_line[idx:]
            values = output_line.replace(" ","").strip('\n').split(',')
            file_info['acodec'] = values[0]
            file_info['sample_rate'] = int(re.findall(r'\d+', values[1])[0])
            file_info['channel']= values[2]
            
    return file_info
        
def file_stream_information(filename):
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

def time2Float(timestr, timeFormat=[3600,60,1]):
    return sum([a*float(b) for a,b in zip(timeFormat, timestr.split(':'))])

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

def centre_of_gravity(face_landmaks):
    return face_landmaks.mean(axis=0).astype(np.int)

#-----------------------------------------------------------------------------
# VideoIO --------------------------------------------------------------------
def split_video(filename, start=None, duration=None, mono=True, numberOfBlocks=50, overlapping=0):
    import subprocess as sp
    import numpy as np
    
    ffmpeg = get_ffmpeg()
    videoInfo = file_stream_information(filename)
    
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
    video_info = video_information(filename)
    
    if duration == None:
        duration = str(video_info['Duration'])
    else:
        duration = str(duration)
     
    if start == None:
        start = '0'
    else:
        start = str(start)
     
    H, W = video_info['height'],video_info['width']
    frame_rate = video_info['frame_rate']
    
    number_of_frames = int(frame_rate * time2Float(duration))
    
    sample_rate = video_info['sample_rate']
    if video_info['channel'] == 'mono':
        number_of_channels = 1
    else:
        number_of_channels = 2
    
    command_video = [ffmpeg, 
               '-i', filename, 
               '-ss', start,
               '-t', duration,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo','-']
    
    command_audio = [ffmpeg,
            '-i', filename,
            '-ss', start,   
            '-t', duration,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', str(number_of_channels), 
            '-']
    
    print('Converting video')
    pipe_video = sp.Popen(command_video, stdout = sp.PIPE, bufsize=10**8)
    
    frames=[]
    print('Generating frames')
    for it in range(number_of_frames):
        print('progress: {} %'.format(int(it*100/(number_of_frames-1))),end='\r')
        raw_image = pipe_video.stdout.read(H * W * 3)
        image =  np.fromstring(raw_image, dtype=np.uint8)
        if (image.size != 0):
            image = image.reshape((H,W,3))
            frames.append(image)

    pipe_video.stdout.close()
    pipe_video.wait()
    
    print('Converting audio')
    pipe_audio = sp.Popen(command_audio, stdout = sp.PIPE, bufsize=10**8)

    num = int(number_of_channels*sample_rate*time2Float(duration))
    raw_audio = pipe_audio.stdout.read(num*2)
    audio_array = np.fromstring(raw_audio, dtype="int16")
    
    pipe_audio.stdout.close()
    pipe_audio.wait()

    if number_of_channels > 1:
        if len(audio_array) % 2 != 0:
            audio_array = audio_array[:-1]
        audio_array = audio_array.reshape((len(audio_array)//number_of_channels,number_of_channels))

    return frames, audio_array

def face_percent(video_frames):
    import dlib
    import cv2
    face_detector = dlib.get_frontal_face_detector()
    total_frames = len(video_frames)
    faced_frames = 0
    for i, frame in enumerate(video_frames):
        print('progress: {} %'.format(int(i*100/(total_frames-1))),end='\r')
        location = face_detector(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY))
        if len(list(location))>0:
            faced_frames += 1
    return faced_frames/total_frames
   

# ---------------------------------------------------------------------
# Face Landmark functions -------------------------------------------------
def face_landmarks(image, predictor,plot=False, array=True):
    import dlib
    import numpy as np
    import cv2

    detector = dlib.get_frontal_face_detector()
    
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faceLocation = detector(grayImage)
    
    if len(list(faceLocation))<1:
        return np.array([])
    
    for region in faceLocation:
        landmarks = predictor(grayImage, region)
    
    if plot:
        import matplotlib.patches as patches
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

def face_max_intensity(face_landmarks):
    import numpy as np
    face_width = face_landmarks[16,0]-face_landmarks[0,0]
    face_height = face_width * 1.375  
    return np.linalg.norm(np.array(face_width, face_height))

def face_part_means(face_landmarks):
    import numpy as np
    face_parts = np.zeros((7,2), int)
    face_parts[0,:] = face_landmarks[17:22].mean(axis=0) # Left eyebrown mean
    face_parts[1,:] = face_landmarks[22:27].mean(axis=0) # Right eyebrown mean
    face_parts[2,:] = face_landmarks[27:31].mean(axis=0) # Top-part nose mean
    face_parts[3,:] = face_landmarks[31:36].mean(axis=0) # Bottom-part nose mean
    face_parts[4,:] = face_landmarks[36:42].mean(axis=0) # Left eye mean
    face_parts[5,:] = face_landmarks[42:48].mean(axis=0) # Right eye mean
    face_parts[6,:] = face_landmarks[48:68].mean(axis=0) # Mouth mean
    return face_parts

def face_parts_features(face_landmarks, reference_point):
    import numpy as np
    magnitude = np.sqrt( ((face_landmarks - reference_point)**2).sum(axis=1))
    magnitude = normalize(magnitude)
    delta = face_landmarks - reference_point
    dx, dy = delta[:,0], delta[:,1] 
    orientation = np.arctan2(dy,dx)
    orientation[orientation < 0] += 2 * np.pi
    orientation /= 2 * np.pi
    feature_vector = np.empty(magnitude.size + orientation.size)
    feature_vector[::2] = magnitude
    feature_vector[1::2]= orientation    
    return feature_vector

def local_means_features(face_landmarks):
    import numpy as np
    local_means = np.zeros((6,2))
    corner_landmarks = np.zeros((21,2), int)
    face_vectors = np.zeros((21,2))
    
    local_means[0,:] = face_landmarks[17:22].mean(axis=0) # Left eyebrown
    corner_landmarks[:3,:] = face_landmarks[17:22:2]
    face_vectors[:3,:] = corner_landmarks[:3,:] - local_means[0,:]
    
    local_means[1,:] = face_landmarks[22:27].mean(axis=0) # Right eyebrown
    corner_landmarks[3:6,:] = face_landmarks[22:27:2]
    face_vectors[3:6,:] = corner_landmarks[3:6,:] - local_means[1,:]
    
    local_means[2,:] = face_landmarks[30]# Nose
    corner_landmarks[6:9,:] = face_landmarks[27:36:4]
    face_vectors[6:9,:] = corner_landmarks[6:9,:] - local_means[2,:]
    
    local_means[3,:] = face_landmarks[36:42].mean(axis=0) # Left eye
    corner_landmarks[9,:] = face_landmarks[36]
    corner_landmarks[10,:] = face_landmarks[37:39].mean(axis=0)
    corner_landmarks[11,:] = face_landmarks[39]
    corner_landmarks[12,:] = face_landmarks[40:42].mean(axis=0)
    face_vectors[9:13,:] = corner_landmarks[9:13,:] - local_means[3,:]
    
    local_means[4,:] = face_landmarks[42:48].mean(axis=0) # Right eye
    corner_landmarks[13,:] = face_landmarks[42]
    corner_landmarks[14,:] = face_landmarks[43:45].mean(axis=0)
    corner_landmarks[15,:] = face_landmarks[45]
    corner_landmarks[16,:] = face_landmarks[46:48].mean(axis=0)
    face_vectors[13:17,:] = corner_landmarks[13:17,:] - local_means[4,:]
    
    local_means[5,:] = face_landmarks[48:68].mean(axis=0) # Mouth
    corner_landmarks[17,:] = face_landmarks[48]
    corner_landmarks[18,:] = face_landmarks[49:54].mean(axis=0)
    corner_landmarks[19,:] = face_landmarks[54]
    corner_landmarks[20,:] = face_landmarks[55:60].mean(axis=0)
    face_vectors[17:21,:] = corner_landmarks[17:21,:] - local_means[5,:]
    
    intensity = np.linalg.norm(face_vectors, axis=1)/face_max_intensity(face_landmarks)
    orientation = np.arctan2(face_vectors[:,1],face_vectors[:,0])
    orientation[orientation<0] += 2*np.pi 
    orientation /= 2*np.pi
    feature_vector = np.empty(intensity.size + orientation.size)
    feature_vector[::2] = intensity
    feature_vector[1::2]= orientation
    
    return feature_vector, local_means, corner_landmarks

def extract_features(image, predictor, detector):
    import cv2
    import dlib
    import matplotlib.pyplot as plt
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face = detector(gray, 1)
    
    if len(list(face))> 0:
        shape = predictor(gray, list(face)[0])

        coords = np.zeros((68, 2), dtype='int')
        main_points = np.zeros((19,2),dtype='int')    

        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        main_points[0,:] = coords[17,:] # left eyebrow left-corner 
        main_points[1,:] = coords[18:21,:].mean(axis=0) # left eyebrow middle
        main_points[2,:] = coords[21,:] # left eyebrow right-corner
        main_points[3,:] = coords[22,:] # right eyebrow left-corner 
        main_points[4,:] = coords[23:26,:].mean(axis=0) # right eyebrow middle
        main_points[5,:] = coords[26,:] # right eyebrow right-corner
        main_points[6,:] = coords[30,:] # nose tip -  Ref point
        main_points[7,:] = coords[36,:] # left eye left-corner
        main_points[8,:] = coords[37:38,:].mean(axis=0) # left eye upper-middle point
        main_points[9,:] = coords[39,:] # left eye right-corner
        main_points[10,:] = coords[40:42,:].mean(axis=0) # left eye bottom-middle point
        main_points[11,:] = coords[42,:] # right eye left-corner
        main_points[12,:] = coords[43:45,:].mean(axis=0) # right eye upper-middle point
        main_points[13,:] = coords[45,:] # right eye right-corner
        main_points[14,:] = coords[46:48,:].mean(axis=0) # right eye bottom-middle point
        main_points[15,:] = coords[48] # mouth left corner
        main_points[16,:] = coords[49:54].mean(axis=0) # upper lips 
        main_points[17,:] = coords[54] # mouth right corner
        main_points[18,:] = coords[55:60].mean(axis=0) # bottom lips

        face_vectors = main_points - main_points[6]
        norm_vectors = np.linalg.norm(face_vectors, axis=1)
        norm_vectors /= norm_vectors.max()
        angle_vectors = np.arctan2(face_vectors[:,1],face_vectors[:,0])
        angle_vectors[angle_vectors<0] += 2*np.pi 
        angle_vectors /= 2*np.pi
        return np.concatenate((norm_vectors,angle_vectors),axis=0)
    else:
        return np.array([])

def video_features(video_frames, predictor):
    import numpy as np
    import sys
    sys.path.insert(0,'./vLib/')
    import vLib
    feature_list = []
    total = len(video_frames)
    for i, frame in enumerate(video_frames):
        print('progress: {} %'.format(int(i*100/(total-1))),end='\r')
        face_landmarks = vLib.face_landmarks(frame, predictor)
        if face_landmarks.size > 0:
            features, _, _ = vLib.local_means_features(face_landmarks)
            feature_list.append(features)
    return feature_list

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

def adjust_columns_variation(df, max_value, min_value, adjust_value = 1):
    idx = (df.max() > max_value) & (df.min() < min_value) 
    selected_columns = df[idx.index[idx]].columns.values
    for column in selected_columns:
        df[column][df[column] > max_value] -= adjust_value
    return df
        
#-----------------------------------------------------------------------
# Display Functions ----------------------------------------------------
def plot_face_parts_features(feature_matrix, frame_rate=None, part_division=[3,3,3,4,4,4], face_parts=None):
    import matplotlib.pyplot as plt
    import numpy as np
    num_samples = feature_matrix.shape[0]
    if frame_rate == None:
        samples_time = np.arange(num_samples)
    else:
        samples_time = np.arange(num_samples)/frame_rate 
    
    if face_parts == None:
        face_parts = ['left eyebrow', 'right eyebrow', 'nose', 'left eye', 'right eyebrow', 'mouth']
    
    intensity = feature_matrix[:,0::2] 
    orientation = feature_matrix[:,1::2]
    start_idx = 0
    for part_idx, final_idx in enumerate(part_division):
        if final_idx == 2:
            color_pack = [(1,0,0),(0,0,1)]
        elif final_idx == 3:
            color_pack = [(1,0,0),(0,1,0),(0,0,1)]
        elif final_idx == 4:
            color_pack = [(1,0,0),(0,1,0),(0,0,1),(0,1,1)]
    
        plt.figure(figsize=(16,12))
        plt.subplot(221)
        plt.title('{} intensity'.format(face_parts[part_idx]))
        plt.xlabel('time (s)')
        plt.ylim([-0.1,1.1])
        for i, it in enumerate(range(start_idx, start_idx+final_idx)):
            plt.plot(samples_time, intensity[:,it], color=color_pack[i], label='L'+str(it))
        
        plt.subplot(222)
        plt.title('{} orientation'.format(face_parts[part_idx]))
        plt.xlabel('time (s)')
        plt.ylim([-0.1,1.1])
        for i, it in enumerate(range(start_idx, start_idx+final_idx)):
            plt.plot(samples_time, orientation[:,it], color=color_pack[i], label='L'+str(it))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        plt.subplot(223)
        plt.title('{} intensity mean'.format(face_parts[part_idx]))
        plt.xlabel('time (s)')
        plt.ylim([-0.1,1.1])
        test = intensity[:,start_idx:start_idx+final_idx] 
        plt.plot(samples_time, intensity[:,start_idx:start_idx+final_idx].mean(1), color='k', label='moviment mean')
        
        plt.subplot(224)
        plt.title('{} orientation mean'.format(face_parts[part_idx]))
        plt.xlabel('time (s)')
        plt.ylim([-0.1,1.1])
        test = orientation[:,start_idx:start_idx+final_idx] 
        plt.plot(samples_time, orientation[:,start_idx:start_idx+final_idx].mean(1), color='k', label='moviment mean')
        
        #plt.savefig('mouth_.png'.format(prefix_file, face_parts[part_idx]))
        start_idx += final_idx
    plt.show()

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

def create_feature_dataframe(feature_list):
    import numpy as np
    import pandas as pd
    feature_matrix = np.asarray(feature_list)
    num_attributes = feature_matrix.shape[1]
    columns = ['L{}_I'.format(i//2) if i%2==0 else 'L{}_O'.format(i//2) for i in range(num_attributes)]
    return pd.DataFrame(columns=columns, data=feature_matrix)


# ------------------------------------------------------------
# Audio Features ----------------------------------------------

#Opensmile Features
def generate_wav_file(input_file, output_file, offset = 0, duration = None, acodec = 'pcm_s16le'):
    import platform
    import subprocess
    from vLib import get_ffmpeg, file_stream_information
    
    FFMPEG_BIN = get_ffmpeg()
    
    if  "wav" not in output_file[-3:]:
        output_file += ".wav"
    
    if duration == None:
        stream = file_stream_information(input_file)
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
        from vLib import file_stream_information
        stream = file_stream_information(input_file)
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



