from contextlib import closing
from PIL import Image
import subprocess
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
import argparse
from pytube import YouTube, __main__
import multiprocessing as mp
import time



def downloadFile(url):
    name = YouTube(url).streams.first().download()
    newname = name.replace(' ','_')
    os.rename(name,newname)
    return newname

def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv,-minv)

'''
    TODO:
    imeplement option for upscaling images, or applying other filters
'''
def copyFrame(path, inputFrame, outputFrame):
    src = path+"/frame{:06d}".format(inputFrame+1)+".jpg"
    dst = path+"/newFrame{:06d}".format(outputFrame+1)+".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if outputFrame%20 == 19:
        print(str(outputFrame+1)+" time-altered frames saved.")
    return True

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex]+"_ALTERED"+filename[dotIndex:]

def createPath(s):
    #assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."

    try:  
        os.mkdir(s)
    except OSError:  
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

def deletePath(s): # Dangerous! Watch out!
    try:  
        rmtree(s,ignore_errors=False)
    except OSError:  
        print ("Deletion of the directory %s failed" % s)
        print(OSError)


def process_video(args):
    TEMP_FOLDER,frameRate,SAMPLE_RATE,NEW_SPEED,SILENT_THRESHOLD,FRAME_SPREADAGE,AUDIO_FADE_ENVELOPE_SIZE = args    


    sampleRate, audioData = wavfile.read(TEMP_FOLDER+"/audio.wav")
    audioSampleCount = audioData.shape[0]
    maxAudioVolume = getMaxVolume(audioData)


    samplesPerFrame = sampleRate/frameRate

    audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))

    hasLoudAudio = np.zeros((audioFrameCount))

    for i in range(audioFrameCount):
        start = int(i*samplesPerFrame)
        end = min(int((i+1)*samplesPerFrame),audioSampleCount)
        audiochunks = audioData[start:end]
        maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume
        if maxchunksVolume >= SILENT_THRESHOLD:
            hasLoudAudio[i] = 1

    chunks = [[0,0,0]]
    shouldIncludeFrame = np.zeros((audioFrameCount))
    for i in range(audioFrameCount):
        start = int(max(0,i-FRAME_SPREADAGE))
        end = int(min(audioFrameCount,i+1+FRAME_SPREADAGE))
        shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
        if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]): # Did we flip?
            chunks.append([chunks[-1][1],i,shouldIncludeFrame[i-1]])

    chunks.append([chunks[-1][1],audioFrameCount,shouldIncludeFrame[i-1]])
    chunks = chunks[1:]

    outputAudioData = np.zeros((0,audioData.shape[1]))
    outputPointer = 0

    lastExistingFrame = None
    for chunk in chunks:
        audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)]
        
        sFile = TEMP_FOLDER+"/tempStart.wav"
        eFile = TEMP_FOLDER+"/tempEnd.wav"
        wavfile.write(sFile,SAMPLE_RATE,audioChunk)
        with WavReader(sFile) as reader:
            with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                tsm.run(reader, writer)
        _, alteredAudioData = wavfile.read(eFile)
        leng = alteredAudioData.shape[0]
        endPointer = outputPointer+leng
        outputAudioData = np.concatenate((outputAudioData,alteredAudioData/maxAudioVolume))

        #outputAudioData[outputPointer:endPointer] = alteredAudioData/maxAudioVolume

        # smooth out transitiion's audio by quickly fading in/out
        
        if leng < AUDIO_FADE_ENVELOPE_SIZE:
            outputAudioData[outputPointer:endPointer] = 0 # audio is less than 0.01 sec, let's just remove it.
        else:
            premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE
            mask = np.repeat(premask[:, np.newaxis],2,axis=1) # make the fade-envelope mask stereo
            outputAudioData[outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask
            outputAudioData[endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask

        startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
        endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
        for outputFrame in range(startOutputFrame, endOutputFrame):
            inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))
            didItWork = copyFrame(TEMP_FOLDER,inputFrame,outputFrame)
            if didItWork:
                lastExistingFrame = inputFrame
            else:
                copyFrame(TEMP_FOLDER,lastExistingFrame,outputFrame)

        outputPointer = endPointer

    wavfile.write(TEMP_FOLDER+"/audioNew.wav",SAMPLE_RATE,outputAudioData)

    '''
    outputFrame = math.ceil(outputPointer/samplesPerFrame)
    for endGap in range(outputFrame,audioFrameCount):
        copyFrame(int(audioSampleCount/samplesPerFrame)-1,endGap)
    '''
def recombine_audio(args):
    video,audio,dest = args
    command = f"ffmpeg -v error -i {video} -i {audio} -strict -2 -c:v copy -c:a aac -hide_banner \'{dest}\'"
    print(command)
    subprocess.call(command, shell=True)

def main():

    parser = argparse.ArgumentParser(description='Modifies a video file to play at different speeds when there is sound vs. silence.')
    parser.add_argument('--input_file', type=str,  help='the video file you want modified')
    parser.add_argument('--url', type=str, help='A youtube url to download and process')
    parser.add_argument('--output_file', type=str, default="", help="the output file. (optional. if not included, it'll just modify the input file name)")
    parser.add_argument('--silent_threshold', type=float, default=0.03, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")
    parser.add_argument('--sounded_speed', type=float, default=1.00, help="the speed that sounded (spoken) frames should be played at. Typically 1.")
    parser.add_argument('--silent_speed', type=float, default=5.00, help="the speed that silent frames should be played at. 999999 for jumpcutting.")
    parser.add_argument('--frame_margin', type=float, default=1, help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")
    parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
    parser.add_argument('--frame_rate', type=float, default=-1, help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
    parser.add_argument('--frame_quality', type=int, default=3, help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.")
    parser.add_argument('--frame_quality_crf',type=int,default=24,help="set crf for final encode, valid range: 0-63, 0 is highest, 51 is lowest, tpyical is 17-28, default is 24")

    args = parser.parse_args()



    frameRate = args.frame_rate
    SAMPLE_RATE = args.sample_rate
    SILENT_THRESHOLD = args.silent_threshold
    FRAME_SPREADAGE = args.frame_margin
    NEW_SPEED = [args.silent_speed, args.sounded_speed]
    if args.url != None:
        INPUT_FILE = downloadFile(args.url)
    else:
        INPUT_FILE = args.input_file
    URL = args.url

    FRAME_QUALITY = args.frame_quality
    FRAME_QUALITY_CRF = args.frame_quality_crf


    assert INPUT_FILE != None , "why u put no input file, that dum"
        
    if len(args.output_file) >= 1:
        OUTPUT_FILE = args.output_file
    else:
        OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

    TEMP_FOLDER = "TEMP"

    AUDIO_FADE_ENVELOPE_SIZE = 400 # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)
        
    createPath(TEMP_FOLDER)

    '''
        TODO:
        put some kind of error handling for these checks
    '''
    if frameRate < 0:
        args = rf"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate '{INPUT_FILE}'"
        print(args)
        rate = subprocess.run([args],shell=True,capture_output=True).stdout.decode()
        numerator,denominator = [int(num) for num in rate.split('/')]
        frameRate = numerator / denominator
    
    # get length of input video in seconds
    command = rf"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '{INPUT_FILE}'"
    print(command)
    output = subprocess.run([command],shell=True,capture_output=True).stdout.decode()
    nsec = float(output)
    '''
        TODO:
        arg option to configure number of threads, default is one slice per thread
    '''
    num_threads = mp.cpu_count()
    if nsec < 60:
        num_threads = 1
    
    tstart = 0
    tfinish = 0
    if num_threads > 1:

        seg_length = nsec / num_threads
        print("seg length",seg_length)

        vid_ext = INPUT_FILE.split('.')[-1]
        #split into one slice for each cpu
        command = f"ffmpeg -v error -v error -i \'{INPUT_FILE}\' -c copy -map 0 -segment_time {seg_length} -f segment -reset_timestamps 1 {os.path.join(TEMP_FOLDER,f'temp_slice%03d.{vid_ext}')}" # 3 digits should be enough, if you have more than 1000 threads then you're a serious baller
        print(command)
        subprocess.call(command,shell=True)
        # slice paths
        slice_videos = [file.path for file in os.scandir(TEMP_FOLDER)]
        slice_videos = [path for path in sorted(slice_videos)]
        temp_folders = [""] * num_threads
        print(slice_videos)
        
        '''
            TODO:
            add verbosity flag for print debugging statements,
            as well as verbosity of ffmpeg -v error
        '''
        print("splitting slices into frames")
        for i,vidPath in enumerate(slice_videos):
            slice_temp_folder = os.path.join(TEMP_FOLDER,f"slice_{i}")
            temp_folders[i] = slice_temp_folder
            createPath(slice_temp_folder)

            # convert input video into jpg sequence
            command = f"ffmpeg -v error -i \'{vidPath}\' -qscale:v {FRAME_QUALITY} -pix_fmt yuvj420p -hide_banner {os.path.join(slice_temp_folder,'frame%06d.jpg')}"
            print(command)
            subprocess.call(command, shell=True)
            # extract audio from input video into .wav file
            command = f"ffmpeg -v error -hide_banner -i \'{vidPath}\' -ab 160k -ac 2 -ar {SAMPLE_RATE} -vn {os.path.join(slice_temp_folder,'audio.wav')}"
            subprocess.call(command, shell=True)
        
        args = [[slice_temp_folder,frameRate,SAMPLE_RATE,NEW_SPEED,SILENT_THRESHOLD,FRAME_SPREADAGE,AUDIO_FADE_ENVELOPE_SIZE] for slice_temp_folder in temp_folders]
        print("stating worker threads")
        tstart = time.time()
        mp.Pool(num_threads).map(process_video,args)
        tfinish = time.time()

        print("recombining frames into slices")
        slices = [""] * num_threads
        for i,slice_folder in enumerate(temp_folders):
            slice_output_path = os.path.join(TEMP_FOLDER,f'newVideo_{i}.mp4')
            command =f"ffmpeg -v error -f image2 -framerate {frameRate} -i {os.path.join(slice_folder,'newFrame%06d.jpg')} -c:v libx264 -crf {FRAME_QUALITY_CRF} -preset fast -pix_fmt yuvj420p -hide_banner \'{os.path.join(slice_folder,'newVideo.mp4')}\'"
            print(command)
            subprocess.call(command, shell=True)

            slices[i] = slice_output_path

        audio_recombine_args = [[os.path.join(slice_folder,'newVideo.mp4'),os.path.join(slice_folder,'audioNew.wav'),slice_output_path] for slice_folder,slice_output_path in zip(temp_folders,slices)]
        mp.Pool(num_threads).map(recombine_audio,audio_recombine_args)

        for folder in temp_folders:
            deletePath(folder)

        print("recombinging slices")
        
        with open(os.path.join(TEMP_FOLDER,"slices.txt"),"w") as fh:
            fh.write('\n'.join([f"file \'{os.path.basename(vidslice)}\'" for vidslice in slices]))
        
        command = f"ffmpeg -v error -safe 0 -f concat -i {os.path.join(TEMP_FOLDER,'slices.txt')} -c copy \'{OUTPUT_FILE}\'"
        print(command)
        subprocess.call(command, shell=True)


    else:
        # convert input video into jpg sequence
        command = f"ffmpeg -v error -i \'{INPUT_FILE}\' -qscale:v {FRAME_QUALITY} {os.path.join(TEMP_FOLDER,'frame%06d.jpg')} -pix_fmt yuvj420p -hide_banner"
        print(command)
        subprocess.call(command, shell=True)
        print(command)
        # extract audio from input video into .wav file
        command = f"ffmpeg -v error -i \'{INPUT_FILE}\' -ab 160k -ac 2 -ar {SAMPLE_RATE} -vn {os.path.join(TEMP_FOLDER,'audio.wav')} -hide_banner"
        subprocess.call(command, shell=True)
        
        tstart = time.time()
        process_video([TEMP_FOLDER,frameRate,SAMPLE_RATE,NEW_SPEED,SILENT_THRESHOLD,FRAME_SPREADAGE,AUDIO_FADE_ENVELOPE_SIZE])
        tfinish = time.time()

        command =f"ffmpeg -v error -f image2 -framerate {frameRate} -i {os.path.join(TEMP_FOLDER,'newFrame%06d.jpg')} -c:v libx264 -crf {FRAME_QUALITY_CRF} -preset fast -pix_fmt yuvj420p -hide_banner \'{os.path.join(TEMP_FOLDER,'newVideo.mp4')}\'"
        print(command)
        subprocess.call(command, shell=True)
        recombine_audio([os.path.join(TEMP_FOLDER,'newVideo.mp4'),os.path.join(TEMP_FOLDER,'audioNew.wav'),OUTPUT_FILE])

    deletePath(TEMP_FOLDER)


    print(f"main processing took {tfinish - tstart:0.2f}s")

if __name__ == "__main__":
    main()
