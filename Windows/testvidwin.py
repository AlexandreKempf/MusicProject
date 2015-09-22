import numpy as np
import cv2
import winsound


def converter(stri):
    result=''
    for i in np.arange(1,len(stri),2):
        result+=stri[i]
    return result


def reader(filename):
    csvfile = open(filename, 'r')
    spamreader = csvfile.readlines()
    charmin=5
    if len(spamreader[-1]) < charmin :
        if len(spamreader[-2]) < charmin :
            line = spamreader[-3]
        else:
            line = spamreader[-2]
    else:
        line = spamreader[-1]

    linesplit = line.split(';')

    event = converter(linesplit[-4])

    mousein = ''
    mouseout = ''
    if event == 'In':
        mousein = converter(linesplit[2])
    elif event == 'Out':
        mouseout = converter(linesplit[2])

    return mousein,mouseout,event

def pretreat(frame):
    grayim = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    # Select the box
    grayim=grayim[:,50:cap.get(3)-50]
#    grayim = cv2.adaptiveThreshold(grayim, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY, 35, 30, dst=None)
#    # Smooth and threshold
    grayim=cv2.GaussianBlur(grayim,(5,5), 5)
    ret,binary = cv2.threshold(grayim,25,255,cv2.THRESH_BINARY_INV)
#    # Smooth to remove individuals pixels
    binary=cv2.GaussianBlur(binary,(5,5), 2)
    return binary
    
    

import time, wave, pymedia.audio.sound as sound

# little to do on the proper audio setup

f0= wave.open( 'chords2f0.wav', 'rb' )
sampleRate0= f0.getframerate() # reads framerate from the file
channels0= f0.getnchannels()
format0= sound.AFMT_S16_LE  # this sets the audio format to most common WAV with 16-bit codec PCM Linear Little Endian, use either pymedia or any external utils such as FFMPEG to check / corvert audio into a proper format.
audioBuffer0 = 300000 




def soundplay(file):
    f= wave.open( file, 'rb' )
    sampleRate= f.getframerate() # reads framerate from the file
    channels= f.getnchannels()
    format= sound.AFMT_S16_LE  # this sets the audio format to most common WAV with 16-bit codec PCM Linear Little Endian, use either pymedia or any external utils such as FFMPEG to check / corvert audio into a proper format.
    audioBuffer = 300000 
    return f, sampleRate, channels, format, audioBuffer


cap = cv2.VideoCapture(0)

#bef=np.zeros((cap.get(4),cap.get(3)))
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    binary = pretreat(frame)

    mousein,mouseout,event = reader('Sorter-15.04.10.csv')

#    # Calcul the centroids si il y a une tache
    if np.sum(binary)>2:
        M = cv2.moments(binary)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        # binary=cv2.rectangle(binary, (centroid_x,centroid_y), (centroid_x+2,centroid_y+2), (0,0,0), -1)

        # Draw the rectangle to know if it is left or right
        if centroid_y < int(binary.shape[0]/2):
            x,y=int(binary.shape[1]),int(binary.shape[0])
            binary=cv2.rectangle(binary, (0,0), (10,10), (255,255,255), -1)
#            winsound.PlaySound(None,winsound.SND_ASYNC)
#            winsound.PlaySound('chords2f0.wav',winsound.SND_ASYNC)
            if not snd.isPlaying():
                f, sampleRate, channels, format, audioBuffer = soundplay('chords2f0.wav')
                snd= sound.Output( sampleRate, channels, format )
                s= f.readframes( audioBuffer )
                snd.play( s )
        else:
            binary=cv2.rectangle(binary, (0,y), (10,y-10), (255,255,255), -1)
#            winsound.PlaySound(None,winsound.SND_ASYNC)
#            winsound.PlaySound('chords2fDmax.wav',winsound.SND_ASYNC)
            if not snd.isPlaying():
                f, sampleRate, channels, format, audioBuffer = soundplay('chords2fDmax.wav')
                snd= sound.Output( sampleRate, channels, format )
                s= f.readframes( audioBuffer )
                snd.play( s )
            
    # Display the image
    cv2.imshow('frame',binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
