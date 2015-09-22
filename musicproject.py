import numpy as np
import cv2
import pyglet as pyglet



def converter(stri):
    """
    Deal with the strange encoding in the CSV file provide by the blackbox. Take one character each 3 characters to remove some weird parts
    STRI : string
    return a string
    """
    result=''
    for i in np.arange(1,len(stri),2):
        result+=stri[i]
    return result

def reader(filename):
    """
    Read the last line of the CSV file provide by the blackbox and return if there is a mouse in the musicbox.
    STRI : csv filename
    return '' or mouse, '' or event
    """
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
    mouse = ''
    if (event == 'In') or (event == 'Out'):
        mouse = converter(linesplit[2])
    return mouse, event

def pretreat(frame):
    """
    Preprocess the image obtained by the webcam
    STRI : frame of a VideoCapture object
    return an image
    """
    grayim = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayim=grayim[:,50:cap.get(3)-50]
    grayim=cv2.GaussianBlur(grayim,(5,5), 5)
    ret,binary = cv2.threshold(grayim,25,255,cv2.THRESH_BINARY_INV)
    binary=cv2.GaussianBlur(binary,(5,5), 2)
    return binary


player1 = pyglet.media.Player()
player1.eos_action = player1.EOS_LOOP
player1.queue(pyglet.media.load('Files/beep-02s.wav'))

player2 = pyglet.media.Player()
player2.eos_action = player2.EOS_LOOP
player2.queue(pyglet.media.load('Files/beep-04s.wav'))

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Files/Film1.avi")
#bef=np.zeros((cap.get(4),cap.get(3)))

while(cap.isOpened()):
    ret, frame = cap.read()
    binary = pretreat(frame)

    # mousein,mouseout,event = reader('Sorter-15.04.10.csv')

    # Calcul the centroids si il y a une tache
    if np.sum(binary)>2:
        M = cv2.moments(binary)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])

        # Draw the rectangle to know if it is left or right
        x, y = int(binary.shape[1]), int(binary.shape[0])
        if centroid_y < int(binary.shape[0]/2):
            binary=cv2.rectangle(binary, (0,0), (10,10), (255,255,255), -1)
            player2.pause()
            player1.play()
        else:
            binary=cv2.rectangle(binary, (0,y), (10,y-10), (255,255,255), -1)
            player1.pause()
            player2.play()

    # Display the image
    cv2.imshow('frame',binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
