#import GPIO
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.cleanup()
import time
#import firebase
from firebase import firebase
import json
##setup firebase
firebase = firebase.FirebaseApplication('Firebase Database link', None)
##initialize variable
interupt=0
interuptOld=0
time1=0
time2=0
time3=0
direction1=0
direction2=0
direction3=0
run = True        

#stop script pin
GPIO.setup(40, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#setup input pins
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) #phase1
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) #phase2
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) #phase3

#interupt Pins
GPIO.setup(19,GPIO.OUT) #pase 1
GPIO.setup(21,GPIO.OUT) #pase 2
GPIO.setup(23,GPIO.OUT) #pase 3
#time Pins
GPIO.setup(29,GPIO.OUT) #1st bit
GPIO.setup(31,GPIO.OUT) #2nd bit
GPIO.setup(33,GPIO.OUT) #3rd bit
#direction Pin
GPIO.setup(35,GPIO.OUT)

   
def get_firebase():
    global interuptOld
    global interupt
    global time1
    global time2
    global time3
    global direction1
    global direction2
    global direction3
    try:
        interupt=firebase.get('output','interupt')
    except OSError:
        print("Network error")
    
    #Set pin low
    GPIO.output(19,GPIO.HIGH)
    GPIO.output(21,GPIO.HIGH)
    GPIO.output(23,GPIO.HIGH)

    if(interupt!=interuptOld):
        try:
            interuptOld = interupt
            time1 = firebase.get('output','time1')
            time2 = firebase.get('output','time2')
            time3 = firebase.get('output','time3')
            direction1 = firebase.get('output','direction1')
            direction2 = firebase.get('output','direction2')
            direction3 = firebase.get('output','direction3')
        
            set_output(1,time1,direction1)
            time.sleep(1)
            set_output(2,time2,direction2)
            time.sleep(1)
            set_output(3,time3,direction3)
            time.sleep(1)
            print("changed")
        except OSError:
            print("print error");
    else:
        print("not changed")
        
def update_firebase(phase):
    firebase.put('input','camera',phase)

def set_output(phase, time, direction):
    
        
    binaryTime =format(time,"03b")
    binaryStr = str(binaryTime)
    
    #set time output
    if(binaryStr[0]=="0"):
        GPIO.output(29,GPIO.HIGH)
    else:
        GPIO.output(29,GPIO.LOW)
        
    if(binaryStr[1]=="0"):
        GPIO.output(31,GPIO.HIGH)
    else:
        GPIO.output(31,GPIO.LOW)
        
    if(binaryStr[2]=="0"):
        GPIO.output(33,GPIO.HIGH)
    else:
        GPIO.output(33,GPIO.LOW)
        
    #set direction output
    if(direction==0):
        GPIO.output(35,GPIO.HIGH)
    else:
        print(direction)
        GPIO.output(35,GPIO.LOW)
        
    #set interupt for relavent phase
    if(phase == 1):
        print("phase1")
        GPIO.output(19,GPIO.LOW)
        GPIO.output(21,GPIO.HIGH)
        GPIO.output(23,GPIO.HIGH)
    elif(phase == 2):
        print("phase1")
        GPIO.output(19,GPIO.HIGH)
        GPIO.output(21,GPIO.LOW)
        GPIO.output(23,GPIO.HIGH)
    elif(phase == 3):
        print("phase1")
        GPIO.output(19,GPIO.HIGH)
        GPIO.output(21,GPIO.HIGH)
        GPIO.output(23,GPIO.LOW)

def button_callback1(channel):
    print("Button1 was pushed!")
    update_firebase(1)
def button_callback2(channel):
    print("Button2 was pushed!")
    update_firebase(2)
def button_callback3(channel):
    print("Button3 was pushed!")
    update_firebase(3)

def button_callbackStop(channel):
    #global run
    print("Stop was pushed!")
     #Set pin low
    GPIO.output(19,GPIO.HIGH)
    GPIO.output(21,GPIO.HIGH)
    GPIO.output(23,GPIO.HIGH)
    GPIO.output(29,GPIO.HIGH)
    GPIO.output(31,GPIO.HIGH)
    GPIO.output(33,GPIO.HIGH)
    GPIO.output(35,GPIO.HIGH)
    update_firebase(0)
    run=False
    
#get input
GPIO.add_event_detect(11,GPIO.RISING,callback=button_callback1)
GPIO.add_event_detect(13,GPIO.RISING,callback=button_callback2)
GPIO.add_event_detect(15,GPIO.RISING,callback=button_callback3)
GPIO.add_event_detect(40,GPIO.RISING,callback=button_callbackStop)

while(run):
    print(run)
    get_firebase()
    time.sleep(0.2)
    
    
    
    
    
