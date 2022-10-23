#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
from check_cam_fps import check_fps
import make_train_data as mtd
import light_remover as lr
import ringing_alarm as alarm

#EAR 계산
def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#눈 뜬 경우의 EAR값 측정
def init_open_ear() :
    time.sleep(5)
    print("open init time sleep")
    ear_list = []
    th_message1 = Thread(target = open_message)
    th_message1.deamon = True
    th_message1.start()
    time.sleep(2)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    th_message3 = Thread(target = finish_message)
    th_message3.start()
    time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)
    print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")

#눈 감은 경우의 EAR값 측정
def init_close_ear() : 
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    ear_list = []
    th_message2 = Thread(target = close_message)
    th_message2.deamon = True
    th_message2.start()
    time.sleep(2)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    th_message3 = Thread(target = finish_message)
    th_message3.start()
    time.sleep(1)
    global CLOSE_EAR
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR)
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :",EAR_THRESH, "\n")

def open_message() :
    print("open your eyes")
    alarm.sound_alarm("openeye.wav")
    
def close_message() :
    print("close your eyes")
    alarm.sound_alarm("closeeye.wav")

def finish_message() :
    print("finish_message")
    alarm.sound_alarm("finish.wav")

#####################################################################################################################
#1. EAR 값에 대한 변수들
#2. 운전자가 졸음 운전중인지 판단하기 위한 변수들
#3. 알람이 울리면 눈을 감은 시간 측정
#4. 알람이 울리면 알람이 울린 횟수를 세어 알람이 계속 울리지 않도록 함
#5. 데이터 라벨링을 위해 눈 뜨고있었던 시간 측정
#6. 훈련된 데이터 생성을 위한 변수와 FPS 계산할 변수 
#7. 얼굴 & 눈 검출
#8. 카메라 실행
#9. EAR_THRESH를 결정하는 함수를 실행할 스레드
#10. 프레임에 사용되는 변수

#1.
OPEN_EAR = 0 #초기 눈 뜬경우 EAR값
EAR_THRESH = 0 #임계값

#2.
EAR_CONSEC_FRAMES = 15 
COUNTER = 0 #눈 감은 프레임 수
CNT = 0 #화면에 얼굴이 감지되지 않는 프레임 수
THRESH = 10 #임계값

#3.
closed_eyes_time = [] #눈 감은 시간
TIMER_FLAG = False #눈 감은 시간을 측정하는 'start_closing' 변수를 활성화하는 플래그
ALARM_FLAG = False #알람이 동작하는지 확인하는 플래그

#4. 
ALARM_COUNT = 0 #총 알람이 울린 횟수
RUNNING_TIME = 0  #알람이 계속 울리지 않도록 하는 변수

#5.    
PREV_TERM = 0  #알람이 울리기 전까지 눈 뜬 시간을 측정하는 변수

#6.
np.random.seed(30) #교육된 데이터 생성
power, nomal, short = mtd.start(25)  
test_data = []  #실제 테스트 데이터들을 저장하는 배열
result_data = [] #실제 라벨링된 테스트 데이터들을 저장하는 배열
prev_time = 0 #FPS 계산하기 위한 변수

#7. 
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#8.
print("starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

#9.
th_open = Thread(target = init_open_ear) # 눈 뜬 경우의 초기 EAR값 측정
th_open.deamon = True
th_open.start()
th_close = Thread(target = init_close_ear) # 눈 감은 경우의 초기 EAR값 측정
th_close.deamon = True
th_close.start()

#10.
STATUS = 'Awake' #운전자 상태
COLOR = None #글씨 색상
LEVEL = None #졸음 단계

#####################################################################################################################

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    
    L, gray = lr.light_removing(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray,0)
    
    #checking fps. If you want to check fps, just uncomment below two lines.
    #prev_time, fps = check_fps(prev_time)
    #cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    for rect in rects:
        #얼굴 검출
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        #눈 검출
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #(leftEAR + rightEAR) / 2 => both_ear. 
        both_ear = (leftEAR + rightEAR) * 500 
        
        #눈을 따라서 그림
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        
        #눈 감은 경우
        if both_ear < EAR_THRESH :
            COLOR = (0, 0, 255)
            if not TIMER_FLAG:
                start_closing = timeit.default_timer()
                TIMER_FLAG = True
            COUNTER += 1

            #졸음 운전인 경우
            if COUNTER >= EAR_CONSEC_FRAMES:

                mid_closing = timeit.default_timer()
                closing_time = round((mid_closing-start_closing),3)

                if closing_time >= RUNNING_TIME:
                    frame = gray
                    if RUNNING_TIME == 0 :
                        CUR_TERM = timeit.default_timer()
                        OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM),3)
                        PREV_TERM = CUR_TERM
                        RUNNING_TIME = 1.75

                    RUNNING_TIME += 2 #알람이 동시에 발생되지 않도록
                    ALARM_FLAG = True
                    ALARM_COUNT += 1

                    print("{0}st ALARM".format(ALARM_COUNT))
                    print("The time eyes is being opened : ", OPENED_EYES_TIME)
                    print("closing time :", closing_time)
                    
                    #알람 레벨 선택
                    test_data.append([OPENED_EYES_TIME, round(closing_time*3.3, 3)])
                    result = mtd.run([OPENED_EYES_TIME, closing_time*3.3], power, nomal, short)
                    result_data.append(result)
                    LEVEL = result
                    if LEVEL == 0:
                        STATUS = 'SLEEPING'
                    elif LEVEL == 1:
                        STATUS = 'FEEL SLEEPY'
                    elif LEVEL == 2:
                        STATUS = 'TIRED'
                    t = Thread(target = alarm.select_alarm, args = (result, ))
                    t.start()
        
        #눈 뜬 경우
        else :
            STATUS = 'AWAKE'
            COLOR = (0, 255, 0)
            COUNTER = 0
            TIMER_FLAG = False
            RUNNING_TIME = 0
            
            #눈을 감았다가 눈을 뜬 경우
            if ALARM_FLAG :
                end_closing = timeit.default_timer()
                closed_eyes_time.append(round((end_closing-start_closing),3))
                print("The time eyes were being offed :", closed_eyes_time)

            ALARM_FLAG = False

        cv2.putText(frame, "{}st ALARM".format(ALARM_COUNT),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        cv2.putText(frame, "STATUS : {}".format(STATUS) ,(10,270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        cv2.putText(frame, "LEVEL  : {}".format(LEVEL) ,(10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
    
    #얼굴이 화면에 보이지 않는 경우
    if not rects :
        CNT += 1
        COLOR = (0, 255, 0)
        if CNT > THRESH :
            frame = gray
            t = Thread(target = alarm.select_alarm, args = (0, ))
            t.start()
        ALARM_FLAG = False
        cv2.putText(frame, "Please look at the camera", (50, 130),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
    

    cv2.imshow("DROWSY DETECTION",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()


# In[ ]:




