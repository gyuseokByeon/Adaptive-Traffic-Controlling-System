from object_Tracking import *
import os
import cv2
import numpy as np
import darknet
import mask_parameter as mask_parameter
from threading import Thread, Lock
from multiprocessing import Process, Queue
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import time
from datetime import datetime
import pandas as pa

now = datetime.now()

fps = 10
dt_string = now.strftime("%m_%d_%H_%M")
write_path = 'add write path here'
# Fetch the service account key JSON file contents
cred = credentials.Certificate(
    'Add firebase URL')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'database URL'
})
ref = db.reference('input')


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cordConverter(detections, class_list):
    detBox = []
    for detection in detections:
        class_ = class_list.index(detection[0].decode('ASCII'))
        score = detection[1]
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))

        detBox.append([xmin, ymin, xmax, ymax, score, class_])
    detBox = np.array(detBox)
    return detBox


class CameraStream(object):
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


class firebase_reader(object):
    def __init__(self, ref_):
        self.started = False
        self.ref = ref_
        self.read_lock = Lock()
        self.data = {'camera': -1}

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            data = self.ref.get()
            self.read_lock.acquire()
            self.data = data
            self.read_lock.release()
            # time.sleep(0.1)

    def read(self):
        self.read_lock.acquire()
        data = self.data
        self.read_lock.release()
        return data['camera']

    def stop(self):
        self.started = False
        self.thread.join()


def mask_creater(shape, roi_indicater):
    mask = np.zeros(shape, dtype=np.uint8)
    if roi_indicater == 1:
        roi_corners = mask_parameter.roi_corners_1
    elif roi_indicater == 2:
        roi_corners = mask_parameter.roi_corners_2
    elif roi_indicater == 3:
        roi_corners = mask_parameter.roi_corners_3
    elif roi_indicater == 4:
        roi_corners = mask_parameter.roi_corners_4

    cv2.fillPoly(mask, roi_corners, (1, 1, 1))
    return mask


def img_operations(frame, mask, split):
    if frame.shape != (576, 704, 3):
        frame = cv2.resize(frame, (704, 576))
        print('Low resolution images are transmitted')
    frame = frame * mask
    frame = frame[split[1]:, split[0]:]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
    return frame


def cvDrawBoxes(detections, img):
    detections = detections.tolist()
    for detection in detections:
        xmin = int(detection[0])
        ymin = int(detection[1])
        xmax = int(detection[2])
        ymax = int(detection[3])

        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    str(int(detection[4])),
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    [0, 0, 0], 1)
    return img


configPath = "Config file path"
weightPath = "weight file path"
metaPath = "metadata path"

# load once
netMain = None
metaMain = None
altNames = None
if netMain is None:
    netMain = darknet.load_net_custom(configPath.encode(
        "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
if metaMain is None:
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
if altNames is None:
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re

            match = re.search("names *= *(.*)$", metaContents,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass
darknet_image = darknet.make_image(darknet.network_width(netMain),
                                   darknet.network_height(netMain), 3)


def read_images(queue_channel1_, queue_channel2_, phase_info, fps):
    shape_img = (576, 704, 3)
    mask_1 = mask_creater(shape_img, 1)
    mask_2 = mask_creater(shape_img, 2)
    mask_3 = mask_creater(shape_img, 3)
    mask_4 = mask_creater(shape_img, 4)

    split_1 = mask_parameter.split_1
    split_2 = mask_parameter.split_2
    split_3 = mask_parameter.split_3
    split_4 = mask_parameter.split_4

    cap1_ = CameraStream(src='ip address ofthe video stream_cam1').start()
    cap2_ = CameraStream(src='ip address ofthe video stream_cam2').start()
    cap3_ = CameraStream(src='ip address ofthe video stream_cam3').start()
    cap4_ = CameraStream(src='ip address ofthe video stream_cam4').start()

    firebase = firebase_reader(ref).start()

    prev_time = 0
    prev_phase = str(firebase.read())
    print('whole process is started')

    flag = False
    temp_stop = False
    while True:
        phase_ = firebase.read()
        if phase_ is None:
            while phase_ is None:
                phase_ = firebase.read()
        phase_ = str(phase_)
        t1 = time.time()

        if phase_ != '-1':
            if phase_ == '4':
                queue_channel1.put('end')
                queue_channel2.put('end')
                break
            if (prev_phase == '1') & (phase_ == '0'):
                temp_stop = True
                queue_channel1.put('phase 1 ended')
                queue_channel2.put('phase 1 ended')
                print('read phase 1 ended')
            if (prev_phase == '2') & (phase_ == '0'):
                temp_stop = True
                queue_channel1.put('phase 2 ended')
                queue_channel2.put('phase 2 ended')
                print('read phase 2 ended')
            if (prev_phase == '3') & (phase_ == '0'):
                temp_stop = True
                queue_channel1.put('phase 3 ended')
                queue_channel2.put('phase 3 ended')
                print('read phase 3 ended')

            if (prev_phase == '0') & (phase_ == '2'):
                # print('phase ', phase_, ' read data started')
                temp_stop = False
                queue_channel1.put('phase 2 started')
                queue_channel2.put('phase 2 started')
                print('read phase 2 started')

            if (prev_phase == '0') & (phase_ == '3'):
                temp_stop = False
                queue_channel1.put('phase 3 started')
                queue_channel2.put('phase 3 started')
                print('read phase 3 started')

            if (prev_phase == '0') & (phase_ == '1'):
                temp_stop = False
                queue_channel1.put('phase 1 started')
                queue_channel2.put('phase 1 started')
                print('read phase 1 started')
                flag = 1

            if flag & (not temp_stop):
                if t1 - prev_time > 1 / fps:
                    if phase_ == '1':
                        frame1_ = cap1_.read()
                        frame2_ = cap2_.read()
                        frame1_ = img_operations(frame1_, mask_1, split_1)
                        frame2_ = img_operations(frame2_, mask_2, split_2)
                        queue_channel1_.put(frame1_)
                        queue_channel2_.put(frame2_)
                        # cv2.imshow('channel1', frame1_)
                        # cv2.imshow('channel2', frame2_)
                        # cv2.waitKey(1)

                    elif phase_ == '2':
                        frame1_ = cap4_.read()
                        frame2_ = cap2_.read()
                        frame1_ = img_operations(frame1_, mask_4, split_4)
                        frame2_ = img_operations(frame2_, mask_2, split_2)
                        queue_channel1_.put(frame1_)
                        queue_channel2_.put(frame2_)
                        # cv2.imshow('channel1', frame1_)
                        # cv2.imshow('channel2', frame2_)
                        # cv2.waitKey(1)

                    elif phase_ == '3':
                        frame1_ = cap4_.read()
                        frame2_ = cap3_.read()
                        frame1_ = img_operations(frame1_, mask_4, split_4)
                        frame2_ = img_operations(frame2_, mask_3, split_3)
                        queue_channel1_.put(frame1_)
                        queue_channel2_.put(frame2_)
                        # cv2.imshow('channel1', frame1_)
                        # cv2.imshow('channel2', frame2_)
                        # cv2.waitKey(1)

                    if ((t1 - prev_time) > 1 / 10) & ((t1 - prev_time) < 1 / 8):
                        pass
                    # else:
                    #     print('Failiure')
                    prev_time = t1

            # else:
            # cv2.destroyAllWindows()
            # print('no update')
            prev_phase = phase_


def detection_on_phase(queue_channel1_, queue_channel2_, queue1, queue2):
    global darknet_image
    class_list = ['motorbike', 'tuk', 'car', 'bus']
    while True:

        if (queue_channel1_.qsize() != 0) & (queue_channel2_.qsize() != 0):
            # print(queue_channel2_.qsize())
            frame_1 = queue_channel1_.get()
            frame_2 = queue_channel2_.get()

            if (type(frame_1) == str) & (type(frame_2) == str):
                # print('Detection is started')
                queue1.put(frame_1)
                queue2.put(frame_2)
                if frame_1 == 'end':
                    break
            else:
                darknet.copy_image_from_bytes(darknet_image, frame_1.tobytes())
                detections_1 = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.35)
                darknet.copy_image_from_bytes(darknet_image, frame_2.tobytes())
                detections_2 = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.35)

                detBox_1 = cordConverter(detections_1, class_list)
                detBox_2 = cordConverter(detections_2, class_list)

                queue1.put([frame_1, detBox_1])
                queue2.put([frame_2, detBox_2])

        else:
            continue
    return


def tracking_channel(inp_queue1, inp_queue2, out_queue1, out_queue2, index):
    flag = 0
    ref_update = db.reference('output')
    mot_tracker1 = Sort(max_age=15, min_hits=3, channel=index[0])
    mot_tracker2 = Sort(max_age=15, min_hits=3, channel=index[1])
    class_list = ['bike', 'tuk', 'car', 'others']
    pcu = np.array([0.7, 0.9, 1, 2.2])
    uniq_ids_class1 = np.empty((0, 4))
    uniq_ids_class2 = np.empty((0, 4))

    g1_last = 32
    g2_last = 35
    g3_last = 48
    cycle_time = 115
    last_interupt = 0
    eff_ph1_last = -1
    eff_ph2_last = -1
    eff_ph3_last = -1
    alpha = 0.85
    file_name = 'flow rate results file_' + dt_string + '.csv'
    while True:

        if not inp_queue1.empty() & inp_queue2.empty():
            detect_info1 = inp_queue1.get()
            detect_info2 = inp_queue2.get()

            if (type(detect_info1) == str) & (type(detect_info2) == str):
                out_queue1.put(detect_info1)
                out_queue2.put(detect_info1)
                if detect_info1 == 'end':
                    break
                if flag == 1:
                    mot_tracker1.reset()
                    mot_tracker2.reset()
                    phase_data = np.empty((1, 0))
                    if detect_info1 == 'phase 1 ended':
                        # channel 4
                        # bike:  38.0
                        # auto:  40.0
                        # car:  39.0
                        # bus:  32.0

                        # channel 2
                        # bike:  47.0
                        # auto:  39.0
                        # car:  52.0
                        # bus:  32.0

                        veh_max_speed_1 = [38, 40, 39, 32]
                        veh_max_speed_2 = [47, 39, 52, 32]
                        temp_tot = 0
                        i = 0
                        for i in range(4):
                            same_class_vehicle_1 = uniq_ids_class1[uniq_ids_class1[:, 1] == i]
                            print(i, same_class_vehicle_1.shape)
                            clss_max_speed_1 = veh_max_speed_1[i]
                            veh_flow_rate_1 = same_class_vehicle_1[:, 3] / clss_max_speed_1

                            same_class_vehicle_2 = uniq_ids_class2[uniq_ids_class2[:, 1] == i]
                            clss_max_speed_2 = veh_max_speed_2[i]
                            veh_flow_rate_2 = same_class_vehicle_2[:, 3] / clss_max_speed_2

                            temp_tot = temp_tot + veh_flow_rate_1.sum() + veh_flow_rate_2.sum()

                        tot_frames = uniq_ids_class1[:, 2].max()
                        green_utilization_1 = temp_tot / (2 * tot_frames)
                        effective_green_1 = g1_last * green_utilization_1
                        time_ = datetime.now()
                        cur_time = time_.strftime("%H_%M_%S")
                        phase_data = np.append(phase_data, [cur_time, 1])
                        phase_data = np.append(phase_data, [temp_tot, tot_frames, effective_green_1])
                        phase_data = np.append(phase_data, g1_last)
                        uniq_ids_class1 = np.empty((0, 4))
                        uniq_ids_class2 = np.empty((0, 4))

                        if eff_ph1_last < 0:
                            eff_ph1_last = effective_green_1
                            print('init')
                        else:
                            effective_green_1 = alpha * effective_green_1 + (1 - alpha) * eff_ph1_last
                            eff_ph1_last = effective_green_1
                        phase_data = np.append(phase_data, effective_green_1)

                        df = pa.DataFrame(np.reshape(phase_data, (1, 7)))
                        df.to_csv(file_name, mode='a', header=False, index=False)
                        print('green_utilization_1: ', effective_green_1)

                    if detect_info1 == 'phase 2 ended':

                        # channe 3
                        # bike:  39.0
                        # auto:  38.0
                        # car:  43.0
                        # bus:  34.0

                        # channel 4
                        # bike:  38.0
                        # auto:  40.0
                        # car:  39.0
                        # bus:  32.0

                        veh_max_speed_1 = [39, 38, 43, 34]
                        veh_max_speed_2 = [38, 40, 39, 32]
                        tot_flow_rate = 0
                        j = 0
                        for j in range(4):
                            same_class_vehicle_1 = uniq_ids_class1[uniq_ids_class1[:, 1] == j]
                            clss_max_speed_1 = veh_max_speed_1[j]
                            veh_flow_rate_1 = same_class_vehicle_1[:, 3] / clss_max_speed_1

                            same_class_vehicle_2 = uniq_ids_class2[uniq_ids_class2[:, 1] == j]
                            clss_max_speed_2 = veh_max_speed_2[j]
                            veh_flow_rate_2 = same_class_vehicle_2[:, 3] / clss_max_speed_2

                            tot_flow_rate = tot_flow_rate + veh_flow_rate_1.sum() + veh_flow_rate_2.sum()

                        tot_frames = uniq_ids_class1[:, 2].max()
                        green_utilization_2 = tot_flow_rate / (2 * tot_frames)
                        effective_green_2 = g2_last * green_utilization_2

                        time_ = datetime.now()
                        cur_time = time_.strftime("%H_%M_%S")
                        phase_data = np.append(phase_data, [cur_time, 2])
                        phase_data = np.append(phase_data, [tot_flow_rate, tot_frames, effective_green_2])
                        phase_data = np.append(phase_data, g2_last)
                        uniq_ids_class1 = np.empty((0, 4))
                        uniq_ids_class2 = np.empty((0, 4))

                        if eff_ph2_last < 0:
                            eff_ph2_last = effective_green_2
                        else:
                            effective_green_2 = alpha * effective_green_2 + (1 - alpha) * eff_ph2_last
                            eff_ph2_last = effective_green_2

                        phase_data = np.append(phase_data, effective_green_2)
                        df = pa.DataFrame(np.reshape(phase_data, (1, 7)))
                        df.to_csv(file_name, mode='a', header=False, index=False)
                        print('green_utilization_2: ', effective_green_2)

                    if detect_info1 == 'phase 3 ended':

                        # channel 1
                        # bike:  48.0
                        # auto:  50.0
                        # car:  54.0
                        # bus:  46.0

                        # channel 2
                        # bike:  47.0
                        # auto:  39.0
                        # car:  52.0
                        # bus:  32.0

                        veh_max_speed_1 = [48, 50, 54, 46]
                        veh_max_speed_2 = [47, 39, 52, 32]
                        tot_flow_rate = 0
                        k = 0
                        for k in range(4):
                            print(uniq_ids_class1.shape)
                            same_class_vehicle_1 = uniq_ids_class1[uniq_ids_class1[:, 1] == k]
                            clss_max_speed_1 = veh_max_speed_1[k]
                            veh_flow_rate_1 = same_class_vehicle_1[:, 3] / clss_max_speed_1

                            same_class_vehicle_2 = uniq_ids_class2[uniq_ids_class2[:, 1] == k]
                            clss_max_speed_2 = veh_max_speed_2[k]
                            veh_flow_rate_2 = same_class_vehicle_2[:, 3] / clss_max_speed_2

                            tot_flow_rate = tot_flow_rate + veh_flow_rate_1.sum() + veh_flow_rate_2.sum()

                        tot_frames = uniq_ids_class1[:, 2].max()
                        green_utilization_3 = tot_flow_rate / (2 * tot_frames)
                        effective_green_3 = g3_last * green_utilization_3

                        time_ = datetime.now()
                        cur_time = time_.strftime("%H_%M_%S")
                        phase_data = np.append(phase_data, [cur_time, 3])
                        phase_data = np.append(phase_data, [tot_flow_rate, tot_frames, effective_green_3])
                        phase_data = np.append(phase_data, g3_last)

                        uniq_ids_class1 = np.empty((0, 4))
                        uniq_ids_class2 = np.empty((0, 4))

                        if eff_ph3_last < 0:
                            eff_ph3_last = effective_green_3
                        else:
                            effective_green_3 = alpha * effective_green_3 + (1 - alpha) * eff_ph3_last
                            eff_ph3_last = effective_green_3

                        phase_data = np.append(phase_data, effective_green_3)
                        df = pa.DataFrame(np.reshape(phase_data, (1, 7)))
                        df.to_csv(file_name, mode='a', header=False, index=False)
                        print('green_utilization_3: ', effective_green_3)

                        tot_sum = np.sqrt(effective_green_1) + np.sqrt(effective_green_2) + np.sqrt(
                            effective_green_3)

                        if tot_sum != 0:
                            g1_new = cycle_time * (np.sqrt(effective_green_1) / tot_sum)
                            g2_new = cycle_time * (np.sqrt(effective_green_2) / tot_sum)
                            g3_new = cycle_time * (np.sqrt(effective_green_3) / tot_sum)
                            print('total new timing g1,g2,g3 new', g1_new, g2_new, g3_new)
                        else:
                            g1_new = g1_last
                            g2_new = g2_last
                            g3_new = g3_last

                        if g1_new > g1_last:
                            g1_update = int(min(g1_new - g1_last, 3))
                        else:
                            g1_update = int(max(g1_new - g1_last, -3))

                        if g2_new > g2_last:
                            g2_update = int(min(g2_new - g2_last, 3))
                        else:
                            g2_update = int(max(g2_new - g2_last, -3))

                        if g3_new > g3_last:
                            g3_update = int(min(g3_new - g3_last, 3))
                        else:
                            g3_update = int(max(g3_new - g3_last, -3))

                        abs_g1_update = np.abs(g1_new - g1_last)
                        abs_g2_update = np.abs(g2_new - g2_last)
                        abs_g3_update = np.abs(g3_new - g3_last)
                        min_update = min(abs_g1_update, abs_g2_update, abs_g3_update)
                        print('abs time changes', abs_g1_update, abs_g2_update, abs_g3_update)

                        if min_update == abs_g1_update:
                            g1_update = -(g2_update + g3_update)
                        elif min_update == abs_g2_update:
                            g2_update = -(g1_update + g3_update)
                        elif min_update == abs_g3_update:
                            g3_update = -(g2_update + g1_update)

                        g1_temp = g1_last + g1_update
                        g2_temp = g2_last + g2_update
                        g3_temp = g3_last + g3_update
                        bool1 = 7 < g1_temp < 90
                        bool2 = 7 < g2_temp < 90
                        bool3 = 7 < g3_temp < 90
                        decide_to_change = bool1 & bool2 & bool3
                        if decide_to_change:
                            time1 = int(np.abs(g1_update))
                            direction1 = int((1 + np.sign(g1_update)) / 2)
                            time2 = int(np.abs(g2_update))
                            direction2 = int((1 + np.sign(g2_update)) / 2)
                            time3 = int(np.abs(g3_update))
                            direction3 = int((1 + np.sign(g3_update)) / 2)
                            g1_last = g1_temp
                            g2_last = g2_temp
                            g3_last = g3_temp
                            print('updated')
                        else:
                            time1 = 0
                            time2 = 0
                            time3 = 0
                            direction1 = 0
                            direction2 = 0
                            direction3 = 0
                            print('not updated')

                        interupt = int(1 - last_interupt)
                        print('writing in firebase')
                        print('next phase setting', g1_last, g2_last, g3_last)
                        print('time updates', time1, time2, time3)
                        ref_update.set({
                            "direction1": direction1,
                            "direction2": direction2,
                            "direction3": direction3,
                            "interupt": interupt,
                            "time1": time1,
                            "time2": time2,
                            "time3": time3})

                        last_interupt = interupt
                if detect_info1 == 'phase 1 started':
                    flag = 1

            elif flag == 1:

                track_bbs_ids1 = mot_tracker1.update(detect_info1[1])
                track_bbs_ids2 = mot_tracker2.update(detect_info2[1])

                out_queue1.put([detect_info1[0], track_bbs_ids1])
                out_queue2.put([detect_info2[0], track_bbs_ids2])

                uniq_ids_class1 = np.concatenate([uniq_ids_class1, track_bbs_ids1[:, [4, 7, 9, 10]]], axis=0)
                uniq_ids_class2 = np.concatenate([uniq_ids_class2, track_bbs_ids2[:, [4, 7, 9, 10]]], axis=0)

        else:
            continue


def visualize(queue_track1, queue_track2, fps):
    while True:
        if (queue_track1.qsize() != 0) & (queue_track2.qsize() != 0):
            track_info1 = queue_track1.get()
            track_info2 = queue_track2.get()

            if (type(track_info1) == str) & (type(track_info2) == str):

                cv2.destroyAllWindows()
                if track_info1 == 'end':
                    break
                continue
            else:
                img1 = cv2.cvtColor(track_info1[0], cv2.COLOR_RGB2BGR)
                img2 = cv2.cvtColor(track_info2[0], cv2.COLOR_RGB2BGR)
                img1 = cvDrawBoxes(track_info1[1], img1)
                img2 = cvDrawBoxes(track_info2[1], img2)

                cv2.imshow('channel1', img1)
                cv2.imshow('channel2', img2)
                cv2.waitKey(int(1000 / fps))
        else:
            continue

"""Solve the linear assignment problem using the Hungarian algorithm.
    The problem is also known as maximum weight matching in bipartite graphs.
    The method is also known as the Munkres or Kuhn-Munkres algorithm.
    Parameters
    ----------
    X : array
        The cost matrix of the bipartite graph
    Returns
    -------
    indices : array
        The pairs of (row, col) indices in the original array giving
        the original ordering.
    References
    ----------
    1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
    2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
       *Naval Research Logistics Quarterly*, 2:83-97, 1955.
    3. Harold W. Kuhn. Variants of the Hungarian method for assignment
       problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
    4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
       *Journal of the Society of Industrial and Applied Mathematics*,
       5(1):32-38, March, 1957.
    5. https://en.wikipedia.org/wiki/Hungarian_algorithm
    """
if __name__ == '__main__':
    queue_channel1 = Queue()
    queue_channel2 = Queue()
    queue_detection1 = Queue()
    queue_detection2 = Queue()
    queue_tracks1 = Queue()
    queue_tracks2 = Queue()

    process1 = Process(target=read_images, args=(queue_channel1, queue_channel2, 'phase_info.txt', 10))
    process2 = Process(target=tracking_channel,
                       args=(queue_detection1, queue_detection2, queue_tracks1, queue_tracks2, (1, 2)))
    process3 = Process(target=visualize, args=(queue_tracks1, queue_tracks2, 10))

    process1.start()
    process2.start()
    process3.start()
    detection_on_phase(queue_channel1, queue_channel2, queue_detection1, queue_detection2)

    process1.join()
    process2.join()
    process3.join()
