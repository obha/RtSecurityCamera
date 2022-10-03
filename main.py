
import time
from threading import Thread

import cv2
import numpy as np

from utils import Arguments

args = Arguments()


def frame_processor(frame):
    classes = []

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    def get_output_layers(net):

        layer_names = net.getLayerNames()

        output_layers = [layer_names[i-1]
                         for i in net.getUnconnectedOutLayers()]

        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label, (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    width = frame.shape[1]
    height = frame.shape[0]
    scale = 0.00392

    # read pre-trained model and config file
    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(
        frame, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(frame, class_ids[i], confidences[i], round(
            x), round(y), round(x+w), round(y+h))

    cv2.imshow("detect", frame)


def empty_frame_processor(frame):
    cv2.imshow("detect", frame)


def chunk_stream(batch_size):
    num = 0
    while num < batch_size:
        yield "PENDING"
        num += 1
    yield "DONE"


class VideoStream(object):
    def __init__(self, src=0) -> None:
        self.fps = 0
        self.frame_processor = None
        self.capture = cv2.VideoCapture(src)

        self.reducer = self.reduce(None, 0)

        self.video_thread = Thread(target=self.update)
        self.video_thread.start()

    def setFrameProcessor(self, processor):
        self.frame_processor = processor

    def update(self):
        self.reducer = next(self.reducer)

        while self.reducer != None:

            self.reducer = next(self.reducer)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

            print("FPS: {0}".format(self.fps))

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.capture.release()
                cv2.destroyAllWindows()
                exit(0)

    def reduce(self, chunk=None, batch_size=30):

        if (chunk == None):
            chunk = chunk_stream(batch_size)

        val = next(chunk)

        start_time = time.time()

        while val == "PENDING":
            val = next(chunk)

        (_, frame) = self.capture.read()

        if self.frame_processor != None:
            self.frame_processor(frame)

        elapsed_time = time.time() - start_time

        self.fps = 1/elapsed_time

        yield self.reduce(chunk_stream(batch_size), batch_size)


class Camera(Thread):
    def __init__(self, src=0):
        Thread.__init__(self)
        self.capture = cv2.VideoCapture(src)

    def run(self) -> None:
        while True:
            self = next(self)

    def __iter__(self):
        return self

    def __next__(self):
        (_, frame) = self.capture.read()
        return frame


def main():
    cam = Camera(args.source)

    for frame in cam:

        if frame is None:
            exit(0)

        cv2.imshow("", frame)

        print(frame)

    cam.start()

    # video = VideoStream(args.source)
    # video.setFrameProcessor(frame_processor)


if __name__ == "__main__":
    main()
