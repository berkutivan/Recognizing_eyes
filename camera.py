import cv2

class Camera:
    counter = 0
    def __init__(self, frame_width = None, frame_height = None, camera_ind = 0):
        Camera.counter += 1
        self.cam = cv2.VideoCapture(camera_ind)
        if frame_width == None:
            self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            self.frame_width = frame_width

        if frame_height == None:
            self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            self.frame_height = frame_height
        print(self.frame_width, self.frame_height)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(f'output{Camera.counter}.mp4', self.fourcc, 20.0, (self.frame_width, self.frame_height))

    def start(self, func = None):
        while True:
            ret, frame = self.cam.read()
            if ret:
                if func == None:
                    pross_frame = frame
                else:
                    pross_frame = func(frame)#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.out.write(pross_frame)
                cv2.imshow('Camera', pross_frame)


            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and writer objects
        self.cam.release()
        self.out.release()
        cv2.destroyAllWindows()



#cam = Camera()
#cam.start(func = lambda image:cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) )