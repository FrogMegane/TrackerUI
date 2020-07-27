from PyQt5 import QtWidgets
from mainForm import MainUi
import sys
import cv2
import os

#Functions Write Here
class M:
    def __init__(self):
        self.resetAll()
    
    '''
        System
    '''
    def closeVideo(self):
        self.cap.release()
        self.resetAll()

    def resetAll(self):
        self.video_path = ''
        self.video_name = ''
        self.width=0
        self.height=0
        self.size=(0,0)
        self.length=0
        self.fps=0
        self.cap= None
        self.curFrame=None

    def loadVideo(self,filename):
        #return true if successed else false
        self.cap = cv2.VideoCapture(filename)
        if self.cap.isOpened():            
            self.video_path,self.video_name = os.path.split(filename)
            self.width,self.height = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.size=(self.width,self.height)
            self.fps=self.cap.get(cv2.CAP_PROP_FPS)
            self.length=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return True
        else:
            return False

    '''
        get values
    '''
    def getImageByIndex(self,index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        check,self.curFrame=self.cap.read() # bool, Frame  (bool: if successed)
        return check,self.curFrame.copy()

    def getLength(self):
        return self.length
    def getFPS(self):
        return self.fps
        
    def getWidth(self):
        return self.width
    def getHeight(self):
        return self.height

    '''
        Functions
    '''
    def clipThenSave(self,startIdx,endIdx):
        if endIdx<startIdx:
            indexes=range(startIdx,endIdx-1,-1)#reversed video index to be saved
        else:
            indexes=range(startIdx,endIdx+1)# index to be saved

        # Video Segment
        frame_list=[]
        for i in indexes:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            check,tmpFrame=self.cap.read()
            frame_list.append(tmpFrame)
            assert(check)

        # Saving
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        '''
            #Todo: maybe *.avi will be saved in *_xxx_xxx.avi but it is saved in mp4 encoded
        '''
        save_path=os.path.join(self.video_path,self.video_name.replace('.','_{:03d}_{:03d}.'.format(startIdx,endIdx)))
        out = cv2.VideoWriter(save_path,fourcc, self.fps, self.size)
        for frame in frame_list:
            out.write(frame)
        out.release()
        print('Saving Done')

    
if __name__ == "__main__":
    M=M()
    app = QtWidgets.QApplication(sys.argv)
    window = MainUi(M)

    #window.show()
    window.showMaximized()
    #window.showFullScreen()
    app.exec_()
