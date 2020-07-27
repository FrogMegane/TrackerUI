from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import math

from PyQt5 import QtWidgets, uic,QtCore
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QFileDialog,QGraphicsScene,QGraphicsPixmapItem,QApplication
from PyQt5.QtGui import  QPixmap,QImage
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

#loat xxx.ui
path = os.getcwd()
qtCreatorFile = path + os.sep + "mainForm" + os.sep + "Main_Window.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


#GUI Write Here
class MainUi(QtWidgets.QMainWindow, Ui_MainWindow):  # Python的多重繼承 MainUi 繼承自兩個類別
    def __init__(self,M):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        '''
        QSlider::groove:horizontal {\
                border: 1px solid;\
                height: 200px;\
                margin: 0px;\
                }\
            QSlider::handle:horizontal {\
                background-color: black;\
                border: 1px solid;\
                height: 200px;\
                width: 2px;\
                margin: -15px 0px;

        self.slider.setStyleSheet("\
            .QSlider::handle:horizontal {\
                background-color: red;\
                height: 608px;\
                width: 5px;\
            ")
        '''
        self.slider.setStyleSheet(".QSlider::handle:horizontal {background: red; width: 1px; height: 608px;}")
        
        
        self.M=M#Functions are called via self.M
        self.initGlobalVar()
        self.initLoacalVar()# init video specific variables
        self.onBind() # btn,.... signals & slots
    
    def initGlobalVar(self):
        self.default_path = os.path.expanduser("~/Desktop")# works in windows10, linux, Mac Not sure
        self.isVideoOpened=False
        self.isPlaying=False

        #GV1
        self.scene1 = QGraphicsScene(self)
        self.pixmap_item1 = QGraphicsPixmapItem()
        self.gv1.setScene(self.scene1)
        self.scene1.addItem(self.pixmap_item1)
        #GV2
        self.scene2 = QGraphicsScene(self)
        self.pixmap_item2 = QGraphicsPixmapItem()
        self.gv2.setScene(self.scene2)
        self.scene2.addItem(self.pixmap_item2)
        #GV3
        self.scene3 = QGraphicsScene(self)#
        self.gv3.setScene(self.scene3)#
        self.canvans=None

    def initLoacalVar(self):
        # Will need to be changed, when re-load another video
        self.lastFrameIdx=-1
        self.videoName=""
        self.csvName=""
        self.curFrame=None
        self.centers={}
        self.Angles=[]
        self.lineX=None
        self.lineY=None
    
    def resetAll(self):
        #if self.isVideoOpened:
        #    self.M.closeVideo()    
        self.sbFrameIdx.setValue(0)
        self.videoName=""
        self.csvName=""
        self.centers.clear()
        self.Angles=[]
        self.lineX=None
        self.lineY=None

        

    
    '''
        Utilities
    '''

    def Mat2QImg(self,cvImage):
        height, width, channel = cvImage.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImage.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return qImg
    
    def readAndShow(self,index):
        successed,cvImg=self.M.getImageByIndex(index)
        #self.curFrame=cvImg.clone()
        if successed:
            if self.centers:
                center=self.centers[index]
                self.draw_cross(cvImg,center,10)

            
            self.setImage1(cvImg)
            self.updateProgressBar(index)
            #self.drawLineChart()
        else:
            print('Failed to read {:3d}th frame'.format(index))

    def draw_cross(self,frame,center,cross_size=5):
        #cross_size=5# one size length
        cv2.line(frame,(int(center[0])-cross_size,int(center[1])),(int(center[0])+cross_size,int(center[1])),(0,0,255),2)
        cv2.line(frame,(int(center[0]),int(center[1])-cross_size),(int(center[0]),int(center[1])+cross_size),(0,0,255),2)
    
    
            

    def doRegression(self,sIdx,eIdx):
        def compute_angle(x,regCoef):
            #y=sum(lin_reg_2.coef_*X_poly[0])+lin_reg_2.intercept_
            c,b,a=regCoef
            angle=-math.atan(2*a*x+b)/math.pi*180 # y=>-y=ax^2+bx+c
            return angle
        datasets_X = []
        datasets_Y = []
        for i in range(sIdx,eIdx+1,1):
            center=self.centers[i]
            datasets_X.append(float(center[0]))
            datasets_Y.append(float(center[1]))
        datasets_X = np.array(datasets_X).reshape([-1,1])
        datasets_Y = np.array(datasets_Y)
            
        #data X Normalization
        poly_reg = PolynomialFeatures(degree=2)#Init
        X_poly = poly_reg.fit_transform(datasets_X)# 1,x,x**2
        
        #Linear Regression
        lin_reg_2 = linear_model.LinearRegression()
        lin_reg_2.fit(X_poly, datasets_Y)
        
        #Regression Line Segments
        minX,maxX = min(datasets_X),max(datasets_X)
        self.lineX = np.arange(minX, maxX).reshape([-1, 1])
        self.lineY=lin_reg_2.predict(poly_reg.fit_transform(self.lineX)).reshape([-1,1])

        regCoef=lin_reg_2.coef_
        Angles=[]
        for idx in range(sIdx,eIdx+1):
            angle=compute_angle(datasets_X[idx-sIdx],regCoef)
            Angles.append(angle)
        return Angles

    '''
        GUI Update
    '''
    def setImage1(self,cvImg):
        qimg=self.Mat2QImg(cvImg)
        self.pixmap_item1.setPixmap(QPixmap.fromImage(qimg))
        self.gv1.fitInView(self.pixmap_item1, QtCore.Qt.KeepAspectRatio)# fit size of piture to gv
    
    def setImage2(self,cvImg):
        qimg=self.Mat2QImg(cvImg)
        self.pixmap_item2.setPixmap(QPixmap.fromImage(qimg))
        self.gv2.fitInView(self.pixmap_item2, QtCore.Qt.KeepAspectRatio)# fit size of piture to gv
    
            
    def setVideoInformation(self,fps,width,height):
        self.lb_videoInfo.setText('FPS={:5.2f},  (W,H)=({:4d},{:4d})'.format(fps,width,height))

    def updateProgressBar(self,value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)

    def drawCenters(self):
        if self.centers and self.isVideoOpened:
            index=self.sbFrameIdx.value()

            successed,cvImg=self.M.getImageByIndex(index)
            if successed:
                #lines
                if self.lineX is not None and self.lineY is not None:
                    polylines=np.append(self.lineX,self.lineY, axis=1).astype(np.int32)
                    cv2.polylines(cvImg, [polylines], False, (255,0,0),2)
                #centers
                for key,center in self.centers.items():
                    self.draw_cross(cvImg,center)              
                self.setImage2(cvImg)
            else:
                print('Failed to read {:3d}th frame'.format(index))
            

    def drawLineChart(self):
        plt.cla()#clean

        #fugure and ax setup
        #fig = plt.figure(figsize=(20,5))
        fig = plt.figure()
        fig.tight_layout()
        ax =fig.add_axes([0.025,0.15,0.96,0.8])# Ratio of left, bottom, width, height margin respect to figsize
        ax.set_title('Frame-Angle Diagram')
        ax.set_xlabel('t frame')
        ax.set_ylabel('degree')

        #draw
        ax.set_xlim([0,self.lastFrameIdx])
        ax.set_ylim([-180,180])

        if self.centers:
            
            # Regression
            sIdx=self.sbRegStart.value()
            eIdx=self.sbRegEnd.value()
            self.Angles=self.doRegression(sIdx,eIdx)
            
            
            # Draw t-theta
            nums=self.lastFrameIdx+1
            T=[i for i in range(0,nums)]
            Theta=[0]*nums
            Theta[sIdx:eIdx+1]=self.Angles
            l1,=ax.plot(T,Theta,'o--',label='Angle')
            #l2,=ax.plot(data2,'o--',label='54321')

        #current frame
        ax.axvline(self.sbRegStart.value(), color='b')
        ax.axvline(self.sbRegEnd.value(),color='r')
        self.updateProgressBar(self.sbFrameIdx.value())
        #x_progressBar=[self.sbFrameIdx.value()]*2
        #y_progressBar=[-180,180]
        #l3,=ax.plot(x_progressBar,y_progressBar,'r-')

        # tick range
        plt.xticks(np.arange(0, self.lastFrameIdx+1, 10))
        startY, endY = ax.get_ylim()
        plt.yticks(np.arange(startY, endY+1, 30.0))

        plt.legend(loc='upper right')#show legend
        #plt.legend(handles=[l1, l2], labels=['up', 'down'],  loc='best')#show legend and change set label
        plt.grid()#show grid
        
        
        # put on gui
        if(self.canvans is None):
            self.cavans = FigureCanvas(fig)
            self.cavans.setGeometry(0,0,self.gv3.width()-20,self.gv3.height()-20)# fit size of piture to gv
            self.scene3.addWidget(self.cavans)
        else:
            self.cavans.setFig(fig)

    '''
        human control interface
    '''
    def onBind(self):
        self.btnPlay.clicked.connect(self.btnPlay_clicked)
        self.btnLoad.clicked.connect(self.btnLoad_clicked)
        self.btnClipSave.clicked.connect(self.btnClipSave_clicked)


        
        self.btnModelRoot.clicked.connect(self.btnModelRoot_clicked)
        self.btnTrack.clicked.connect(self.btnTrack_clicked)
        self.btnSaveCSV.clicked.connect(self.btnSaveCSV_clicked)
        self.btnLoadCSV.clicked.connect(self.btnLoadCSV_clicked)


        self.btnUpdateGv2.clicked.connect(self.btnUpdateGv2_clicked)
        self.btnUpdateLineChart.clicked.connect(self.btnUpdateLineChart_clicked)
        
        

        self.slider.valueChanged.connect(self.sliderChanged)
        self.sbFrameIdx.valueChanged.connect(self.sbFrameIdxChanged)

    def btnModelRoot_clicked(self):
        dir=QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir!='':
            self.lbModelRoot.setText(dir)
    def btnUpdateLineChart_clicked(self):
        self.drawLineChart()
    def btnUpdateGv2_clicked(self):
        self.drawCenters()
    
    def btnTrack_clicked(self):
        vid_name = QFileDialog.getOpenFileName(
            self, "Choose Video", self.default_path, "Video Files (*.avi *.mp4 *.mov)")[0]
        if vid_name!='':
            #model_root='D:/Trackers/pysot-master'# python_working_directory
            model_root=self.lbModelRoot.text()
            #fps='--fps {:f}'.format(self.M.getFPS())
            instruction='python ./demo.py --config {:s}/experiments/siammask_r50_l3/config.yaml --snapshot {:s}/experiments/siammask_r50_l3/model.pth --video {:s}'
            instruction=instruction.format(model_root,model_root,vid_name)
            #print(instruction)
            os.system(instruction)

    def btnSaveCSV_clicked(self):
        if not self.centers:
            # Nothing to save
            return
        if self.csvName!="":
            csv_path=self.csvName
        else:
            csv_path=os.path.join(self.default_path,'tmp.csv')
        fileName=QFileDialog.getSaveFileName(self, 'Save *.csv File', csv_path,"CSV (*.csv)")[0]

        if fileName!='':
            with open(fileName,'w') as fp:
                fp.write("frame_idx,center_x,center_y\n")
                for key,center in self.centers.items():
                    print(1)
                    fp.write(str(key)+","+str(center[0])+","+str(center[1])+"\n")
                fp.close()

    def btnLoadCSV_clicked(self):
        if self.csvName!="":
            csv_path=self.csvName
        else:
            csv_path='./output/'
        fileName = QFileDialog.getOpenFileName(
            self, "Choose CSV File", csv_path, "CSV (*.csv)")[0]


        if fileName!='':
            with open(fileName,'r') as fp:
                self.centers.clear()
                lines=fp.readlines()
                for line in lines[1:]:
                    tokens=line.split(',')
                    frame=int(tokens[0])
                    center=(float(tokens[1]),float(tokens[2]))
                    self.centers[frame]=center
                fp.close()
                #self.csvName=fileName
                self.drawCenters()
    
    def btnPlay_clicked(self):
        print('Play Clicked')
        if self.isVideoOpened:
            if self.isPlaying:
                self.btnPlay.setText('play')
                self.isPlaying=False
            else:
                #play start
                self.btnPlay.setText('pause')
                self.isPlaying=True

                #playing
                if self.sbFrameIdx.value() ==self.lastFrameIdx:
                        self.sbFrameIdx.setValue(0)
                else:
                    while self.isPlaying:
                        curIdx=self.sbFrameIdx.value()
                        self.readAndShow(curIdx)
                        
                        QApplication.processEvents()
                        if curIdx !=self.lastFrameIdx:
                            self.sbFrameIdx.setValue(curIdx+1)
                        else:
                            break

                # play over
                self.btnPlay.setText('play')
                self.isPlaying=False
        
    def sliderChanged(self,value):
        if  self.isVideoOpened and (not self.isPlaying):
            self.sbFrameIdx.setValue(value)
    def sbFrameIdxChanged(self,index):
        if  self.isVideoOpened and (not self.isPlaying):
            self.readAndShow(index)
        
    def btnClipSave_clicked(self):
        if  self.isVideoOpened and (not self.isPlaying):
            startIndex=self.sbFrameStart.value()
            endIndex=self.sbFrameEnd.value()
            self.M.clipThenSave(startIndex,endIndex)

    def btnLoad_clicked(self):
        fileName = QFileDialog.getOpenFileName(
            self, "Choose Video", self.default_path, "Video Files (*.avi *.mp4 *.mov)")

        #check file exists
        if fileName[0]!='':
            fileName=fileName[0]
            print('Try to load video from: ',fileName)
            # file is opened correctly
            self.isVideoOpened=self.M.loadVideo(fileName)
            if self.isVideoOpened:
                self.resetAll()#reset for new video, if video was previously opened
                self.setVideoInformation(self.M.getFPS(),self.M.getWidth(),self.M.getHeight())
                self.lastFrameIdx=self.M.getLength()-1 # last index = length -1
                self.sbFrameIdx.setMaximum(self.lastFrameIdx)
                self.sbFrameIdx.setSuffix('/'+str(self.lastFrameIdx))
                self.sbFrameStart.setMaximum(self.lastFrameIdx)
                self.sbFrameEnd.setMaximum(self.lastFrameIdx)
                self.sbFrameEnd.setValue(self.lastFrameIdx)
                self.sbRegStart.setMaximum(self.lastFrameIdx)
                self.sbRegEnd.setMaximum(self.lastFrameIdx)
                self.sbRegEnd.setValue(self.lastFrameIdx)
                self.slider.setMaximum(self.lastFrameIdx)

                self.videoName=fileName
                self.csvName=str(self.videoName[:-3]).replace('.','.csv')
                print('Successed!')
                
                self.readAndShow(0)# show first frame
                self.drawLineChart()
                
            else:
                print('Failed to open video, maybe it is not exist or being occupied')