from numpy import empty
from cv2 import imread,resize
import cv2

def rgbCount():
          skinRgbCnt=empty(shape=(256,256,256))
          skinRgbCnt.fill(0)
          nonSkinRgbCnt=empty(shape=(256,256,256))
          nonSkinRgbCnt.fill(0)
          totalSkinColor=0
          totalNonSkinColor=0

          totalImages=42
          for i in range(totalImages):
                    maskImage=imread(f"Mask-Image/mask_train_{i+2}.jpg")
                    realImage=imread(f"Real-Image/real_train_{i+2}.jpg")
                    if maskImage.shape[:2] != realImage.shape[:2]:
                              continue
                    
                    height,width=maskImage.shape[:2] 
                    for x in range(height):
                              for y in range(width):
                                        maskRed=maskImage[x,y,2]
                                        maskGreen=maskImage[x,y,1]
                                        maskBlue=maskImage[x,y,0]
                                        
                                        realRed=realImage[x,y,2]
                                        realGreen=realImage[x,y,1]
                                        realBlue=realImage[x,y,0]
                                        
                                        if maskRed==0 and maskGreen==0 and maskBlue==0:
                                                  nonSkinRgbCnt[realRed,realGreen,realBlue]+=1
                                                  totalNonSkinColor+=1
                                        else:
                                                  skinRgbCnt[realRed,realGreen,realBlue]+=1
                                                  totalSkinColor+=1
          print(skinRgbCnt,totalSkinColor,nonSkinRgbCnt,totalNonSkinColor)
          return skinRgbCnt,totalSkinColor,nonSkinRgbCnt,totalNonSkinColor


def writeOutput(skinRgbCnt,totalSkinColor,nonSkinRgbCnt,totalNonSkinColor):
          filename="naiveBayesOutput.txt"
          fp=open(filename,'w')
          for r in range(256):
                    for g in range(256):
                              for b in range(256):
                                        skinProb=skinRgbCnt[r,g,b]/totalSkinColor 
                                        nonSkinProb=nonSkinRgbCnt[r,g,b]/totalNonSkinColor 

                                        if nonSkinProb>0:
                                                  threshold=skinProb/nonSkinProb 
                                        else:
                                                  threshold=0
                                        fp.write(f'{threshold}\n')

def naiveBayesTrain():
          skinRgbCnt,totalSkinColor,nonSkinRgbCnt,totalNonSkinColor=rgbCount()
          writeOutput(skinRgbCnt,totalSkinColor,nonSkinRgbCnt,totalNonSkinColor)



naiveBayesTrain()