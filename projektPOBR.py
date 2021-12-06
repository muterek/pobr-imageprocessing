# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(3500)

def oneColorRegions(inputImage, lightColor, darkColor):
    h, w, c = inputImage.shape
    oneColorImage = np.zeros((h,w,c), np.uint8)
    for x in range(h):
        for y in range(w):
                if inputImage[x,y,0]>=darkColor[0] and inputImage[x,y,0]<=lightColor[0] and inputImage[x,y,1]>=darkColor[1] and inputImage[x,y,1]<=lightColor[1] and inputImage[x,y,2]>=darkColor[2] and inputImage[x,y,2]<=lightColor[2]:
                    oneColorImage[x, y, 0] = 255
                    oneColorImage[x, y, 1] = 255
                    oneColorImage[x, y, 2] = 255
                else:
                    oneColorImage[x, y, 0] = 0
                    oneColorImage[x, y, 1] = 0
                    oneColorImage[x, y, 2] = 0
    return oneColorImage

def imgErosion(inputImage):
    
    h, w, c = inputImage.shape
    imageErosion = np.zeros((h,w,c), np.uint8)
    for x in range(2,h-1):
        for y in range(2,w-1):
            if inputImage[x, y, 0] == 0 and inputImage[x-1, y, 0] == 0 and inputImage[x+1, y, 0] == 0 and inputImage[x, y-1, 0] == 0 and inputImage[x, y+1, 0] == 0 and inputImage[x-1, y, 0] == 0 and inputImage[x-1, y-1, 0] == 0 and inputImage[x-1, y+1, 0] == 0 and inputImage[x+1, y-1, 0] == 0 and inputImage[x+1, y+1, 0] == 0:
                imageErosion[x,y,0]=0
                imageErosion[x,y,1]=0
                imageErosion[x,y,2]=0
            elif inputImage[x, y, 0] == 255 and inputImage[x-1, y, 0] == 255 and inputImage[x+1, y, 0] == 255 and inputImage[x, y-1, 0] == 255 and inputImage[x, y+1, 0] == 255 and inputImage[x-1, y-1, 0] == 255 and inputImage[x-1, y+1, 0] == 255 and inputImage[x+1, y-1, 0] == 255 and inputImage[x+1, y+1, 0] == 255:
                imageErosion[x,y,0]=255
                imageErosion[x,y,1]=255
                imageErosion[x,y,2]=255
            elif inputImage[x, y, 0] == 255 and (inputImage[x-1, y, 0] == 0 or inputImage[x+1, y, 0] == 0 or inputImage[x, y-1, 0] == 0 or inputImage[x, y+1, 0] == 0 or inputImage[x-1, y-1, 0] == 0 or inputImage[x-1, y+1, 0] == 0 or inputImage[x+1, y-1, 0] == 0 or inputImage[x+1, y+1, 0] == 0):
                imageErosion[x,y,0]=0
                imageErosion[x,y,1]=0
                imageErosion[x,y,2]=0
            elif inputImage[x, y, 0] == 0 and (inputImage[x-1, y, 0] == 255 or inputImage[x+1, y, 0] == 255 or inputImage[x, y-1, 0] == 255 or inputImage[x, y+1, 0] == 255 or inputImage[x-1, y, 0] == 255 or inputImage[x-1, y-1, 0] == 0 or inputImage[x-1, y+1, 0] == 0 or inputImage[x+1, y-1, 0] == 0 or inputImage[x+1, y+1, 0] == 0):
                imageErosion[x,y,0]=0
                imageErosion[x,y,1]=0
                imageErosion[x,y,2]=0
            else:
                imageErosion[x,y,0]=100
                imageErosion[x,y,1]=100
                imageErosion[x,y,2]=100
                
    return imageErosion

def imgDilation(inputImage):
    
    h, w, c = inputImage.shape
    imageDilation = np.zeros((h,w,c), np.uint8)
    for x in range(2,h-1):
        for y in range(2,w-1):
            if inputImage[x, y, 0] == 0 and inputImage[x-1, y, 0] == 0 and inputImage[x+1, y, 0] == 0 and inputImage[x, y-1, 0] == 0 and inputImage[x, y+1, 0] == 0 and inputImage[x-1, y, 0] == 0 and inputImage[x-1, y-1, 0] == 0 and inputImage[x-1, y+1, 0] == 0 and inputImage[x+1, y-1, 0] == 0 and inputImage[x+1, y+1, 0] == 0:
                imageDilation[x,y,0]=0
                imageDilation[x,y,1]=0
                imageDilation[x,y,2]=0
            elif inputImage[x, y, 0] == 255 and inputImage[x-1, y, 0] == 255 and inputImage[x+1, y, 0] == 255 and inputImage[x, y-1, 0] == 255 and inputImage[x, y+1, 0] == 255 and inputImage[x-1, y-1, 0] == 255 and inputImage[x-1, y+1, 0] == 255 and inputImage[x+1, y-1, 0] == 255 and inputImage[x+1, y+1, 0] == 255:
                imageDilation[x,y,0]=255
                imageDilation[x,y,1]=255
                imageDilation[x,y,2]=255
            elif inputImage[x, y, 0] == 0 and (inputImage[x-1, y, 0] == 255 or inputImage[x+1, y, 0] == 255 or inputImage[x, y-1, 0] == 255 or inputImage[x, y+1, 0] == 255 or inputImage[x-1, y-1, 0] == 255 or inputImage[x-1, y+1, 0] == 255 or inputImage[x+1, y-1, 0] == 255 or inputImage[x+1, y+1, 0] == 255):
                imageDilation[x,y,0]=255
                imageDilation[x,y,1]=255
                imageDilation[x,y,2]=255
            else:
                imageDilation[x,y,0]=255
                imageDilation[x,y,1]=255
                imageDilation[x,y,2]=255
                
    return imageDilation

def imageResBnW(inputImage):
    h,w,c = inputImage.shape
    imageResBnW = np.zeros((h,w,c), np.uint8)
    for x in range(h):
        for y in range(w):
                if inputImage[x,y,0]>0 or inputImage[x,y,1]>0 or inputImage[x,y,2]>0:
                    imageResBnW[x, y, 0] = 255
                    imageResBnW[x, y, 1] = 255
                    imageResBnW[x, y, 2] = 255
                else:
                    imageResBnW[x, y, 0] = 0
                    imageResBnW[x, y, 1] = 0
                    imageResBnW[x, y, 2] = 0
    return imageResBnW
    
def colorObj(image, x, y, newColor):
    obj_list = list()
    h, w, c = image.shape
    obj_list.append([y,x])
    image[y,x,:]=newColor
    n = 0
    while n+1<=len(obj_list):
        x = obj_list[n][1]
        y = obj_list[n][0]
        for x1 in [-1, 0, 1]:
            for y1 in [-1, 0, 1]:
                if y+y1<h and y+y1>=0 and x+x1>=0 and x+x1<w:
                    if image[y+y1,x+x1,0]==255 and not [y+y1,x+x1] in obj_list:
                        obj_list.append([y+y1,x+x1])
                        image[y+y1,x+x1,:]=newColor
        n=n+1
    
def listOfOneColorObj(imageResBnW):
    h,w,c = imageResBnW.shape
    oldColor = np.array([255, 255, 255], np.uint8)
    newColor = np.array([50,205,50], np.uint8)
    single = list()
    allObj = list()
    
    for x in range(h-1):
        for y in range(w-1):
            
            if (np.array_equal(imageResBnW[x,y,:],oldColor)==True):
                colorObj(imageResBnW, y, x, newColor)
                single = list()
                for xg in range(h-1):
                    for yg in range(w-1):
                        flag = False
                        for o in allObj:
                            if ([xg,yg] in o): flag = True
                        if flag == False:
                            if (np.array_equal(imageResBnW[xg,yg,:],newColor)==True):
                                single.append([xg,yg])
                allObj.append(single)
    return allObj

def m(image, p, q):
    h,w,c = image.shape
    m = 0.0
    for x in range(h):
        for y in range(w):
            m = m + ((x)**p)*((y)**q)*image[x,y,0]/255
    return m

def findMoments(imageBnW, listOfObjects):
    h,w,c = imageBnW.shape
    
    M1 = []
    M2 = []
#    M3 = []
#    M4 = []
#    M5 = []
#    M6 = []
    M7 = []
#    M8 = []
#    M9 = []
#    M10 = []
    
    for x in range(len(listOfObjects)):
        m00 = 0.0
        m01 = 0 
        m10 = 0 
        m11 = 0 
        m02 = 0 
        m20 = 0
        M11 = 0
        M20 = 0 
        M02 = 0
        hImg = np.zeros((h,w,c), np.uint8)
        for i in range(h):
            for j in range(w):
                flag = False
                for o in listOfObjects[x]:
                    if (np.array_equal([i,j],o)): flag = True
                if flag == True:
                    hImg[i,j,0] = 255
                    hImg[i,j,1] = 255
                    hImg[i,j,2] = 255
                else:
                    hImg[i,j,0] = 0
                    hImg[i,j,1] = 0
                    hImg[i,j,2] = 0
#        plt.figure(figsize=(20,10))
#        plt.imshow(hImg)
#        plt.title("x = " + str(x))

        m00 = m(hImg, 0, 0)
        m01 = m(hImg, 0, 1)
        m10 = m(hImg, 1, 0)
        m11 = m(hImg, 1, 1)
        m02 = m(hImg, 0, 2)
        m20 = m(hImg, 2, 0)
#        m21 = m(hImg, 2, 1)
#        m12 = m(hImg, 1, 2)
#        m30 = m(hImg, 3, 0)
#        m03 = m(hImg, 0, 3)
        
        # centrum obrazu
#        it = m10/m00
#        jt = m01/m00
        
#        M00 = m00
#        M01 = m01 - (m01/m00)*m00
#        M10 = m10 - (m10/m00)*m00
        M11 = m11 - m10*m01/m00
        M20 = m20 - m10**2/m00
        M02 = m02 - m01**2/m00
#        M21 = m21 - 2*m11*it - m20*jt + 2*m01*(it)**2
#        M12 = m12 - 2*m11*jt - m02*it + 2*m10*(jt)**2
#        M30 = m30 - 3*m20*it + 2*m10*(it)**2
#        M03 = m03 - 3*m02*jt + 2*m01*(jt)**2
        
        M1.append( (M20 + M02) / m00**2 )
        M2.append( ( (M20 - M02)**2 + M11**2) / m00**4)
#        M3.append( ( (M30 - 3 * M12)**2 + (3 * M21 - M03)**2) / m00**5)
#        M4.append( ( (M30 + M12)**2 + (M21 + M03)**2) / m00**5)
#        M5.append( ( (M30 - 3 * M12)*(M30 + M12)*((M30 + M12)**2 - 3 * (M21 + M03)**2) + (3*M21 - M03)*(M21 + M03)*(3*(M30 + M12)**2 - (M21 + M03)**2)) / m00**10)
#        M6.append( ( (M20 - M02)*((M30 + M12)**2 - (M21 + M03)**2) + 4 * M11 * (M30 + M12)*(M21+M03)) / m00**7)
        M7.append( (M20*M02 - M11**2) / m00**4 )
#        M8.append( (M30*M12 + M21*M03 - M12**2 - M21**2) / m00**5)
#        M9.append( (M20 * (M21*M03 - M12**2) + M02*(M03*M12 - M21**2) - M11*(M30*M03 - M21*M12)) / m00**7)
#        M10.append( ((M30*M03 - M12*M21)**2 - 4 * (M30*M12 - M21**2)*(M03*M21 - M12)) / m00**10)
        
    return M1, M2, M7 #M3, M4, M5, M6, M7, M8, M9

def findSidePixels(logoObj):
    
    sidePixels = list()
    
    for x in range(len(logoObj)):
        single = list()
        xmin = 1000
        xmax = 0
        ymin = 1000
        ymax = 0
        upLeft = [0,0]
        upRight = [0,0]
        downLeft = [0,0]
        downRight = [0,0]
        for y in range((len(logoObj[x]))-1):
            if logoObj[x][y][0] < xmin: xmin = logoObj[x][y][0]
            if logoObj[x][y][0] > xmax: xmax = logoObj[x][y][0]
            if logoObj[x][y][1] < ymin: ymin = logoObj[x][y][1]
            if logoObj[x][y][1] > ymax: ymax = logoObj[x][y][1]
            
        upLeft = [5*xmin, 5*ymin]
        upRight = [5*xmin, 5*ymax]
        downLeft = [5*xmax, 5*ymin]
        downRight = [5*xmax, 5*ymax]
        
        single.append(upLeft)
        single.append(upRight)
        single.append(downLeft)
        single.append(downRight)
        
        sidePixels.append(single)
    
    return sidePixels    
        
#%% ----------------- Show input image -----------------------
plt.figure()
image = cv2.imread('pepco2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape
#hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
image.shape
plt.figure()
plt.imshow(image)
plt.title("Original image")

#%% ----------------- Blue - light and dark --------------------
light_blue = np.array([50,150,255])
dark_blue = np.array([0,0,100])
low_b = np.full((10, 10, 3), light_blue, dtype=np.uint8)/255.0
up_b = np.full((10, 10, 3), dark_blue, dtype=np.uint8)/255.0
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(low_b)
plt.title("Light blue")
plt.subplot(1, 2, 2)
plt.imshow(up_b)
plt.title("Dark blue")
plt.show()

#%% ----------------- Blue regions in image ---------------------

imageBlue = oneColorRegions(image, light_blue, dark_blue)
plt.figure(figsize=(15,5))
plt.imshow(imageBlue)
plt.title("Blue areas")

#%% ----------------- Blue - erosion and dilation --------------------------
imageErosionBlue = imgErosion(imageBlue)
imageDilationBlue = imgDilation(imageErosionBlue)
plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.imshow(imageErosionBlue)
plt.title("Erosion - blue")
plt.subplot(1, 2, 2)
plt.imshow(imageDilationBlue)
plt.title("Dilation - blue")
plt.show()

#%% ----------------- Flood fill algorithm - blue ------------------ 

imageResBlue = cv2.resize(imageDilationBlue, (round(w/5), round(h/5)))
plt.figure()
plt.imshow(imageResBlue)
plt.title("Resized image - blue")

imageResBnWBlue = imageResBnW(imageResBlue)         
plt.figure(figsize=(15,5))
plt.imshow(imageResBnWBlue)
plt.title("Resized blue image - black and white")

allObjBlue = listOfOneColorObj(imageResBnWBlue)

plt.figure(figsize=(15,10))
plt.imshow(imageResBnWBlue)
plt.title("Blue image after floodfill")

#%% ----------------- Momenty - blue ---------------------------

imageResBnWBlue = imageResBnW(imageResBlue)         
plt.figure(figsize=(15,5))
plt.imshow(imageResBnWBlue)

#M1b, M2b, M3b, M4b, M5b, M6b, M7b, M8b, M9b = findMoments(imageResBnWBlue, allObjBlue)
M1b, M2b, M7b = findMoments(imageResBnWBlue, allObjBlue)

h,w,c = imageResBnWBlue.shape
hImg = np.zeros((h,w,c), np.uint8)
for x in range(len(M1b)-1): 
    if (M2b[x]>0.035 and M2b[x]<0.21 and M7b[x]>0.015 and M7b[x]<0.04):
        for i in range(h):
            for j in range(w):
                flag = False
                for o in allObjBlue[x]:
                    if (np.array_equal([i,j],o)): flag = True
                if flag == True:
                    hImg[i,j,0] = 255
                    hImg[i,j,1] = 255
                    hImg[i,j,2] = 255

plt.figure(figsize=(15,10))
plt.imshow(hImg)
plt.title("Blue logo")

blueLogoObj = listOfOneColorObj(hImg)
sidePixels = findSidePixels(blueLogoObj)

croppedImages = list()

for x in range(len(sidePixels)):
    crop_img = image[sidePixels[x][0][0]:sidePixels[x][3][0], sidePixels[x][0][1]:sidePixels[x][3][1], :]
    plt.figure()
    plt.imshow(crop_img)
    plt.title("Cropped image " + str(x))
    croppedImages.append(crop_img)

#%% ------------------- Yellow - light and dark ----------------------
light_yellow = np.array([255,255,80])
dark_yellow = np.array([150,150,0])
low_y = np.full((10, 10, 3), light_yellow, dtype=np.uint8)/255.0
up_y = np.full((10, 10, 3), dark_yellow, dtype=np.uint8)/255.0
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(low_y)
plt.title("Light yellow")
plt.subplot(1, 2, 2)
plt.imshow(up_y)
plt.title("Dark yellow")
plt.show()

#%% --------------------------------- Yellow regions in image -----------------------------
imagesYellow = list()
for x in range(len(croppedImages)):
    imageYellow = oneColorRegions(croppedImages[x], light_yellow, dark_yellow)
    imagesYellow.append(imageYellow)
    
#%% ----------------- Yellow - erosion and dilation --------------------------
for x in range(len(imagesYellow)):
    imageYellow = imagesYellow[x]
    imageErosionYellow = imgErosion(imageYellow)
    imageDilationYellow = imgDilation(imageErosionYellow)
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.imshow(imageErosionYellow)
    plt.title("Erosion - yellow")
    plt.subplot(1, 2, 2)
    plt.imshow(imageDilationYellow)
    plt.title("Dilation - yellow")
    plt.show()
    imagesYellow[x] = imageDilationYellow

#%% ------------------------ Floodfill algorithm --------------------------------
   
allObjYellow = list()
for x in range(len(imagesYellow)):
    objYellow = listOfOneColorObj(imagesYellow[x])
    allObjYellow.append(objYellow)
    
for x in range(len(imagesYellow)):
    imagesYellow[x] = imageResBnW(imagesYellow[x]) 
    plt.figure(figsize=(15,5))
    plt.imshow(imagesYellow[x])
    plt.title("Yellow areas BnW " + str(x))

#%% ---------------------------------Moments - yellow -----------------------------------

momentsYellow = list()
for x in range(len(imagesYellow)):
    single = list()
#    M1y, M2y, M3y, M4y, M5y, M6y, M7y, M8y, M9y = findMoments(imagesYellow[x], allObjYellow[x])
    M1y, M2y, M7y = findMoments(imagesYellow[x], allObjYellow[x])

    single.append(M1y)
    single.append(M2y)
#    single.append(M3y)
#    single.append(M4y)
#    single.append(M5y)
#    single.append(M6y)
    single.append(M7y)
#    single.append(M8y)
#    single.append(M9y)
    
    momentsYellow.append(single)

#%% ---------------------------- Red - light and dark --------------------------------
light_red = np.array([255,128,128])
dark_red = np.array([150,0,0])
low_r = np.full((10, 10, 3), light_red, dtype=np.uint8)/255.0
up_r = np.full((10, 10, 3), dark_red, dtype=np.uint8)/255.0
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(low_r)
plt.title("Light red")
plt.subplot(1, 2, 2)
plt.imshow(up_r)
plt.title("Dark red")
plt.show()

#%% --------------------------------- Red regions in image -----------------------------

imagesRed = list()
for x in range(len(croppedImages)):
    imageRed = oneColorRegions(croppedImages[x], light_red, dark_red)
    imagesRed.append(imageRed)
    
#%% ----------------- Red - erosion and dilation --------------------------
for x in range(len(imagesRed)):
    imageRed = imagesRed[x]
    imageErosionRed = imgErosion(imageRed)
    imageDilationRed = imgDilation(imageErosionRed)
#    imageDilationRed = imgDilation(imageDilationRed)
#    imageDilationRed = imgDilation(imageDilationRed)
#    imageDilationRed = imgDilation(imageDilationRed)
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.imshow(imageErosionRed)
    plt.title("Erosion - red" + str(x))
    plt.subplot(1, 2, 2)
    plt.imshow(imageDilationRed)
    plt.title("Dilation - red" + str(x))
    plt.show()
    imagesRed[x] = imageDilationRed

#%% ------------------------ Floodfill algorithm --------------------------------

allObjRed = list()
for x in range(len(imagesRed)):
    objRed = listOfOneColorObj(imagesRed[x])
    allObjRed.append(objRed)
    
for x in range(len(imagesRed)):
    imagesRed[x] = imageResBnW(imagesRed[x]) 
    plt.figure(figsize=(15,5))
    plt.imshow(imagesRed[x])
    plt.title("Red areas BnW " + str(x))
    
#%% ---------------------------------Moments - red -----------------------------------

momentsRed = list()
for x in range(len(imagesRed)):
    single = list()
#    M1r, M2r, M3r, M4r, M5r, M6r, M7r, M8r, M9r = findMoments(imagesRed[x], allObjRed[x])
    M1r, M2r, M7r = findMoments(imagesRed[x], allObjRed[x])

    single.append(M1r)
    single.append(M2r)
#    single.append(M3r)
#    single.append(M4r)
#    single.append(M5r)
#    single.append(M6r)
    single.append(M7r)
#    single.append(M8r)
#    single.append(M9r)
    
    momentsRed.append(single)

#%% --------------------- White - light and dark -------------------------
light_white = np.array([255,255,255])
dark_white = np.array([160,160,180])
low_w = np.full((10, 10, 3), light_white, dtype=np.uint8)/255.0
up_w = np.full((10, 10, 3), dark_white, dtype=np.uint8)/255.0
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(low_w)
plt.title("Light white")
plt.subplot(1, 2, 2)
plt.imshow(up_w)
plt.title("Dark white")
plt.show()

#%% --------------------------------- White regions in image -----------------------------
imagesWhite = list()
for x in range(len(croppedImages)):
    imageWhite = oneColorRegions(croppedImages[x], light_white, dark_white)
    imagesWhite.append(imageWhite)
    
#%% ----------------- Red - erosion and dilation --------------------------
for x in range(len(imagesWhite)):
    imageWhite = imagesWhite[x]
    imageErosionWhite = imgErosion(imageWhite)
    imageDilationWhite = imgDilation(imageErosionWhite)
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.imshow(imageErosionWhite)
    plt.title("Erosion - white" + str(x))
    plt.subplot(1, 2, 2)
    plt.imshow(imageDilationWhite)
    plt.title("Dilation - white" + str(x))
    plt.show()
    imagesWhite[x] = imageDilationWhite

#%% ------------------------ Floodfill algorithm --------------------------------

imagesWhite = list()
for x in range(len(croppedImages)):
    imageWhite = oneColorRegions(croppedImages[x], light_white, dark_white)
    imagesWhite.append(imageWhite)
   
allObjWhite = list()
for x in range(len(imagesWhite)):
    objWhite = listOfOneColorObj(imagesWhite[x])
    allObjWhite.append(objWhite)
    
for x in range(len(imagesWhite)):
    imagesWhite[x] = imageResBnW(imagesWhite[x]) 
    plt.figure(figsize=(15,5))
    plt.imshow(imagesWhite[x])
    plt.title("White areas BnW " + str(x))
    
#%% --------------------------------- Moments - white -----------------------------------

momentsWhite = list()
for x in range(len(imagesWhite)):
    single = list()
#    M1w, M2w, M3w, M4w, M5w, M6w, M7w, M8w, M9w = findMoments(imagesWhite[x], allObjWhite[x])
    M1w, M2w, M7w = findMoments(imagesWhite[x], allObjWhite[x])

    single.append(M1w)
    single.append(M2w)
#    single.append(M3w)
#    single.append(M4w)
#    single.append(M5w)
#    single.append(M6w)
    single.append(M7w)
#    single.append(M8w)
#    single.append(M9w)
    
    momentsWhite.append(single)
    
#%% ------------------------------- Logo segmentation ------------------------------
findLogo = list()
for x in range(len(croppedImages)):  
    fr = False
    fyu = False
    fyd = False
    fp = False
    fe = False
    fc = False
    fo = False
    
    literaP = list()
    literaE = list()
    literaC = list()
    literaO = list()
    
    lewyO = list()
    lewyC = list()
    lewyE = list()
    lewyP = list()
    
    for o in range(len(momentsRed[x][0])):
        if (momentsRed[x][0][o] >0.36 and momentsRed[x][0][o]<0.375 
            and momentsRed[x][1][o]> 0.014 and momentsRed[x][1][o] < 0.065 
            and momentsRed[x][2][o]> 0.018 and momentsRed[x][2][o]<0.029): 
            print("Jest czerwone logo PEPCO")
            fr = True
            break
        else: print("Nie znaleziono koloru czerwonego w logo PEPCO")
    for o in range(len(momentsYellow[x][0])):
        if (momentsYellow[x][0][o] >0.62 and momentsYellow[x][0][o]<0.72 
            and momentsYellow[x][1][o]> 0.19 and momentsYellow[x][1][o] < 0.42 
            and momentsYellow[x][2][o]> 0.018 and momentsYellow[x][2][o]<0.024): 
            print("Jest żółty górny pasek logo PEPCO")
            fyu = True
        if (momentsYellow[x][0][o] >0.48 and momentsYellow[x][0][o]<0.71 
            and momentsYellow[x][1][o]> 0.1 and momentsYellow[x][1][o] < 0.4 
            and momentsYellow[x][2][o]> 0.012 and momentsYellow[x][2][o]<0.023): 
            print("Jest żółty dolny pasek logo PEPCO")
            fyd = True
        else: print("Nie znaleziono żółtego dolnego paska w logo PEPCO")
    
    for o in range(len(momentsWhite[x][0])):
                    
        if (momentsWhite[x][0][o] >0.24 and momentsWhite[x][0][o]<0.276 
            and momentsWhite[x][1][o]> 0.0038 and momentsWhite[x][1][o] < 0.0094 
            and momentsWhite[x][2][o]> 0.011 and momentsWhite[x][2][o]<0.014): 
            print("Jest litera P w logo PEPCO")
            fp = True
            literaP.append(o)
            lewyPikselP = min(allObjWhite[x][o])[1]
            lewyP.append(lewyPikselP)
        else: print("Nie znaleziono litery P w logo PEPCO")
    
        if (momentsWhite[x][0][o] >0.2543 and momentsWhite[x][0][o]<0.288 
            and momentsWhite[x][1][o]> 0.00918 and momentsWhite[x][1][o] < 0.02459 
            and momentsWhite[x][2][o]> 0.01326 and momentsWhite[x][2][o]<0.014493): 
            print("Jest litera E w logo PEPCO")
            fe = True
            literaE.append(o)
            lewyPikselE = min(allObjWhite[x][o])[1]
            lewyE.append(lewyPikselE)
        else: print("Nie znaleziono litery E w logo PEPCO")
    
        if (momentsWhite[x][0][o] >0.32 and momentsWhite[x][0][o]<0.39 
            and momentsWhite[x][1][o]> 0.016 and momentsWhite[x][1][o] < 0.05 
            and momentsWhite[x][2][o]> 0.021 and momentsWhite[x][2][o]<0.025): 
            print("Jest litera C w logo PEPCO")
            fc = True
            literaC.append(o)
            lewyPikselC = min(allObjWhite[x][o])[1]
            lewyC.append(lewyPikselC)
        else: print("Nie znaleziono litery C w logo PEPCO")
    
        if (momentsWhite[x][0][o] >0.28 and momentsWhite[x][0][o]<0.33 
            and momentsWhite[x][1][o]> 0.00071 and momentsWhite[x][1][o] < 0.0077 
            and momentsWhite[x][2][o]> 0.019 and momentsWhite[x][2][o]<0.025): 
            print("Jest litera O w logo PEPCO")
            fo = True
            literaO.append(o)
            lewyPikselO = min(allObjWhite[x][o])[1]
            lewyO.append(lewyPikselO)
        else: print("Nie znaleziono litery O w logo PEPCO")
              
    pkt = 0
    if lewyO == []: lewyO = [0]
    if lewyE == []: lewyE = [0]
    if lewyP == []: lewyP = [0]
    if lewyC == []: lewyC = [0]
    litO = max(lewyO)
    if litO>max(lewyP + lewyE + lewyC): pkt=pkt+1
    pierwszeP = min(lewyP)
    if pierwszeP<min(lewyO + lewyE + lewyC): pkt=pkt+1
    if max(lewyC)<litO and max(lewyC)>max(lewyP + lewyE): pkt=pkt+1
    if max(lewyP)<litO and max(lewyP)<max(lewyC) and max(lewyP)>min(lewyP + lewyE): pkt = pkt+1
    if min(lewyE)<litO and min(lewyE)<max(lewyC + lewyP) and min(lewyE)>pierwszeP: pkt=pkt+1
    
    if (fr == True and fyu == True and fyd == True and fp == True and 
        fe == True and fc == True and fo == True and pkt >= 4):
        print("Znaleziono logo PEPCO")
        findLogo.append(x)
        plt.figure(figsize=(15,10))
        plt.imshow(croppedImages[x])
        plt.title("Znalezione logo " + str(x))
    else: print("Nie znaleziono logo PEPCO")
    
#%% ------------------------ Find logo ---------------------------------------

for x in range(len(findLogo)):
    sidePixels = findSidePixels(blueLogoObj)
    start_point = (sidePixels[x][0][1],sidePixels[x][0][0])
    end_point = (sidePixels[x][3][1],sidePixels[x][3][0])
    color = (255, 255, 255) 
    thickness = 2
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    plt.figure(figsize=(15,10))
    plt.imshow(image)
    plt.title("Znalezione logo " + str(x))


            
       
        

        