from numpy import zeros,empty
from cv2 import imread,imwrite



def naiveBayesTest():
    input_img="goat.jpg"
    output_img="result.jpg"
    trained_value=zeros(shape=(256,256,256))
    new_img=imread(input_img)


    fp=open('naiveBayesOutput.txt',"r")   


    for i in range(256):
        for j in range(256):
            for k in range(256):
                val = fp.readline()
                trained_value[i][j][k] = float(val) 


    height,width,_=new_img.shape
    T=0.5

    for x in range(height):
        for y in range(width):
            red=new_img[x,y,2]
            green = new_img[x,y,1]
            blue=new_img[x,y,0]

            if (trained_value[red,green,blue] <= T):
                new_img[x,y,0]=0
                new_img[x,y,1]=0
                new_img[x,y,2]=0
    fp.close()
    imwrite(output_img,new_img)

naiveBayesTest()
          