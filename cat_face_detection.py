import cv2
import sys


imagepath= sys.argv[1]
cascadepath = sys.argv[2]

#create tha haar cascade
#CascadeClassifier(xmlFILE)


cat_ext_Cascade = cv2.CascadeClassifier(cascadepath)

#Read the image

image = cv2.imread(imagepath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# DETECT faces in the image
#detectMultiscale --> use t detect objects
# first argument --> grayscale image
#second argument --> scaleFactor 1.1
#third arument --> minNeighbors --> it defines how many
#objects are detected near the current one
#fourth arguments --> minSize
#fifth arument --> flags

cat_faces = cat_ext_Cascade.detectMultiScale(gray ,
scaleFactor=1.5 ,minNeighbors = 5,
minSize=(30,30))


#draw a rectangle around faces

for(x,y,w,h) in cat_faces:
    cv2.rectangle(image, (x,y) , (x+w , y+h), (0,225,0),2)
	
	
cv2.imshow("faces Found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



