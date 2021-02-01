# Machine-Learning-Projects


# Cat-Dog-Human Classification

> Simple classification project with Python-Tensorflow-Keras.
> The dataset is in PetImages folder. 
> The main idea of code is finding the class and probability of given image.


# Plate Detection with contour

> The main idea is detecting plate in given image. 
> Image Processing functions and contour area are used to detect the plate.
> It isn't working good because finding contour areas is depending on plate's area. 
> If the car plate is too big or too big let's say out of the area size the plate can't be detected.
> We can see finding plate is depends on plate's area below in code:

```
  if area > (image.shape[0] + image.shape[1]) * 3 and (w > h) and center_y > height / 2:
```

# Plate Detection with YOLO

> YOLO is used to detect plate. It's working really good. 
> The Cfg file of YOLO is loaded but model weights couldn't loaded beacuse It is more than 100 MB.
> You can find the weights via link https://pjreddie.com/darknet/yolo/
