# Seamless Image Cloning with Semantic Segmentation

## Contributors
* Jiteng Mu
* Tianshi Liao
* Xiaoxiao Liu
* Yuxiang Gao

## Idea
We propose an automated system to crop out certain objects, especially human, in a given photo and fuse the object into a new background to form a new photo. The first part of our work will focus on building and train a semantic segmentation model that features in detecting and recognizing the human figure in a photo, and cropping out according to its outline. For this part, our supervised semantic segmentation algorithm will be based on Fully Convolutional Networks (FCN). The second part of our focus of work, also the FUN part, will be stitching the cropped part into a new background image and try to create a new scene. Poisson Image Editing will be the main technique for this part, while some other blending methods might also be used, e.g. Feathering, The Laplacian Pyramid, Alpha Blending. This is useful in multiple photo editing scenarios, especially in portrait photo editing. An easy inverse of this method can be used to erase the unwanted objects from a photo.

## Related CV topics
* Fully Convolutional Networks for Semantic Segmentation
* Poisson Image Editing

## Use Case	
With semantic segmentation, we will be able to separate human portrait from image and substitute the background easily. A simple use case would be allowing users to choose new background scene by preferences. Also, we can do portrait editing since we have already detected where the people are in an image. For instance, we will be able to adjust color, illumination of the object of interest without changing the rest of the background, which is sort of similar to what smart phones cameras can do nowadays. We can also use an inverse version of this function to eliminate undesired portions of an image.

## To view our result
We included all of our exploration methods, model-building steps and results in a finalized paper: [Seamless Image Cloning with Semantic Segmentation](./Seamless%20Image%20Cloning%20with%20Semantic%20Segmentation.pdf)

## To train our model 

### Instructions:  <br />
1. To train FCN32:<br />
                                          ./FCN32.py --save 32.pkl --enable_testing<br />
2. To train FCN16:  <br />
a. Training using 32.pkl:                 ./FCN16.pkl --load 32.pkl --save 16.pkl --enable_testing<br />
b. Training only with pretrained vgg:     ./FCN.py --save 16.pkl --enable_testing<br />

3. To train FCN8:  <br />
a. Training using 16.pkl:                  ./FCN8.pkl --load 16.pkl --save 8.pkl --enable_testing<br />
b. Training only with pretrained vgg:      ./FCN.py --save 8.pkl --enable_testing<br />

### NOTE: 
1. FCN32_old.py is the previous version we trained last time.<br />
2. Changes:<br />
a. rewrite the network, the previous one cannot be extended to FCN16 or FCN8<br />
b. add normalization after totensor()<br />
c. change the image size to 224 as this is the size works best for vgg? NOT SURE!!!<br />
d. loss is saved as txt files automatially, such as 32_trainingloss.txt 16_trainingloss_16.txt.<br />
e. Change the final upsampling layer, seems to work fine!
		
### VOC dataset download:		
 http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar	
 
### VOC segmentation download(300MB, we use this):
 https://drive.google.com/open?id=1CTulFL9A-VyQ7br_4GMxJ4ZRNvym5Pwj
