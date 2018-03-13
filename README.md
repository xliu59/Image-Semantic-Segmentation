# Image Semantic Segmentation		
 A Computer Vision course project		

Instructions:  <br />
1. To train FCN32:<br />
                                          ./FCN32.py --save 32.pkl --enable_testing<br />

2. To train FCN16:  <br />
a. Training using 32.pkl:                 ./FCN16.pkl --load 32.pkl --save 16.pkl --enable_testing<br />
b. Training only with pretrained vgg:     ./FCN.py --save 16.pkl --enable_testing<br />

3. To train FCN8:  <br />
a. Training using 16.pkl:                  ./FCN8.pkl --load 16.pkl --save 8.pkl --enable_testing<br />
b. Training only with pretrained vgg:      ./FCN.py --save 8.pkl --enable_testing<br />


# NOTE: 
1. FCN32_old.py is the previous version we trained last time.<br />
2. Changes:<br />
a. rewrite the network, the previous one cannot be extended to FCN16 or FCN8<br />
b. add normalization after totensor()<br />
c. change the image size to 224 as this is the size works best for vgg? NOT SURE!!!<br />
d. loss is saved as txt files automatially, such as 32_trainingloss.txt 16_trainingloss_16.txt.<br />
e. Change the final upsampling layer, seems to work fine!

		
# VOC dataset download:		
 http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar	
 
# VOC segmentation download(300MB, we use this):
 https://drive.google.com/open?id=1CTulFL9A-VyQ7br_4GMxJ4ZRNvym5Pwj
