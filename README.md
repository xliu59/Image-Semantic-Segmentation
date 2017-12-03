# Image Semantic Segmentaion		
 A Computer Vision course project		

Instructions:
1. To train FCN32:
./FCN32.py --save 32.pkl --enable_testing

2. To train FCN16:
a. Training using 32.pkl:  ./FCN16.pkl --load 32.pkl --save 16.pkl --enable_testing
b. Training only with pretrained vgg:  ./FCN.py --save 16.pkl --enable_testing

3. To train FCN8:
a. Training using 16.pkl:  ./FCN8.pkl --load 16.pkl --save 8.pkl --enable_testing
b. Training only with pretrained vgg:  ./FCN.py --save 8.pkl --enable_testing
