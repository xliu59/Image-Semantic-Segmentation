# Image Semantic Segmentaion		
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
