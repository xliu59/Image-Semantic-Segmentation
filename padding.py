from PIL import Image

## to read
# img = Image.open('xxx.yy')

def padding(img, nt, mt):
    new_img = Image.new("RGB", (nt, mt), "black")
    # alternatively, Image.new("RGB", (nt, mt), "white"), whichever applies
    new_img.paste(img, (50, 50)) # (50,50) is the left corner.. which can be changed according to your need
    ## to save
    # new_img.save("whatever.extension")
    ## to show
    # new_img.show()
    new_img.show()






