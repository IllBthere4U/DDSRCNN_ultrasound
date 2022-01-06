import numpy as np
import math
from PIL import Image


'''
PSNR = 20*log10[peak/(标准差)]
'''


def getpsnr(imagePath):
    image = Image.open(imagePath)
    image_array = np.array(image)
    image_std = np.std(image_array, ddof=1)
    return 20 * math.log(255, 10) - 10 * math.log(image_std, 10)


if __name__ == "__main__":
    psnr = getpsnr('ESPCN_256_24bit.png')
    print(psnr)