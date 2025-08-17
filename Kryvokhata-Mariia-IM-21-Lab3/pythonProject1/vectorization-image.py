from PIL import Image, ImageDraw
from pylab import *


def vector_circuit(img):
    figure()
    contour(img, origin='image')
    axis('equal')
    show()
    contour(img, levels=[170], colors='black', origin='image')
    axis('equal')
    show()

    return


def mono(image):
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()  # pixels' values

    print('Enter mono factor')
    factor = int(input('factor:'))
    print('Processing...')
    for i in range(width):
        for j in range(height):
            r = pix[i, j][0]
            g = pix[i, j][1]
            b = pix[i, j][2]
            S = r + g + b
            if (S > (((255 + factor) // 2) * 3)):  # deciding what color the pixel resembles more: white or black
                r, g, b = 255, 255, 255
            else:
                r, g, b = 0, 0, 0
            draw.point((i, j), (r, g, b))

    plt.imshow(image)
    plt.show()
    image.save("mono-img.jpg", "JPEG")
    del draw

    return


if __name__ == '__main__':

    print("Select source image:")
    print("1 - Ideal image")
    print("2 - Real image")
    mode_1 = int(input("mode: "))

    if mode_1 == 1:
        im = array(Image.open('Elephant_perfect.jpg').convert('L'))
        image = Image.open("Elephant_perfect.jpg")
        vector_circuit(im)

    if mode_1 == 2:
        im = array(Image.open('Dniproges.jpg').convert('L'))
        image = Image.open("Dniproges.jpg")
        vector_circuit(im)

    print("Enhance image quality?")
    print("1 - Yes")
    print("2 - No")
    mode = int(input('mode:'))

    if mode == 1:
        mono(image)
        im = array(Image.open('mono-img.jpg').convert('L'))
        vector_circuit(im)

    if (mode == 2):
        sys.exit()
