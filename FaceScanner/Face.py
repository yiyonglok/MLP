from PIL import Image
import numpy


class Face:

    def __init__(self, image, modify):
        self.image = image
        if modify:
            self.crop()
            self.average_pooling()
            self.average_pooling()
            print(self.image)


    def max_pooling(self):

        # use max pooling to shrink image
        new_image = []
        for i in range(0, len(self.image), 2):
            row = []
            for j in range(0, len(self.image[i]), 2):
                row.append(max(self.image[i][j], self.image[i + 1][j], self.image[i][j + 1], self.image[i + 1][j + 1]))
            new_image.append(row)

        # print(new_image)
        # print(len(new_image), len(new_image[0]))
        self.image = numpy.array(new_image)

    def min_pooling(self):

        # use max pooling to shrink image
        new_image = []
        for i in range(0, len(self.image), 2):
            row = []
            for j in range(0, len(self.image[i]), 2):
                row.append(min(self.image[i][j], self.image[i + 1][j], self.image[i][j + 1], self.image[i + 1][j + 1]))
            new_image.append(row)

        # print(new_image)
        # print(len(new_image), len(new_image[0]))
        self.image = numpy.array(new_image)

    def average_pooling(self):

        # use max pooling to shrink image
        new_image = []
        for i in range(0, len(self.image), 2):
            row = []
            for j in range(0, len(self.image[i]), 2):
                calculate_avg = self.image[i][j] + self.image[i + 1][j] + self.image[i][j + 1] + self.image[i + 1][j + 1]
                calculate_avg = calculate_avg / 4
                row.append(calculate_avg)
            new_image.append(row)

        # print(new_image)
        # print(len(new_image), len(new_image[0]))
        self.image = numpy.array(new_image)

    def flatten_face(self):
        return numpy.array([num for sublist in self.image for num in sublist])

    def crop(self):
        # crop image from 250x250 to 200x200
        self.image = self.image[25:-25, 25:-25]

    def save_face(self, name):
        iimage = Image.fromarray(self.image * 255)
        if iimage.mode != 'RGB':
            iimage = iimage.convert('RGB')
        iimage.save(name+".jpg")

    def print_face(self):
        print(self.image)
