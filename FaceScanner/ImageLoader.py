import numpy
import scipy
import matplotlib.pyplot
import random
import math
import skimage.measure
import skimage.io
import io
from PIL import Image
import Face
import os


def crop(image):
    # crop image from 250x250 to 200x200
    return image[50:-50, 50:-50]

def get_images(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    #for f in files:
    #    print(f)
    return files


load_faces = get_images('faces in the wild training')
load_non_faces = get_images('non_faces')

all_faces = []

#load_faces = load_faces[:5]
#load_non_faces = load_non_faces[:5]

for file in load_faces:
    print(file)
    face = skimage.io.imread(file, as_gray=True)  # load the image as grayscale
    face = numpy.array(face)
    face = Face.Face(face, 1)
    all_faces.append(face.flatten_face())

for file in load_non_faces:
    print(file)
    face = skimage.io.imread(file, as_gray=True)  # load the image as grayscale
    face = numpy.array(face/255)
    face = Face.Face(face, 1)
    all_faces.append(face.flatten_face())

all_faces = numpy.array(all_faces)

Bias = numpy.full((len(all_faces),1), 1)
all_faces = numpy.concatenate((Bias, all_faces), axis=1)

Class_one = numpy.full((len(load_faces),1), 1)
Class_two = numpy.full((len(load_non_faces),1), 0)
Classes = numpy.concatenate((Class_one, Class_two), axis=0)

all_faces = numpy.concatenate((all_faces, Classes), axis=1)

#print(all_faces)

print(len(all_faces))

numpy.save("testfile", all_faces)




