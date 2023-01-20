
import numpy
import skimage.io
import Face
import time

with open('neuron_weights.npy', 'rb') as opened_file:
    neuron_weights = numpy.load(opened_file)

with open('output_weights.npy', 'rb') as opened_file:
    output_weights = numpy.load(opened_file)

#print("neuron_weights:",neuron_weights)
#print("output_weights:",output_weights)

#Image Scanner
testing_picture = skimage.io.imread('testing/test images/1442313353nasa-small.jpg', as_gray=True)

#print("picture:", testing_picture)

#print("testing Picture:", len(testing_picture), "x", len(testing_picture[0]))

test_data = []
test_data_images = []

for row in range(len(testing_picture) - 50):
    for column in range(len(testing_picture[row]) - 50):
        #print(column)
        test_area = testing_picture[row:row+50, column:column+50]
        test_area = numpy.array(test_area)
        #print(test_area)

        new_image = Face.Face(test_area, 0)
        test_data_images.append(new_image)

        #print(len(test_area), "x", len(test_area[1]))
        #print(test_area)
        test_area = [num for sublist in test_area for num in sublist]
        #print(len(test_area))
        test_data.append(test_area)

Bias = numpy.full((len(test_data),1), 1)
test_data = numpy.concatenate((Bias, test_data), axis=1)

#Populate Classifiers for Image Sampling
test_data_classifiers = numpy.zeros(len(test_data))

test_data_classifiers[31303] = 1
test_data_classifiers[35514] = 1
test_data_classifiers[23788] = 1
test_data_classifiers[26799] = 1
test_data_classifiers[28039] = 1
test_data_classifiers[26344] = 1
test_data_classifiers[27000] = 1
test_data_classifiers[27062] = 1

test_data_classifiers[63733] = 1
test_data_classifiers[58497] = 1
test_data_classifiers[51498] = 1
test_data_classifiers[62763] = 1
test_data_classifiers[62822] = 1
test_data_classifiers[56393] = 1
test_data_classifiers[61176] = 1
test_data_classifiers[52976] = 1
test_data_classifiers[46570] = 1

test_data_classifiers[95034] = 1
test_data_classifiers[95136] = 1
test_data_classifiers[96414] = 1
test_data_classifiers[90603] = 1
test_data_classifiers[88946] = 1
test_data_classifiers[90825] = 1

#print(len(test_data_images))
#print(test_data_images[0])
#test_data_images[0].print_face()
#test_data_images[0].save_non_face("sampleFace")

#print("test data row 0:", len(test_data[0]))
#print("test data row 1:", len(test_data[1]))
#print("test data length:", len(test_data))

#chuck data through MLP
NEURONS = 50
alpha = 0.1

detected_faces = 0
detected_faces_index = []

print("Rows of data:", len(test_data))

# confusion matrix
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

start = time.time()#Start Timer

for i in range(len(test_data)):
    print("Progress: ", i*100/len(test_data),"%")
    # forward calculation
    z = numpy.ones(NEURONS + 1)

    # last weight should only have 6 items
    for j in range(NEURONS):
        #print("Data Row:", test_data[i])
        #print("Neuron Weights:", neuron_weights[j])
        dot_product_hidden = -numpy.dot(test_data[i], neuron_weights[j])
        z[j] = 1 / (1 + numpy.exp(dot_product_hidden))

    dot_product_output = -numpy.dot(numpy.append(1, z[0:NEURONS]), output_weights)
    z[NEURONS] = 1 / (1 + numpy.exp(dot_product_output))

    # prediction
    prediction = round(z[NEURONS], 0)
    if prediction > 0:
        detected_faces += 1
        detected_faces_index.append(i)

    #Confusion Matrix Logic
    if prediction == 1:
        if test_data_classifiers[i] == 1:
            true_positive = true_positive + 1
            test_data_images[i].save_face("TruePositive"+str(i))
        else:
            false_positive = false_positive + 1
    else:
        if test_data_classifiers[i] == 1:
            false_negative = false_negative + 1
        else:
            true_negative = true_negative + 1

    print("prediction", prediction)
    print("class", test_data_classifiers[i])

    #print("Faces:", detected_faces)
    #print("Face locations: ", detected_faces_index)

end = time.time()

print("\nConfusion Matrix:")
print("True Positive: ", true_positive, "|  True Negative: ", true_negative)
print("False Positive: ", false_positive, "|  False Negative: ", false_negative)

print("Testing Time:", end - start)

time_string = "Testing Time: " + str(end - start)

#open text file
text_file = open("TestingTime.txt", "w")
text_file.write(time_string)
text_file.close()

exit()

