import numpy
import matplotlib.pyplot
import random
import warnings
import time

warnings.filterwarnings('ignore')

with open('testfile.npy', 'rb') as opened_file:
    faces_array = numpy.load(opened_file)

print(faces_array)

indexArray = list(range(numpy.size(faces_array,0)))
random.shuffle(indexArray)
faces_array = faces_array[indexArray]

faces_array = faces_array[:12500]

count_non = 0
count_faces = 0

for i in faces_array:
    if i[-1] == 0:
        count_non += 1
    else:
        count_faces += 1

print("Non-faces:", count_non)
print("Faces:", count_faces)

#faces_array = faces_array[:100]

faces_classifiers = faces_array.T[-1]
faces_array = faces_array[:, :-1]

# define number of hidden layers
NEURONS = 50
ATTRIBUTES = len(faces_array[0]) - 1

# weights pre-initialized
neuron_weights = []
output_weights = [random.random() * 2 - 1]
for j in range(NEURONS):
    w = []
    output_weights = numpy.append([random.random() * 2 - 1], output_weights)
    for i in range(ATTRIBUTES + 1):
        w = numpy.append([random.random() * 2 - 1], w)
    neuron_weights.append(w)

# print("neuron weights: ", neuron_weights)
# print("neuron weights size: ", len(neuron_weights), len(neuron_weights[0]))
# print("neuron output weights: ", output_weights)
# print("neuron output weights size: ", len(output_weights))


#MLP Training
EPOCHS = 500
alpha = 0.1
previous_error = 0
same_error_count = 0
total_error_list = []

start = time.time()#Start Timer

for epoch in range(EPOCHS):
    total_error = 0
    print("Progress:", str(epoch*100/EPOCHS)+"%")
    for i in range(len(faces_array)):

        # forward calculation
        z = numpy.ones(NEURONS + 1)

        # last weight should only have 6 items
        for j in range(NEURONS):
            dot_product_hidden = -numpy.dot(faces_array[i, :], neuron_weights[j])
            z[j] = 1 / (1 + numpy.exp(dot_product_hidden))

        dot_product_output = -numpy.dot(numpy.append(1, z[0:NEURONS]), output_weights)
        z[NEURONS] = 1 / (1 + numpy.exp(dot_product_output))

        # prediction
        prediction = round(z[NEURONS], 0)
        # print("prediction", prediction)
        # print("class", faces_classifiers[i])

        total_error = total_error + abs(faces_classifiers[i] - prediction)
        # print("total error", total_error)

        # error propagation
        error = numpy.ones(NEURONS + 1)
        error[NEURONS] = z[NEURONS] * (1 - z[NEURONS]) * (faces_classifiers[i] - z[NEURONS])
        for j in range(NEURONS):
            error[j] = z[j] * (1 - z[j]) * (error[NEURONS] * output_weights[j + 1])

        output_weights[0] = output_weights[0] + alpha * error[NEURONS]
        for j in range(NEURONS):
            output_weights[j + 1] = output_weights[j + 1] + alpha * error[NEURONS] * z[j]
            neuron_weights[j][0] = neuron_weights[j][0] + alpha * error[j]
            for k in range(ATTRIBUTES):
                neuron_weights[j][k + 1] = neuron_weights[j][k + 1] + alpha * error[j] * faces_array[i, k + 1]

    total_error_list.append(total_error)

    if previous_error != total_error_list[-1]:
        previous_error = total_error_list[-1]
        same_error_count = 0
    else:
        same_error_count += 1

    if same_error_count > 3:
        break

end = time.time()

numpy.save("neuron_weights", neuron_weights)
numpy.save("output_weights", output_weights)

print("Number of errors during training:", total_error_list[-1])
print("Percent of errors during training:", total_error_list[-1]/len(faces_array) * 100, "%")

matplotlib.pyplot.plot(total_error_list)
matplotlib.pyplot.title(label=f"Alpha: {alpha}")
#matplotlib.pyplot.show()

with open('neuron_weights.npy', 'rb') as opened_file:
    neuron_weights = numpy.load(opened_file)

with open('output_weights.npy', 'rb') as opened_file:
    output_weights = numpy.load(opened_file)

print("neuron_weights:",neuron_weights)
print("output_weights:",output_weights)
print("Training Time:", end - start)

time_string = "Training Time: " + str(end - start)

#open text file
text_file = open("TrainingTime.txt", "w")
text_file.write(time_string)
text_file.close()