import numpy as np
# import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imsave
from init_centroids import init_centroids


def plot_loss_func(loss, k):
    plt.title("Average loss function for " + str(k) + " centroids")
    plt.xlabel("Iteration number")
    plt.ylabel("Average loss")
    plt.plot(range(len(loss)), loss)
    plt.show()


def format_int(num):
    num = np.floor(num * 100)/100
    return num


def print_iteration(centroids, iter_num):
    str_to_print = "iter " + str(iter_num) + ": "

    for i in range(len(centroids) - 1):
        str_to_print += "["
        for j in range(len(centroids[0]) - 1):
            str_to_print = str_to_print + str(format_int(centroids[i][j])) + ", "
        str_to_print = str_to_print + str(format_int(centroids[i][len(centroids[0]) - 1])) + "], "

    str_to_print += "["
    for j in range(len(centroids[0]) - 1):
        str_to_print = str_to_print + str(format_int(centroids[len(centroids) - 1][j])) + ", "
    str_to_print = str_to_print + str(format_int(centroids[len(centroids) - 1][len(centroids[0]) - 1])) + "]"

    print(str_to_print)


# dev each cell in the vector x by the number num
def dev_by_number(x, num):
    ans = np.zeros(len(x))
    for i in range(len(x)):
        ans[i] = x[i] / num

    return ans


# Adding two pixels
def add(x, y):
    sum_x_y = np.zeros(len(x))
    for i in range(len(x)):
        sum_x_y[i] = x[i] + y[i]

    return sum_x_y


# Check if two pixels are equal
def equal(x, y):
    is_equal = True

    for i in range(len(x)):
        if x[i] != y[i]:
            is_equal = False
            break

    return is_equal


# Calc the distance between two pixels
def distance(x, y):
    dst = 0

    for i in range(len(x)):
        dst += pow(x[i] - y[i], 2)
    return dst


# Find for each pixel its closest centroid
def find_closest_centroid(image, centroids):
    final_image = np.zeros(len(image) * len(image[0]))
    final_image = final_image.reshape(len(image), len(image[0]))
    loss = 0

    # Go throgut all the pixels in the image
    for pixel_index in range(len(image)):
        # Create an array that will contain the distance of the current pixel from all centroids
        cost = np.zeros(len(centroids))

        # Calc the distance of each centroid from the current pixel
        for centroid_index in range(len(centroids)):
            cost[centroid_index] = distance(image[pixel_index], centroids[centroid_index])

        # Find the closest centroid
        closest_dist = cost[0]
        closest_dist_index = 0

        for centroid_index in range(len(centroids)):
            if closest_dist > cost[centroid_index]:
                closest_dist = cost[centroid_index]
                closest_dist_index = centroid_index

        # Put the closest centroid of the current pixel in the final image
        final_image[pixel_index] = centroids[closest_dist_index]

        # Update the loss
        loss += closest_dist

    loss = (loss / len(image)).astype(float)
    return final_image, loss


def update_centroids(image, centroids, final_image):
    new_centroids = np.zeros(len(centroids) * len(centroids[0]))
    new_centroids = new_centroids.reshape(len(centroids), len(centroids[0]))
    num_of_pixel_for_each_centroid = np.zeros(len(centroids))

    # Calc the new centroids
    for pixel_index in range(len(image)):
        # Check the group which the current pixel belongs to
        for j in range(len(centroids)):
            if equal(final_image[pixel_index], centroids[j]):
                num_of_pixel_for_each_centroid[j] += 1
                new_centroids[j] = add(new_centroids[j], image[pixel_index])
                break

    for i in range(len(new_centroids)):
        new_centroids[i] = dev_by_number(new_centroids[i], num_of_pixel_for_each_centroid[i])

    return new_centroids


def k_means(k):
    # Data preparations
    path = 'pic.jpeg'
    a = imread(path)
    a = a.astype(float) / 255.
    img_size = a.shape
    image = a.reshape(img_size[0] * img_size[1], img_size[2])

    centroids = init_centroids(image, k)
    final_image = 0
    loss_in_itereations = np.zeros(10)

    print("k=" + str(k))

    # 10 Iterations of the algorithm
    for i in range(0, 10):
        print_iteration(centroids, i)
        final_image, loss_in_itereations[i] = find_closest_centroid(image, centroids)
        centroids = update_centroids(image, centroids, final_image)

    final_image = final_image.reshape(img_size[0], img_size[1], img_size[2])
    final_image = final_image * 255
    final_image = final_image.astype(int)

    return final_image, loss_in_itereations


loss_x_coords = range(10)
final_image2, loss2 = k_means(2)
imsave("pic_2.jpeg", final_image2)
plot_loss_func(loss2, 2)

final_image4, loss4 = k_means(4)
imsave("pic_4.jpeg", final_image4)
plot_loss_func(loss4, 4)

final_image8, loss8 = k_means(8)
imsave("pic_8.jpeg", final_image8)
plot_loss_func(loss8, 8)

final_image16, loss16 = k_means(16)
imsave("pic_16.jpeg", final_image16)
plot_loss_func(loss16, 16)
