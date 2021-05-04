import numpy as np
import random
import matplotlib.pyplot as plt
import math


def read_signal(file_name, signal_length):
    data = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            new_line = line.replace('[', '').replace(']', '')
            data.append(new_line.split(', '))
    float_data = []
    for item in data:
        float_data.append([float(x) for x in item])

    np_data = np.asarray(float_data)
    data = np.reshape(np_data, (np_data.shape[1] // signal_length, signal_length))

    return data


def draw_signal(signal, index_signal, title = ''):
    plt.title('Сигнал ' + str(index_signal) + title)
    plt.plot(range(len(signal)), signal)
    plt.show()
    plt.savefig('signal' + str(index_signal) + '.jpg')


def draw_hist(signal):
    bins = int(math.log2(len(signal) + 1))
    hist = plt.hist(signal, bins=bins)
    plt.title('Гистограмма сигнала')
    plt.show()
    plt.savefig('histogram.jpg')

    return bins, hist


def get_areas(signal, bins, hist):
    count = []
    start = []
    finish = []
    types = [''] * bins
    for i in range(bins):
        count.append(hist[0][i])
        start.append(hist[1][i])
        finish.append(hist[1][i + 1])

    sorted_hist = sorted(count)
    repeat = 0
    for i in range(bins):
        for j in range(bins):
            if sorted_hist[-(1 + i)] == count[j]:
                if repeat == 0:
                    types[j] = 'Фон'
                elif repeat == 1:
                    types[j] = 'Сигнал'
                else:
                    types[j] = 'Переход'
                repeat += 1

    return start, finish, types


def get_converted_data(signal, start, finish, types):
    p_type = [0] * len(signal)
    zones = []
    areas_type = []

    for i in range(len(signal)):
        for j in range(len(types)):
            if (signal[i] >= start[j]) and (signal[i] <= finish[j]):
                p_type[i] = types[j]

    current_type = p_type[0]
    start = 0
    for i in range(len(p_type)):
        if current_type != p_type[i]:
            finish = i
            areas_type.append(current_type)
            zones.append([start, finish])
            start = finish
            current_type = p_type[i]

    if current_type != areas_type[-1]:
        areas_type.append(current_type)
        zones.append([finish, len(signal) - 1])

    return zones, areas_type


def reduce_emissions(signal, area_data, types, hist, bins):
    while len(types) > 5:
        for i in range(len(types)):
            if (types[i] == "переход") and (types[i - 1] == types[i + 1]):
                startValue = signal[area_data[i - 1][1] - 1]
                finishValue = signal[area_data[i + 1][0] + 1]
                newValue = (startValue + finishValue) / 2
                num = area_data[i][1] - area_data[i][0]
                for j in range(num):
                    signal[area_data[i][0] + j] = newValue

        start, finish, types = get_areas(signal, bins, hist)
        area_data, types = get_converted_data(signal, start, finish, types)

    return signal, area_data, types


def draw_areas(signal, area_data, types):
    plt.title('Разделение области для сигнала без выбросов')
    index = [i for i in range(len(signal))]
    for i in range(len(area_data)):
        if types[i] == "Фон":
            color = 'g'
        elif types[i] == "Сигнал":
            color = 'r'
        else:
            color = 'y'

        plt.plot(index[area_data[i][0]:area_data[i][1]],
                 signal[area_data[i][0]:area_data[i][1]], color=color, label=types[i])
    plt.legend()
    plt.show()
    plt.savefig('areas.jpg')


def get_inter_group_d(signal):
    sum = 0
    mean = np.empty(signal.shape[0])
    for i in range(len(signal)):
        mean[i] = np.mean(signal[i])
    mean_mean = np.mean(mean)

    for i in range(len(mean)):
        sum += (mean[i] - mean_mean) ** 2
    sum /= (signal.shape[0] - 1)

    return len(signal) * sum


def get_intar_group_d(signal):
    result = 0
    for i in range(signal.shape[0]):
        mean = np.mean(signal[i])
        sum = 0
        for j in range(signal.shape[1]):
            sum += (signal[i][j] - mean) ** 2
        sum /= (signal.shape[0] - 1)
        result += sum

    return result / signal.shape[0]


def get_f(signal, k):
    new_size_y = int(signal.size / k)
    new_size_x = k
    print("k = " + str(k))
    split_data = np.reshape(signal, (new_size_x, new_size_y))
    inter_group = get_inter_group_d(split_data)
    print("Inter = " + str(inter_group))
    intra_group = get_intar_group_d(split_data)
    print("Intar = " + str(intra_group))
    print("F = " + str(inter_group / intra_group))
    print()
    return inter_group / intra_group


def get_k(num):
    i = 4
    while num % i != 0:
        i += 1
    return i


def get_fisher(signal, area_data):
    fishers = []
    for i in range(len(area_data)):
        start = area_data[i][0]
        finish = area_data[i][1]
        k = get_k(finish - start)
        while k == finish - start:
            finish += 1
            k = get_k(finish - start)
        fishers.append(get_f(signal[start:finish], int(k)))
    return fishers


if __name__ == "__main__":
    data = read_signal('wave_ampl.txt', 1024)
    index_signal = random.randint(0, len(data) - 1)
    signal = data[index_signal]

    draw_signal(signal, index_signal)
    bins, hist = draw_hist(signal)

    start, finish, types = get_areas(signal, bins, hist)
    zones, areas_type = get_converted_data(signal, start, finish, types)
    signal_without_emissions, area_data, types = reduce_emissions(signal, zones, areas_type, hist, bins)

    draw_signal(signal_without_emissions, index_signal, ' без выбросов')
    draw_areas(signal_without_emissions, zones, types)

    fishers = get_fisher(signal, zones)

    print(fishers)

