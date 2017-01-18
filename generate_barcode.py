"""
Returns the barcode for the given ID based on http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136487
"""
import numpy as np


def get_barcode(id):
    """
    Returns the barcode for the ID as numpy array of size 14*14
    Numpy array is transposed so that array[i, j] is array[x, y]
    """
    # first 15 bits
    binary_code = bin(id)[2:]
    while(len(binary_code) < 15):
        binary_code = "0"+binary_code
    binary = [int(char) for char in binary_code]
    binary = np.array(binary).reshape((3, 5)).transpose()

    # next 3 bits
    error_code = []
    for i in range(binary.shape[1]):
        parity_bit = 0
        for j in range(binary.shape[0]):
            parity_bit = parity_bit ^ binary[j, i]
        error_code.append(parity_bit)

    # 4th bit
    parity_bit = 0
    for i in range(3):
        for j in range(binary.shape[1]):
            parity_bit = parity_bit ^ binary[j, i]
    error_code.append(parity_bit)

    # 5th bit
    parity_bit = 0
    for i in range(3, binary.shape[0]):
        for j in range(binary.shape[1]):
            parity_bit = parity_bit ^ binary[i, j]
    error_code.append(parity_bit)

    # next 5 bits
    error_code.extend(error_code[::-1])
    error_code = np.array(error_code).reshape((2, 5)).transpose()

    # combining all 25 bits
    barcode = np.concatenate((binary, error_code), axis=1)
    barcode = np.pad(barcode, 1, 'constant', constant_values=1)
    barcode = np.pad(barcode, 1, 'constant', constant_values=0)

    resized_barcode = np.zeros((2*barcode.shape[0], 2*barcode.shape[1]))
    for i in range(resized_barcode.shape[0]):
        for j in range(resized_barcode.shape[1]):
            resized_barcode[i, j] = barcode[int(i/2), int(j/2)]

    # since we want indexing of array in x, y
    return resized_barcode.transpose()

if __name__ == '__main__':
    get_barcode(11)
