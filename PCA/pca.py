import numpy as np
import argparse

def compress(x, l):
    y = np.matmul(x.T , x)
    eigenvalues, eigenvectors = np.linalg.eig(y)
    max_l_eivalues = eigenvalues.argsort()[::-1][:l]
    d = eigenvectors[:, max_l_eivalues]
    c = np.matmul(d.T , x)
    return c, d

def decompress(c, d):
    return np.matmul(d, c)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='compress numpy data')
    parser.add_argument('--type', type=str, help='compress or decompress')
    parser.add_argument('--data', type=str, help='Path to the numpy data file.')
    parser.add_argument('--l', type=int, help='the dimension of the coded vectors')   
    parser.add_argument('--cdata', type=str, help='Path to the numpy compressed data file.')
    parser.add_argument('--decode', type=str, help='Path to the numpy decoding matrix file.')

    args = parser.parse_args()
    t = args.type
    if t == 'compress':
        x = np.load(args.data)
        l = args.l
        c, d = compress(x, l)
        np.save('compressed_data.npy', c)
        np.save('decoding_matrix.npy', d)
    elif t == 'decompress':
        c, d = np.load(args.cdata), np.load(args.decode)
        x_hat = decompress(c, d)
        np.save('reconstructed.npy', x_hat)