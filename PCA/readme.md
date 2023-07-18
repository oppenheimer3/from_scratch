Compress and Decompress Numpy Data

This Python script provides a simple implementation for compressing and decompressing data using Principal Component Analysis (PCA) based on Numpy library. PCA is a widely used technique in data analysis and dimensionality reduction.
Requirements

    Python 3.x
    Numpy library (Install using pip install numpy if you haven't installed it)

Usage

-Save your data in a Numpy format (.npy file).

-Open the terminal and run the script with the desired arguments.

Arguments

The script can be run with the following command-line arguments:

    --type: Specify the operation type. Use 'compress' for data compression or 'decompress' for data decompression.
    --data: Path to the Numpy data file to be compressed or decompressed.
    --l: The dimension of the coded vectors for compression.
    --cdata: Path to the Numpy compressed data file (only required for decompression).
    --decode: Path to the Numpy decoding matrix file (only required for decompression).

Example

To compress the data:

bash

    python pca.py --type compress --data path/to/your/data.npy --l 2

This will compress the data with 10 coded vectors and save the compressed data as 'compressed_data.npy' and the decoding matrix as 'decoding_matrix.npy' in the current directory.

To decompress the data:

bash

    python pca.py --type decompress --cdata path/to/compressed_data.npy --decode path/to/decoding_matrix.npy

This will decompress the data using the provided compressed data and decoding matrix and save the reconstructed data as 'reconstructed.npy' in the current directory.

example.npy

    [[1 2 3]
     [2 4 5]
     [3 5 6]]
compressed_data.npy

    [[-3.72093206 -6.70488789 -8.36085845]
     [ 0.38008036  0.16915167 -0.30480078]]
reconstructed.npy

    [[0.94030074 2.07444375 2.96686941]
     [2.07444375 3.90717016 5.04131317]
     [2.96686941 5.04131317 5.98161391]]
Important Notes
    
The script utilizes PCA to compress and decompress data, Keep in mind that PCA may not be the best choice for all types of data compression tasks.
Ensure that the input data is in Numpy format (.npy).
Always remember to backup your original data before running any compression process, as compression can result in information loss.

License

This script is open-source and distributed under the MIT License. Feel free to modify and use it according to your needs.
