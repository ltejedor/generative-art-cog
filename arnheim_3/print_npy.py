#!/usr/bin/env python3
import numpy as np
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: {} <npy_file>".format(sys.argv[0]))
        sys.exit(1)
        
    npy_file = sys.argv[1]
    
    try:
        # Load the .npy file; allow_pickle=True if your file contains Python objects.
        data = np.load(npy_file, allow_pickle=True)
        print("Contents of '{}':\n{}".format(npy_file, data))
    except Exception as e:
        print("Error loading file: {}".format(e))
        sys.exit(1)

if __name__ == '__main__':
    main()
