## Code Structure for assignment 1

### To execute all the tasks present in the assignment run the following commands

bash
```
pip install -r requirements.txt
```

python 
```
python main.py
```

### Code Structure

- ip_images dir : contains input images
- op_images dir : contain the output images generated from each task
- src dir       : contains implementation of otsu's algorithm and connected component extarction (all variants i.e brute force , better & optimal)

- tasks dir     : contains broilerplate code to run each task

- main.py       : puts code for executing each task in a single file

### Sailent features

This implementation for otsu's algorithm beats the official highly c++ optimized code use by cv2 library by almost 10x

After running the experiment for multiple times these are averaged out results

This Implementation : 4.82 * 10 ^ -6 sec
CV2 Implementation  : 4.88 * 10 ^ -5 sec