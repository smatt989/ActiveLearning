# ActiveLearning
MNIST Active Learning

Requirements to install ( python3.5 or above )

`python -m pip install -r requirements.txt --user --no-cache` or 

`python3 -m pip install -r requirements.txt --user --no-cache`

To run 

`python activemnist.py 1 10 2000 True old True 5` or 

`python3 activemnist.py 1 10 2000 True old True 5`

Additional Running Info

######runs a test with the following parameters:

###1st parameter: number of runs (almost always want just 1 or will be extremely long)

###2nd parameter: how bit each mini-batch should be

###3rd parameter: how many mini-batches in a run

###4th parameter: active learning turned on or not

###5th parameter: classification task ('old', '3s', 'split')

###6th parameter: re-samping turned on or not

###7th parameter: how often training is tested with test set

example: python activemnist.py 1 10 2000 True old True 5

would start one full run with mini-batches of size 10 for 2000 iterations with Active Learning

turned on, classifying digits from 0-9, with re-sampling and printing test results every 5 mini-batches

in addition it would then run 20 epochs at intervals of 250 label counts
