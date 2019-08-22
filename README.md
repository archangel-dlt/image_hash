# Content hash for image
This repo contains code to hash images, create a hash database and check whether a new image has a near duplicate entry in the database.

## Hash an image
Check [example.py](example.py) for an example of how to extract content hash from an image. Basically,

```
from utils.extractor import Extractor
hasher = Extractor()
feat, th = hasher.extract2('path_to_my_example_image.jpg')
```

This will return the hash of an image as a vector of float32 numbers (`feat`). It also returns a near duplication threshold value (`th`) which can be used to determine two images are near identical or not.

## Create a hash database
Example can be found at [example.py](example.py). It is almost the same as hashing a single image except you provide a list of image paths to the hasher:

```
feats, ths = hasher.extract_batch(list_of_image_paths)
np.savez('hash_database.npz', feats=feats, ths=ths)
```
This will create a database file (`hash_database.npz`) in numpy format. At the moment we support numpy and hdf5 format however you can use any data format (e.g. sql) to store `feats` and `ths`. Just add a python wrapper class in [utils/database.py](utils/database.py) describing how to read it.

## Build a search model
To make search faster we can prebuild a search model.
```
python build_searchtree.py -i hash_database.npz -o search_index.pkl
```
Here the search model is saved at `search_index.pkl`. As users add more images into the database, we need to occasionally run this command to rebuild the search model.

## Check near-duplication
```
python check.py -i my_test_image.jpg -d hash_database.npz -s search_index.pkl
```
This script (`check.py`) inputs a query image, paths to the database file and the search model. It then tells you if there is a near-duplicated image in the database plus the index of the closest image.

## TODO
- <s>ANN search.</s>
- <s>Create docker file.</s>
- Turn hash (currently numerical) into hex string: dropped as unnecessary.
- zeromq to save time from loading model into RAM.
