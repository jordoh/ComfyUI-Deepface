# ComfyUI Deepface

ComfyUI nodes wrapping the [deepface](https://github.com/serengil/deepface) library.

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```

## Nodes


### Deepface Extract Faces

TODO: document

### Deepface Verify

Given a set of input images and a set of reference images, only output the input images that - on average - have a 
distance to the set of reference images less than or equal to the specified threshold. Output images are sorted by
average distance to the reference images (closest first). 

![verify workflow](./workflows/verify.png)
