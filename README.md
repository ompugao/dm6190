# Installation
```sh
pip install -r requirements.txt
```

# Data
The Semantic Drone Dataset https://www.tugraz.at/index.php?id=22387

smaller dataset is created with the following command:

```sh
mkdir -p semantic_drone_dataset/small_images semantic_drone_dataset/small_label_images_semantic
size=600x400
#size=1200x800
mogrify -path semantic_drone_dataset/small_images -resize $size semantic_drone_dataset/original_images/*.jpg
mogrify -path semantic_drone_dataset/small_label_images_semantic -resize $size semantic_drone_dataset/label_images_semantic/*.png



```
