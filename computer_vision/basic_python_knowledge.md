# basic python knowledge

* sys
  1. sys.argv commond line argument
  2. sys.exit() exit the program
  3. sys.path python path  (sys.path.append())

* from PIL import Image
  1. img = Image.open('img.path') PIL.Image.Image 对象
     img_array = np.array(img)
  2. img.show() -> PIL.Image.Image 对象
  3. img.save('img.path')
  4. width, height = img.size
  5. new_img = img.resize((width, height))
> Image将每个图片变成了一个实体对象，调用这些类中包含的方法，可以操作图片。

* import matplotlib.pyplot as plt
  1. img = plt.imread('img.path') numpy.ndarray 对象
  2. plt.imshow(img) -> numpy.ndarray 对象 or PIL.Image.Image 对象(H, W, C)
  3. plt.show()
  4. plt.imsave('img.path', img)
> plt将这些图片实际上是变成了一些客体，并不是直接的去调用这些东西，而是用plt来进行操作。

* import os
  1. new_path = os.path.join("parent_dir", "child_dir", "file.txt") concat path
  2. os.path.exists(path) 判断路径是否存在
  3. abs_path = os.path.abspath("file.txt") 获取绝对路径