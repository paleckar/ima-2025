# Image analysis 2025

> [!NOTE]
> The class will close on 31st January 2026. After that, it will no longer be possible to submit solutions to assignments and the grading will be finalized.


## Lectures

| #  | date       | lecture                                                               | topics                                                     |
|----|------------|-----------------------------------------------------------------------|------------------------------------------------------------|
| 1. | 2025/09/15 | [Preliminaries](lectures/preliminaries.ipynb)                         | setup, NumPy, Matplotlib, scikit-image, OpenCV             |
| 2. | 2025/09/22 | [Digital images](lectures/digital_images.ipynb)                       | acquisition, digital image, formats, histogram             |
| 3. | 2025/09/29 | [Intensity transformations](lectures/intensity_transformations.ipynb) | histogram equalization, matching, gamma correction, CLAHE  |
| 4. | 2025/10/06 | [Spatial filtering](lectures/spatial_filtering.ipynb)                 | linear & nonlinear filters, convolution, separability      |
| 4. | 2025/10/13 | [Frequency domain](lectures/frequency_domain.ipynb)                   | DFT (+2D), filtering in spectrum, restoration, compression |


## Assignments

| #  | assignment                                                         | points      | deadline   |
|----|--------------------------------------------------------------------|-------------|------------|
| 1. | [Watermarking](assignments/watermarking.ipynb)                     | 3           | 2025/10/12 |
| 2. | [Histogram equalization](assignments/histogram_equalization.ipynb) | 1+2+2(+5)   | 2025/10/19 |
| 3. | [Filter effects](assignments/filter_effects.ipynb)                 | 3+5         | 2025/10/26 |
| 4. | [Notch filters](assignments/notch_filters.ipynb)                   | 2+3+5+3(+3) | 2025/11/02 |

## Useful resources

- Wilhelm Burger, Mark J. Burge: *Digital Image Processing: An Algorithmic Introduction (3rd edition)*  
  https://imagingbook.com/books/english-edition-hardcover/
- Rafael C. Gonzalez, Richard E. Woods.: *Digital Image Processing (4th edition)*  
  https://www.imageprocessingplace.com/
- Vincent Mazet: *Basics of Image Processing*  
  https://vincmazet.github.io/bip/index.html


## Setup

### 1. Prerequisites
- The class repository uses [uv](https://docs.astral.sh/uv/) as a package manager.
- It can be installed as  
  ```
  # Windows:
  winget install --id=astral-sh.uv  -e
  # Ubuntu:
  sudo snap install astral-uv
  # macOS brew:
  brew install uv
  # Other Linux distros and macOS:
  wget -qO- https://astral.sh/uv/install.sh | sh
  ```

### 2. Virtual environment and dependencies installation
- Download or clone the repository.
- Go to the repository folder
  ```
  cd /path/to/ima-2025/
  ```
- Install all the necessary dependencies by running  
  ```
  uv sync
  ```

### 3. Viewing the content
- The class materials are presented predominantly in the form of [Jupyter notebooks](https://jupyter.org/).
- You may use any of your favorite tools to view and edit the notebooks.
- The recommended way is to use [Visual Studio Code](https://code.visualstudio.com/) with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions.
