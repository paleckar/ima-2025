# Image analysis 2025

> [!NOTE]
> The class will close on 31st January 2026. After that, it will no longer be possible to submit solutions to assignments and the grading will be finalized.


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


## Lectures

| #  | date       | lecture                                       | topics                                         |
|----|------------|-----------------------------------------------|------------------------------------------------|
| 1. | 2025/09/15 | [Preliminaries](lectures/preliminaries.ipynb) | Setup, NumPy, Matplotlib, scikit-image, OpenCV |


## Useful resources

- Wilhelm Burger, Mark J. Burge: *Digital Image Processing: An Algorithmic Introduction (3rd edition)*  
  https://imagingbook.com/books/english-edition-hardcover/
- Rafael C. Gonzalez, Richard E. Woods.: *Digital Image Processing (4th edition)*  
  https://www.imageprocessingplace.com/
- Vincent Mazet: *Basics of Image Processing*  
  https://vincmazet.github.io/bip/index.html
