[Overview](#Solar-Panel-Defect-Detection-with-Supervised-Learning)  
[Usage](#Implementation)  

# Solar Panel Defect Detection with Supervised Learning
### Created By: [Levi K](https://www.linkedin.com/in/levijkuhn/), Mukta M, Tony N, Chiamaka O  

## Project Overview  
This project uses transfer learning on a ResNet-18 model to classify solar panel images into six condition categories (Clean, Dusty, Bird-drop, Snow-Covered, Physical-Damage, Electrical-damage). Early fault detection can help maintenance teams prioritize inspections and keep solar installations running efficiently. The model is trained on a Kaggle image dataset, evaluated against a baseline, and interpreted using GradCAM to visualize what the model "sees" when making predictions. A Streamlit app lets users upload an image and get an instant classification.

### Research Question   
"Can convolutional neural networks (CNNs) accurately classify environmental degradation in solar panel images, such as snow cover, dust layers, cracking, or discoloration, to support automated maintenance detection?"  

## Algorithm     
Using an existing computer vision model, ResNet18, we performed a transferred learning technique to fine-tune its ability to recognize solar panels and the associated defects by replacing its classification head. 

## Model Evaluation  
This is one sample training of our stratified sampling model with its accuracy sorted by categories. 
<p align="center">
   <img width="700" alt="Per-class test accuracy" src="https://github.com/user-attachments/assets/e7b45e4a-4f1f-45c1-a83d-ddc5ddc398be"/>
   <br>
     <em>Per-class test accuracy. Physical-Damage is both the weakest and smallest class — with only 14 test images, five errors move the estimate by over 35       percentage points, so this figure is far less precise than the others.</em>
</p>


## Impact and Bias  
Grad-CAM helps to ensure that the model’s predictions are based on the panel’s surface rather on the other elements in the image.
Stratified Sampling helps ensure that the categories within the minority are present in every training epoch.

## Documentation   
This project fine-tunes a ResNet-18 model pretrained on ImageNet, obtained via torchvision.models.  
Solar panel imagery is from the [Solar Panel Images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) dataset by pythonafroz.  
GitHub Repository Template forked from [Here](https://github.com/amish-github/ml-project.git)  
GitHub Repository Page Template forked from [Here](https://github.com/AI4ALL-Official/ai4all-official.github.io.git)  

## Next Steps  
### Improving the Model  
Expand the dataset with additional solar panel defect images  
* Improves performance on underrepresented categories
* Ensures over-fitting is less likely to occur
Compare to additional CNN architectures

### Features to consider
Add severity impact estimation  
Detect multiple defects in one image  
Deploy the model for real-time inspection  




# Implementation
## Repository Structure

```
Solar-Panel-Defect-Detection/
├── data/               Solar panel images, organized by class (gitignored)
├── src/                Reusable modules (data loading, model, training, evaluation, GradCAM)
├── scripts/            Entry points: download_data.py, train.py, predict.py
├── notebooks/          EDA and model comparison notebook
├── config/             Hyperparameters in YAML
├── models/             Saved model weights (gitignored)
├── outputs/            Figures, metrics, logs
├── app/                Streamlit demo app
│   └── app.py
├── requirements.txt    Pinned versions of packages
├── .gitignore
└── README.md           Setup, usage, results (this file)
```

**Why this layout?**

| Folder | Purpose |
|---|---|
| `data/` | Keep data out of version control. The dataset is downloaded from Kaggle by `scripts/download_data.py` rather than committed. |
| `src/` | Code you import. Data loading, the model definition, the training loop, evaluation, and GradCAM all live here so scripts, the app, and the notebook share one implementation. |
| `scripts/` | Code you run. Thin entry points that wire together modules from `src/` using settings from `config/`. |
| `notebooks/` | Code you explore with. Class distributions, sample images, model comparisons, and GradCAM visualizations. |
| `config/` | No magic numbers buried in code. Learning rate, epochs, batch size, image size, and class names all live in `config.yaml`. |
| `models/` | Trained weights. The fine-tuned ResNet-18 is ~45MB, under GitHub's 100MB limit, but is gitignored here since it is reproducible from `scripts/train.py`. |
| `outputs/` | Everything your runs produce: the confusion matrix figure and any other metrics or logs. |
| `app/` | A lightweight Streamlit demo — upload a solar panel image and get a predicted fault class with confidence scores. |

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/LeviJKuhn/Solar-Panel-Defect-Detection.git
cd Solar-Panel-Defect-Detection

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

## Usage

**Download the dataset** (pulls the Kaggle solar panel image dataset into `data/`):

```bash
python scripts/download_data.py
```

**Train a model** (reads hyperparameters from `config/config.yaml`, saves the model to `models/` and the confusion matrix to `outputs/figures/`):

```bash
python scripts/train.py
```

**Predict on a new image** (or a folder of images):

```bash
python scripts/predict.py --input data/Faulty_solar_panel/Dusty/example.jpg
```

**Launch the demo app**:

```bash
streamlit run app/app.py
```

**Explore the data and model comparisons**:

```bash
jupyter lab notebooks/01_eda_and_comparisons.ipynb
```
