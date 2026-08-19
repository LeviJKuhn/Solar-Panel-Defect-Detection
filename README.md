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
