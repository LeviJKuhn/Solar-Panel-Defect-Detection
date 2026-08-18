# Solar Panel Defect Detection with Supervised Learning
### Created By: [Levi K](https://www.linkedin.com/in/levijkuhn/), Mukta M, Tony N, Chiamaka O  

## Project Overview  
This project uses transfer learning on a ResNet-18 model to classify solar panel images into six condition categories (Clean, Dusty, Bird-drop, Snow-Covered, Physical-Damage, Electrical-damage). Early fault detection can help maintenance teams prioritize inspections and keep solar installations running efficiently. The model is trained on a Kaggle image dataset, evaluated against a baseline, and interpreted using GradCAM to visualize what the model "sees" when making predictions. A Streamlit app lets users upload an image and get an instant classification.

### Research Question   
"Can convolutional neural networks (CNNs) accurately classify environmental degradation in solar panel images, such as snow cover, dust layers, cracking, or discoloration, to support automated maintenance detection?"  

## Algorithm     
Using an existing computer vision model, ResNet18, we performed a transferred learning technique to fine-tune its ability to recognize solar panels and the associated defects.  

## Model Evaluation  
<img width="992" height="402" alt="image" src="https://github.com/user-attachments/assets/d66ed422-65c7-4e3b-bbfc-bf91cbb282ee" />


## Impact and Bias  
Grad-CAM helps to ensure that the model’s predictions are off the panel’s surface rather on the other elements in the image.
Stratified Sampling helps ensure that the categories within the minority are present in every training epoch.

## Documentation   
This project fine-tunes a ResNet-18 model pretrained on ImageNet, obtained via torchvision.models.  
Solar panel imagery is from the [Solar Panel Images dataset](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) by pythonafroz.

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
