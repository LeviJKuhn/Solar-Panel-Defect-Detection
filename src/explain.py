"""GradCAM explainability."""
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def apply_gradcam(model, image_tensor, target_layer, device, class_idx=None):
    model.eval()

    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = None
    if class_idx is not None:
        targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    img = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())

    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

    return grayscale_cam
