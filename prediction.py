import coremltools as ct
import PIL.Image
import numpy as np
import re
import pandas as pd
import sys

def clean_breed_name(breed_str):
    breed_str = re.sub(r'^n\d+\-', '', breed_str)
    breed_str = breed_str.replace('_', ' ')
    breed_str = ' '.join(word.capitalize() for word in breed_str.split())
    return breed_str

def core_ml_image_prediction(model: str, path: str, resize_to: tuple[int, int] | bool = True, round_value: int = 5) -> dict:
    """
    Performs image prediction using a Core ML model.

    Args:
        model (str): Path to the Core ML model file.
        path (str): Path to the input image file.
        resize_to (tuple or bool, optional): If a tuple (width, height) is provided, the input image will be resized to the specified dimensions. If True, the image will be resized to the model's input dimensions. If False, no resizing will be performed. Defaults to True.
        round_value (int, optional): Number of decimal places to round the predicted probabilities. Defaults to 5.

    Returns:
        dict: A dictionary containing the predicted breed and a pandas DataFrame with the predicted probabilities for each breed, sorted in descending order.

    Example:
        >>> model_path = 'path/to/model.mlmodel'
        >>> image_path = 'path/to/image.jpg'
        >>> result = core_ml_image_prediction(model_path, image_path)
        >>> print(result['breed'])
        'golden_retriever'
        >>> print(result['probabilities'].head())
                   Breed  Probability
        0  golden_retriever     0.98765
        1         labrador     0.01234
        2            poodle     0.00001
    """
    # resize_to: (Width, Height)
    model = ct.models.MLModel(model)
    img = PIL.Image.open(path)
    if isinstance(resize_to, tuple):
        img = img.resize(resize_to, PIL.Image.LANCZOS)
    elif resize_to is not False:
        width = model.get_spec().description.input[0].type.imageType.width
        height = model.get_spec().description.input[0].type.imageType.height
        img = img.resize((width,height), PIL.Image.LANCZOS)
    img_np = np.array(img).astype(np.float32)
    model_output = model.predict({'image': img})
    target_breed = clean_breed_name(model_output['target'].split('-')[1])
    rounded_probabilities = [(clean_breed_name(breed), prob) for breed, prob in model_output['targetProbability'].items()]
    rounded_probabilities = pd.DataFrame(rounded_probabilities, columns=['Breed', 'Probability']).sort_values(by='Probability', ascending=False, ignore_index=True)
    rounded_probabilities['Probability'] = rounded_probabilities['Probability'].round(round_value)
    cleaned_result = {'breed': target_breed, 'probabilities': rounded_probabilities}
    return img_np, img, cleaned_result

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    img_np, img, result = core_ml_image_prediction(model_path, image_path)
    print(f"The doggy is: {result['breed']}")
    print(result['probabilities'].head())

if __name__ == "__main__":
    main()
