import coremltools as ct
import PIL.Image
import numpy as np
import re
import pandas as pd

def clean_breed_name(breed_str):
    breed_str = re.sub(r'^n\d+\-', '', breed_str)
    breed_str = breed_str.replace('_', ' ')
    breed_str = ' '.join(word.capitalize() for word in breed_str.split())
    return breed_str

def core_ml_image_prediction(model, path, resize_to=True, round_value = 5):
    # resize_to: (Width, Height)
    model = ct.models.MLModel(model)
    width = model.get_spec().description.input[0].type.imageType.width
    height = model.get_spec().description.input[0].type.imageType.height
    img = PIL.Image.open(path)
    if resize_to is not False:
        img = img.resize((width,height), PIL.Image.LANCZOS)
    img_np = np.array(img).astype(np.float32)
    model_output = model.predict({'image': img})
    target_breed = clean_breed_name(model_output['target'].split('-')[1])
    rounded_probabilities = [(clean_breed_name(breed), prob) for breed, prob in model_output['targetProbability'].items()]
    rounded_probabilities = pd.DataFrame(rounded_probabilities, columns=['Breed', 'Probability']).sort_values(by='Probability', ascending=False, ignore_index=True)
    rounded_probabilities['Probability'] = rounded_probabilities['Probability'].round(round_value)
    cleaned_result = {'breed': target_breed, 'probabilities': rounded_probabilities}
    return img_np, img, cleaned_result

if __name__ == "__main__":
    clean_breed_name()
    core_ml_image_prediction()



