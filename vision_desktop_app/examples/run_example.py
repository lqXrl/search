import os
from vision_desktop_app.training.sample_data_generator import generate_sample, CLASS_NAMES
from vision_desktop_app.training.trainer import Trainer
from vision_desktop_app.training.predictor import load_model, predict_image

if __name__ == '__main__':
    data_dir = 'data/sample_dataset'
    os.makedirs('data', exist_ok=True)
    generate_sample(data_dir, n_images=50)
    # Train a classification model on the sample dataset
    model_dir = 'models/inside_outside'
    num_classes = len(CLASS_NAMES)
    tr = Trainer(model_type='classification', model_dir=model_dir, input_shape=(128,128,3), num_classes=num_classes)
    tr.build_model()
    tr.train(data_dir, epochs=2, batch_size=8)
    # Run prediction on first image
    sample_image = os.path.join(data_dir, 'image_0000.png')
    model = load_model(os.path.join(model_dir, 'saved_model'))
    res = predict_image(model, sample_image, input_size=(128,128))
    print('Prediction result:', res)
