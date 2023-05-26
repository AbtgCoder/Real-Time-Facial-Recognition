
from image_capture import capture_images
from data_preprocessing import preprocess_images
from model_training import train_model
from model_evaluation import evaluate_model

EPOCHS = 50

if __name__ == "__main__":
    # capture_images()
    train_data, test_data = preprocess_images()
    trained_siamese_model = train_model(train_data, EPOCHS)
    evaluate_model(trained_siamese_model, test_data)
