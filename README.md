# The Waste Classifier

Welcome in the repository of my first complete AI/ML project.

<img width="1920" height="1632" alt="Glass" src="https://github.com/user-attachments/assets/83c523a2-91cd-49f9-b47c-e7b7f4c9468c" />

The Waste Classifier is a web application which enables people to automatically classify their wastes into one of ten categories:
- Battery
- Biological
- Cardboard
- Clothes
- Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash

Application is publicly available at https://waste-classifier-ui-943543256910.europe-central2.run.app

All you need to do is to upload an image and after a while you will know what kind of waste you have (there is a cold start, so at first time you need to wait 30 seconds, but then it is running really fast).

## How it is build?

### Tech stack
- Python, Tensorflow, scikit-learn for learning neural network of the Classifier
- Google Cloud Run, Terraform, REST API, Docker for hosting the Classifier
- Python Streamlit and above for frontend application

### Classifier
In heart of the system lies the classifier, which is a trained neural network model. It was trained on a few Kaggle datasets, containing nearly 20 000 unique images, using transfer learning method from MobileNetV2 network. The training time was nearly 10 hours because it was done purely on the CPU (on 9 years old laptop :)).

Due to the strong augmentation on the training dataset, the training curve never crosses the validation curve, but the overall accuracy on the test set was 92.69%, which is satisfying for such architecture.

<img width="1950" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/3f1929bf-0763-400c-afb8-810f789fd02b" />

The confusion matrix presents the model performance for all classes.

<img width="1500" height="1200" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c5c29d8e-f7b3-446f-bf50-98f34d9df520" />

### API

The classifier model is provided to the users by the REST API with an endpoint to make a prediction on the input image. It is hosted on Google Cloud Run and available on demand, but there is a cold start at the first time, which lasts 30 seconds.

### Application

The user interface is brought by simple Streamlit application hosted also on Google Cloud Run. The application calls the API with user input image and presents the results of classification.

### Documentation

All important details about the project parts are written in the Documentation directory. 
