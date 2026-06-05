# The Waste Classifier

Welcome in the repository of my first complete AI/ML project.

The Waste Classifier is a web application which enables people to automatically classify their wastes into one of ten categories:
- Battery,
- Biological,
- Cardboard,
- Clothes,
- Glass,
- Metal,
- Paper,
- Plastic,
- Shoes,
- Trash.

Application is publicly available at https://waste-classifier-ui-943543256910.europe-central2.run.app

All you need to do is to upload an image and after a while you will know what kind of waste you have.

<img width="1920" height="1632" alt="Glass" src="https://github.com/user-attachments/assets/83c523a2-91cd-49f9-b47c-e7b7f4c9468c" />

## How it is build?

### Classifier
The heart of the system is the classifier, which is a trained neural network model. It was trained on a few Kaggle datasets using transfer learning method from MobileNetV2 network. The training time was nearly 10 hours because it was done purely on the CPU.

Due to the stong augmentation on the training dataset, the training curve never crosses the validation curve, but the overall accuracy on the test set was 92.69%.

<img width="1950" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/3f1929bf-0763-400c-afb8-810f789fd02b" />

The confusion matrix presents the model performance for all classes.

<img width="1500" height="1200" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c5c29d8e-f7b3-446f-bf50-98f34d9df520" />

The technological stack used was Python, TensorFlow and Keras.

### API

The classifier is 

### Application
