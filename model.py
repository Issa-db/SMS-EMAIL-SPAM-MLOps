import os
import joblib
import onnx
from pathlib import Path
from preparation import prepare_data
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from config import settings 

def build_model():
    """
    Build and train the model using the prepared data.
    Returns:
        model: The trained model.
    """
    
    # load the data 
    X, y = prepare_data()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    # Vectorize the text data
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_data(X_train, X_test)
    # Train the model
    NB = train_model(X_train_vectorized, y_train)
    # Evaluate the model
    evaluate_model(NB, X_test_vectorized, y_test)
    # save_model_onnx(NB, vectorizer)
    
    save_model_onnx(NB,
                    vectorizer, 
                    model_path = settings.model_path / settings.model_name,
                    vectorizer_path = settings.vectorizer_path / settings.vectorizer_name)
    
    

def split_train_test(X,y):
    """
       Split arrays into random train and test subsets.
       
    Args:
        X (str): The input features (text messages).
        y (int): The target labels (spam or ham).
    """
    # Split the data into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test

def vectorize_data(X_train, X_test):
    """
    Vectorize the text data using CountVectorizer.
    
    Args:
        X_train (str): The input features for training (text messages).
        X_test (str): The input features for testing (text messages).
    """
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(lowercase=True,
                                 stop_words="english",
                                 ngram_range=(1, 2),
                                 min_df=5)
    
    # Fit and transform the training data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Transform the test data
    X_test_vectorized = vectorizer.transform(X_test)
    
    return X_train_vectorized, X_test_vectorized, vectorizer

def train_model(X_train_vectorized, y_train):
    """
    Train the model using MultinomialNB.
    
    Args:
        X_train_vectorized (str): The input features for vectorized training (text messages).
        y_train (int): The target labels for training (spam or ham).
    """
    # build model and train it
    grid_space = {
        "alpha": [0.1, 0.5, 1.0],
        "fit_prior": [True, False],
    }
    grid_search = GridSearchCV(MultinomialNB(),
                               grid_space,
                               cv=5,
                               scoring="accuracy")
    # train the model
    grid_search.fit(X_train_vectorized, y_train)
    # return model
    return grid_search.best_estimator_ 

def evaluate_model(model, X_test_vectorized, y_test):
    """
    Evaluate the model's performance on the test set.
    
    Args:
        model: The trained model.
        X_test_vectorized (scipy.sparse._matrix): The input features for vectorized testing (text messages).
        y_test (int): The target labels for testing (spam or ham).
    """
    # Evaluate the model
    accuracy = model.score(X_test_vectorized, y_test)
    #print(f"Model Accuracy: {accuracy:.4f}")
    return print(f"Model Accuracy: {accuracy:.4f}")

def save_model_onnx(model, vectorizer, model_path=None, vectorizer_path=None):
    """
    Save the trained model and vectorizer to disk in ONNX format.
    
    Args:
        model: The trained model.
        vectorizer: The CountVectorizer used for text vectorization.
        model_path (str): The path to save the ONNX model.
        vectorizer_path (str): The path to save the vectorizer.
    """

    # Save the vectorizer using joblib
    joblib.dump(vectorizer, vectorizer_path)

    n_features = len(vectorizer.get_feature_names_out())
    # Convert the model to ONNX format
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to disk
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    return print(f"Model saved to {model_path} and vectorizer saved to {vectorizer_path}")

# run the build_model function to train and evaluate the model
model = build_model()
