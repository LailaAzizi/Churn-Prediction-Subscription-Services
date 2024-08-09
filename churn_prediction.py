import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, preprocess_data

def predict_churn(df):
    X = df[['age', 'total_spent', 'num_logins']]
    y = df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    
    print(f'Churn Prediction Accuracy: {accuracy:.2f}')
    return model

if __name__ == "__main__":
    df = load_data('subscription_data.db')
    df = preprocess_data(df)
    model = predict_churn(df)
