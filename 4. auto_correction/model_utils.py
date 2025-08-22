import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    print("Loading data and model...")
    
    try:
        df = pd.read_csv('C:/Users/فاطمة الزهراني/Desktop/ابحاث/الداون/Codes/Real Codes/01_data_processing/processed_dataset.csv')
        print(f"Loaded {len(df)} samples")
        
        model = None
        model_path = 'simplified_high_accuracy_model.pkl'
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            print("Model saved as dictionary")
            if 'model' in model_data:
                model = model_data['model']
                print("Model extracted successfully")
            else:
                print("Model not found in dictionary")
                model = create_simple_model(df)
        else:
            model = model_data
            print("Model loaded directly")
        
        return df, model
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def create_simple_model(df):
    print("Creating simple test model...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    feature_columns = [col for col in df.columns if col not in ['word', 'speaker', 'file_path']]
    X = df[feature_columns]
    y = df['word']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Alternative model created")
    return model