import data_loader  
import preprocessing  
import model  
import visualization  
import joblib
import shap

def print_section_title(title):
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50 + "\n")

def print_status_message(message, status="..."):
    icons = {
        "loading": "📦",
        "success": "✅",
        "processing": "⚙️",
        "saving": "💾",
        "visualizing": "📊",
    }
    icon = icons.get(status, "ℹ️")
    print(f"{icon} {message}")

def pipeline():  
    # Load data  
    print_section_title("Step 1: Loading Data")
    print_status_message("Loading frequency and generation data...", "loading")
    frequency_data, generation_data = data_loader.load_data()  
    print_status_message("Data loaded successfully.", "success")
  
    # Get events  
    print_section_title("Step 2: Getting Events")
    print_status_message("Identifying positive and negative events...", "processing")
    positive_events, negative_events = preprocessing.get_events(frequency_data)  
    print_status_message("Events identified successfully.", "success")
  
    # Preprocess data  
    print_section_title("Step 3: Preprocessing Data")
    print_status_message("Merging and preprocessing data...", "processing")
    merged_data = preprocessing.preprocess_data(frequency_data, generation_data, positive_events, negative_events)  
    print_status_message("Data preprocessed successfully.", "success")
  
    # Create features 
    print_section_title("Step 4: Creating Features")
    print_status_message("Generating features from data...", "processing")
    filtered_data = preprocessing.create_features(merged_data)  
    print_status_message("Features created successfully.", "success")
  
    # Split data  
    print_section_title("Step 5: Splitting Data")
    print_status_message("Splitting data into training, validation, and test sets...", "processing")
    X_train, y_train, X_validate, y_validate, X_test, y_test = preprocessing.split_data(filtered_data)  
    print_status_message("Data split successfully.", "success")

    # Train and evaluate model  
    print_section_title("Step 6: Training and Evaluating Model")
    print_status_message("Training and evaluating the model...", "processing")
    classifier, baseline_model = model.train_and_evaluate(X_train, y_train, X_validate, y_validate, X_test, y_test)  
    print_status_message("Model trained and evaluated successfully.", "success")

    # Compute SHAP values
    print_section_title("Step 7: Computing SHAP Values")
    print_status_message("Calculating SHAP values...", "processing")
    shap_values = model.compute_shap_values(classifier, X_test, y_test)
    print_status_message("SHAP values computed successfully.", "success")

    # Save SHAP values
    print_section_title("Step 8: Saving SHAP Values")
    print_status_message("Saving SHAP values to disk...", "saving")
    joblib.dump(shap_values, "data/shap_values.lib")
    print_status_message("SHAP values saved successfully.", "success")

    # Visualize results  
    print_section_title("Step 9: Visualizing Results")
    print_status_message("Generating visualizations for model performance...", "visualizing")
    visualization.visualize_results(classifier, X_test, y_test, filtered_data)
    print_status_message("Visualizations created successfully.", "success")

    return classifier, baseline_model, shap_values, X_train, y_train, X_validate, y_validate, X_test, y_test  
  
if __name__ == "__main__":  
    pipeline()
