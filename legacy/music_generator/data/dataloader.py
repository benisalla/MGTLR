import json

class DataLoader:
    def __init__(self, train_path, val_path):
        """Initializes the DataLoader with file paths for the training and validation datasets."""
        self.train_path = train_path
        self.val_path = val_path

    def load_data(self, file_path):
        """Loads JSON data from the specified file path."""
        with open(file_path, 'r') as file:
            return json.load(file)

    def preprocess_data(self, data, start_token="<SOS>", end_token="<EOS>", join_str="\n\n"):
        """Processes raw data by adding start and end tokens to each item and joining them with a specified delimiter."""
        processed = [f"{start_token}{item['abc notation']}{end_token}" for item in data]
        return join_str.join(processed)

    def get_data(self):
        """Loads and preprocesses both the training and validation data, returning them as formatted text."""
        train_data = self.load_data(self.train_path)
        val_data = self.load_data(self.val_path)
        
        train_text = self.preprocess_data(train_data)
        val_text = self.preprocess_data(val_data)
        
        return train_text, val_text