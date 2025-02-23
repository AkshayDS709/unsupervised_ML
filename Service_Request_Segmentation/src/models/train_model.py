import pandas as pd
from sklearn.cluster import KMeans

def train_model(input_path, output_path, num_clusters=10):
    """Trains a KMeans clustering model to segment service requests."""
    df = pd.read_csv(input_path)
    service_numbers = df['service_request_number']
    X = df.drop(columns=['service_request_number'])
    
    model = KMeans(n_clusters=num_clusters, random_state=42)
    df['group_number'] = model.fit_predict(X)

    # Save clustered data
    clustered_df = pd.DataFrame({'service_request_number': service_numbers, 'group_number': df['group_number']})
    clustered_df.to_csv(output_path, index=False)
    print("Clustering completed.")

if __name__ == "__main__":
    train_model("data/service_request_features.csv", "data/service_request_clusters.csv")
