import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(input_path, output_path):
    """Converts service request descriptions into numerical features using TF-IDF."""
    df = pd.read_csv(input_path)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    feature_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    feature_df['service_request_number'] = df['service_request_number']
    feature_df.to_csv(output_path, index=False)
    print("Feature extraction completed.")

if __name__ == "__main__":
    extract_features("data/service_requests.csv", "data/service_request_features.csv")
