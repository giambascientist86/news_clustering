# --- src/main.py ---
if __name__ == "__main__":
    # Load Data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    
    # Preprocess Text
    processor = TextPreprocessor()
    df['clean_text'] = df['headline'].apply(processor.clean_text)
    
    # Vectorize Text
    vectorizer = TextVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    
    # Cluster Texts
    clusterer = NewsClustering(num_clusters=5)
    df['cluster'] = clusterer.fit(X)
    
    # Evaluate Clustering
    eval_results = clusterer.evaluate(X, df['category'])
    logging.info(f"Clustering Evaluation: {eval_results}")
