import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RetrievalBot:
    def __init__(self, dataset_path="Dataset_Chatbot.xlsx"):
        # Load dataset
        self.df = pd.read_excel(dataset_path)

        # Use QueryText (farmer question) and KccAns (answer)
        self.qa_pairs = self.df[['QueryText', 'KccAns']].dropna().reset_index(drop=True)

        # Vectorize queries
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.qa_pairs['QueryText'])

    def get_response(self, user_query: str) -> str:
        # Convert user query to vector
        query_vec = self.vectorizer.transform([user_query])

        # Compute cosine similarity
        similarity = cosine_similarity(query_vec, self.tfidf_matrix)

        # Find best match
        best_idx = similarity.argmax()
        best_score = similarity.max()

        # Confidence threshold
        if best_score < 0.3:
            return "❌ Sorry, I don’t have information on that. I can connect you to an expert."

        return self.qa_pairs.iloc[best_idx]['KccAns']


def main():
    bot = RetrievalBot("Dataset_Chatbot.xlsx")
    print("🌱 Farmer Support Retrieval Bot (type 'exit' to quit) 🌱\n")

    while True:
        user_query = input("👩‍🌾 You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("🤖 Bot: Goodbye! Stay safe in farming 🌾")
            break

        response = bot.get_response(user_query)
        print("🤖 Bot:", response, "\n")


if __name__ == "__main__":
    main()
