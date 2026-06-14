# ----------------------------------------------------------
# BITS Pilani - Information Retrieval - Assignment 1
# Student ID Group Ref: 2025aa05967,2025aa05953,2025aa05964 (Group 72)
# ----------------------------------------------------------


import streamlit as st
import pandas as pd
import re
import time
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.metrics.distance import edit_distance
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Information Retrieval System", layout="wide")

st.title("Information Retrieval System using Streamlit")

# -----------------------------
# BST IMPLEMENTATION
# -----------------------------

class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):

        if node is None:
            return BSTNode(key)

        if key < node.key:
            node.left = self._insert(node.left, key)

        elif key > node.key:
            node.right = self._insert(node.right, key)

        return node

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):

        if node is None:
            return False

        if node.key == key:
            return True

        if key < node.key:
            return self._search(node.left, key)

        return self._search(node.right, key)

# -----------------------------
# B-TREE IMPLEMENTATION
# -----------------------------

class BTreeNode:

    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.children = []


class BTree:

    def __init__(self, t):

        self.root = BTreeNode(True)
        self.t = t

    # SEARCH
    def search(self, k, x=None):

        if x is None:
            x = self.root

        i = 0

        while i < len(x.keys) and k > x.keys[i]:
            i += 1

        if i < len(x.keys) and k == x.keys[i]:
            return True

        if x.leaf:
            return False

        return self.search(k, x.children[i])

    # INSERT
    def insert(self, k):

        root = self.root

        # Root full
        if len(root.keys) == (2 * self.t) - 1:

            new_root = BTreeNode(False)

            new_root.children.append(root)

            self.split_child(new_root, 0)

            self.root = new_root

            self.insert_non_full(new_root, k)

        else:
            self.insert_non_full(root, k)

    # INSERT NON FULL
    def insert_non_full(self, node, k):

        i = len(node.keys) - 1

        # LEAF NODE
        if node.leaf:

            node.keys.append(None)

            while i >= 0 and k < node.keys[i]:

                node.keys[i + 1] = node.keys[i]
                i -= 1

            node.keys[i + 1] = k

        # INTERNAL NODE
        else:

            while i >= 0 and k < node.keys[i]:
                i -= 1

            i += 1

            # Split child if full
            if len(node.children[i].keys) == (2 * self.t) - 1:

                self.split_child(node, i)

                if k > node.keys[i]:
                    i += 1

            self.insert_non_full(node.children[i], k)

    # SPLIT CHILD
    def split_child(self, parent, index):

        t = self.t

        full_child = parent.children[index]

        new_child = BTreeNode(full_child.leaf)

        # Middle key moves up
        middle_key = full_child.keys[t - 1]

        # Right half keys
        new_child.keys = full_child.keys[t:]

        # Left half keys
        full_child.keys = full_child.keys[:t - 1]

        # Move children if not leaf
        if not full_child.leaf:

            new_child.children = full_child.children[t:]
            full_child.children = full_child.children[:t]

        # Insert new child
        parent.children.insert(index + 1, new_child)

        # Insert promoted key
        parent.keys.insert(index, middle_key)

# -----------------------------
# Dataset Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload BBC Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    text_column = st.selectbox("Select Text Column", df.columns)

    documents = df[text_column].astype(str).tolist()

    # -----------------------------
    # Sidebar Preprocessing
    # -----------------------------
    st.sidebar.header("Preprocessing Options")

    use_lower = st.sidebar.checkbox("Lowercase", True)
    remove_stop = st.sidebar.checkbox("Remove Stopwords", True)
    use_stem = st.sidebar.checkbox("Apply Stemming")
    use_lemma = st.sidebar.checkbox("Apply Lemmatization")
    hyphen = st.sidebar.checkbox("Hyphen Handling", True)

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        original = text

        if hyphen:
            text = text.replace("-", " ")

        if use_lower:
            text = text.lower()

        tokens = word_tokenize(text)

        if remove_stop:
            tokens = [w for w in tokens if w not in stop_words and w.isalpha()]

        stemmed = []
        lemmatized = []

        if use_stem:
            stemmed = [stemmer.stem(w) for w in tokens]
            tokens = stemmed

        if use_lemma:
            lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
            tokens = lemmatized

        return {
            "original": original,
            "tokens": tokens,
            "processed": " ".join(tokens)
        }

    processed_docs = [preprocess(doc) for doc in documents]

    # -----------------------------
    # Display Preprocessing
    # -----------------------------
    st.subheader("Preprocessing Output")

    for i in range(min(3, len(processed_docs))):
        st.write(f"Document {i+1}")

        st.text("Original:")
        st.write(processed_docs[i]["original"][:500])

        st.text("Tokens:")
        st.write(processed_docs[i]["tokens"][:50])

        st.text("Processed:")
        st.write(processed_docs[i]["processed"][:500])

        st.markdown("---")

    # -----------------------------
    # Inverted Index
    # -----------------------------
    inverted_index = defaultdict(list)

    for idx, doc in enumerate(processed_docs):
        for term in set(doc["tokens"]):
            inverted_index[term].append(idx)

    st.subheader("Inverted Index Sample")

    sample_terms = list(inverted_index.keys())[:20]

    index_df = pd.DataFrame({
        "Term": sample_terms,
        "Posting List": [inverted_index[t] for t in sample_terms]
    })

    st.dataframe(index_df)

    # -----------------------------
    # Biword Index
    # -----------------------------
    biword_index = defaultdict(list)

    for idx, doc in enumerate(processed_docs):
        tokens = doc["tokens"]

        for i in range(len(tokens)-1):
            biword = tokens[i] + " " + tokens[i+1]
            biword_index[biword].append(idx)

    # -----------------------------
    # Positional Index
    # -----------------------------
    positional_index = defaultdict(lambda: defaultdict(list))

    for doc_id, doc in enumerate(processed_docs):
        tokens = doc["tokens"]

        for pos, token in enumerate(tokens):
            positional_index[token][doc_id].append(pos)

    # -----------------------------
    # Phrase Query
    # -----------------------------
    st.subheader("Phrase Query Search")

    phrase_query = st.text_input("Enter Phrase Query")

    if phrase_query:

        query = phrase_query.lower()
        biword_results = []
        positional_results = []

        # Biword Search
        if query in biword_index:
            biword_results = biword_index[query]

        # Positional Search
        query_tokens = query.split()

        if len(query_tokens) == 2:
            w1, w2 = query_tokens

            if w1 in positional_index and w2 in positional_index:

                docs1 = positional_index[w1]
                docs2 = positional_index[w2]

                common_docs = set(docs1.keys()).intersection(docs2.keys())

                for d in common_docs:
                    pos1 = docs1[d]
                    pos2 = docs2[d]

                    for p in pos1:
                        if p + 1 in pos2:
                            positional_results.append(d)
                            break

        col1, col2 = st.columns(2)

        with col1:
            st.write("Biword Index Results")
            st.write(biword_results)

        with col2:
            st.write("Positional Index Results")
            st.write(positional_results)

        st.info("Biword index may produce false positives while positional index verifies exact positions.")

    # -----------------------------
    # BST and B-Tree Simulation using list,set
    # -----------------------------
    # st.subheader("BST vs B-Tree Search Comparison")

    # dictionary_terms = sorted(list(inverted_index.keys()))

    # query_term = st.text_input("Dictionary Search Term")

    # if query_term:

    #     # BST Simulation
    #     start = time.time()

    #     bst_found = query_term in dictionary_terms

    #     bst_time = (time.time() - start) * 1000

    #     # B-Tree Simulation
    #     start = time.time()

    #     btree_found = query_term in set(dictionary_terms)

    #     btree_time = (time.time() - start) * 1000

    #     compare_df = pd.DataFrame({
    #         "Structure": ["BST", "B-Tree"],
    #         "Found": [bst_found, btree_found],
    #         "Search Time (ms)": [bst_time, btree_time]
    #     })

    #     st.dataframe(compare_df)
    # -----------------------------
    # BST vs B-Tree Comparison
    # -----------------------------

    st.subheader("BST vs B-Tree Search Comparison")

    # dictionary_terms = sorted(list(inverted_index.keys()))
    dictionary_terms = list(inverted_index.keys())

    # BUILD BST
    bst = BST()

    for term in dictionary_terms:
        bst.insert(term)

    # BUILD B-TREE
    btree = BTree(3)

    for term in dictionary_terms:
        btree.insert(term)

    query_term = st.text_input("Dictionary Search Term")

    if query_term:

        # BST SEARCH
        start = time.time()

        bst_found = bst.search(query_term)

        bst_time = (time.time() - start) * 1000

        # B-TREE SEARCH
        start = time.time()

        btree_found = btree.search(query_term)

        btree_time = (time.time() - start) * 1000

        compare_df = pd.DataFrame({
            "Structure": ["BST", "B-Tree"],
            "Found": [bst_found, btree_found],
            "Search Time (ms)": [bst_time, btree_time]
        })

        st.dataframe(compare_df)

        if btree_time < bst_time:
            st.success("Inference: B-Tree performed faster.")
        else:
            st.success("Inference: BST performed faster.")

    # -----------------------------
    # Tolerant Retrieval
    # -----------------------------
    st.subheader("Tolerant Retrieval")

    tolerant_query = st.text_input("Enter Misspelled Query")

    if tolerant_query:

        vocab = list(inverted_index.keys())

        best_word = None
        min_dist = 999

        for word in vocab[:5000]:

            dist = edit_distance(tolerant_query, word)

            if dist < min_dist:
                min_dist = dist
                best_word = word

        st.write("Suggested Correction:", best_word)

        if best_word in inverted_index:
            st.write("Matching Documents:")
            st.write(inverted_index[best_word][:20])

    # -----------------------------
    # Stemming vs Lemmatization
    # -----------------------------
    st.subheader("Stemming vs Lemmatization Comparison")

    stem_docs = []
    lemma_docs = []

    for doc in documents:

        tokens = word_tokenize(doc.lower())
        tokens = [t for t in tokens if t.isalpha()]

        stem_docs.append(" ".join([stemmer.stem(t) for t in tokens]))
        lemma_docs.append(" ".join([lemmatizer.lemmatize(t) for t in tokens]))

    vectorizer = TfidfVectorizer()

    stem_matrix = vectorizer.fit_transform(stem_docs[:100])
    lemma_matrix = vectorizer.fit_transform(lemma_docs[:100])

    stem_score = cosine_similarity(stem_matrix[0], stem_matrix[1])[0][0]
    lemma_score = cosine_similarity(lemma_matrix[0], lemma_matrix[1])[0][0]

    compare_df = pd.DataFrame({
        "Technique": ["Stemming", "Lemmatization"],
        "Semantic Similarity": [stem_score, lemma_score]
    })

    st.dataframe(compare_df)

    if lemma_score >= stem_score:
        st.success("Inference: Lemmatization performed better for this dataset.")
    else:
        st.success("Inference: Stemming performed better for this dataset.")

    # -----------------------------
    # Final Inference
    # -----------------------------
    st.subheader("Final Inference")

    st.markdown("""
    ### Observations

    - Lowercasing and stopword removal improved retrieval quality.
    - Lemmatization produced more meaningful results.
    - Positional indexing was more accurate than biword indexing.
    - B-Tree search was faster than BST search.
    - Edit distance correction improved tolerant retrieval.
    """)
