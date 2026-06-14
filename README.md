
# Information Retrieval System using Streamlit

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Dataset Download

```python
import kagglehub

path = kagglehub.dataset_download("yufengdev/bbc-fulltext-and-category")

print(path)
```

Use the CSV file from the downloaded dataset inside Streamlit.
