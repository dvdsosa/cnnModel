import weaviate

import weaviate.classes.config as wc
import json
import os

# Instantiate your client
client = weaviate.connect_to_local()
print(f"Your Weaviate client library version is: {weaviate.__version__}.")

try:
    # Work with the client here - e.g.:
    assert client.is_live()
    pass

finally:  # This will always be executed, even if an exception is raised
    client.close()  # Close the connection & release resources


#metainfo = client.get_meta()
#print(json.dumps(metainfo, indent=2))  # Print the meta information in a readable format

#sk-proj-F50EzqBpFwlxblxLdaLST3BlbkFJZKbUjd6nDifnNq5b6Miu

client.collections.create(
    name="Movie",
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="overview", data_type=wc.DataType.TEXT),
        wc.Property(name="vote_average", data_type=wc.DataType.NUMBER),
        wc.Property(name="genre_ids", data_type=wc.DataType.INT_ARRAY),
        wc.Property(name="release_date", data_type=wc.DataType.DATE),
        wc.Property(name="tmdb_id", data_type=wc.DataType.INT),
    ],
    # Define the vectorizer module
    vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
    # Define the generative module
    generative_config=wc.Configure.Generative.openai()
)

client.close()