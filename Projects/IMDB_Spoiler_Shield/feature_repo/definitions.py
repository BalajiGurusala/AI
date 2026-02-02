from feast import Entity, FeatureView, FileSource, ValueType, Field
from feast.types import Float32, String
from datetime import timedelta

# Define an entity for the movie
movie = Entity(name="movie_id", value_type=ValueType.STRING, description="The ID of the movie")

# Define the data source
movie_stats_source = FileSource(
    path="/opt/airflow/data/processed/movie_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define the Feature View
movie_stats_view = FeatureView(
    name="movie_stats",
    entities=[movie],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="duration_minutes", dtype=Float32),
        Field(name="rating", dtype=Float32),
        Field(name="genre", dtype=String),
        Field(name="avg_review_length", dtype=Float32),
        Field(name="avg_sentiment_score", dtype=Float32),
    ],
    online=True,
    source=movie_stats_source,
    tags={"team": "imdb_spoiler_shield"},
)