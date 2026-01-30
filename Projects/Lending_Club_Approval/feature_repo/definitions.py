# feature_repo/definitions.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# Define the Parquet file source
lending_source = FileSource(
    name="lending_source",
    path="../data/lending_club_cleaned.parquet",
    timestamp_field="event_timestamp",
)

# Define the entity
borrower = Entity(
    name="borrower", 
    join_keys=["borrower_id"], 
    description="Borrower entity for lending club loans"
)

# Define the feature view
lending_features = FeatureView(
    name="lending_features",
    entities=[borrower],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="annual_inc", dtype=Float64),
        Field(name="dti", dtype=Float64),
        Field(name="revol_bal", dtype=Float64),
        Field(name="revol_util", dtype=Float64),
        Field(name="open_acc", dtype=Float64),
        Field(name="total_acc", dtype=Float64),
        Field(name="mort_acc", dtype=Float64),
        Field(name="pub_rec", dtype=Float64),
        Field(name="pub_rec_bankruptcies", dtype=Float64),
        Field(name="earliest_cr_year", dtype=Int64),
        Field(name="home_ownership", dtype=String),
    ],
    source=lending_source,
    online=True,
    tags={"team": "risk_assessment"},
)
