from sklearn2pmml import PMMLPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml import sklearn2pmml
import pandas


if __name__ == "__main__":
    iris_df = pandas.read_csv("Iris.csv")

    iris_pipeline = PMMLPipeline([
        ("classifier", RandomForestClassifier(n_estimators=5))
    ])
    labels = iris_df["Name"]
    features = iris_df[iris_df.columns.difference(["Name"])]

    iris_pipeline.fit(features, labels)

    sklearn2pmml(pipeline=iris_pipeline, pmml="RandomForestIris.pmml", with_repr=True)
