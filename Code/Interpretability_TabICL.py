base_dataset = "SpurDike"

import pandas as pd
# from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
model = "TabICL"

# Read the tab-separated file
dfSD = pd.read_csv("../Data/SpurDike.tsv", sep="\t")
print(dfSD.shape)
# Show first rows
print(dfSD.head())
dfSD["Source"].value_counts()

selected_Source = ["Dey&Barbhuiya2005","Nasrollahi2008","Coleman2003","Pandey2016","Lim1997"]

dfSD_filtered = dfSD[dfSD["Source"].isin(selected_Source)]
print(dfSD_filtered.shape)

X, y = dfSD_filtered.drop(columns=["dsl", "Source"]), dfSD_filtered["dsl"]

feature_names = X.columns.to_numpy()
print(feature_names)


# from tabpfn_extensions import interpretability
import shap

feature_names = X.columns.to_numpy()

# reg = TabPFNRegressor()
reg = TabICLRegressor()

reg.fit(X, y)

explainer = shap.Explainer(
    reg.predict,
    X
)

shap_values = explainer(X)

#Save shap_values
import pickle

with open(f"../Results/shap_values_{base_dataset}_all_{model}.pkl", "wb") as f:
    pickle.dump(shap_values, f)
