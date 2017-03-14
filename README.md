# Grupo-Bimbo-inventory-demand
Place 13 solution (Private LB) for the inventory demand forecasting challenge on Kaggle https://www.kaggle.com/c/grupo-bimbo-inventory-demand

![Bakeries](grupobimbo_660x_crop.png)

## The main idea
- The dataset is huge, contains only a few features (which are all categorical with high cardinality) and has a time component
- To deal with that, we only train with the latest observations and use most of the past data for creating features that represent 
summary statistics (mean/median/count/std..) of the target (demand) given specific combinations of the categorical features 
- These out-of-sample features of past demand given specific clients, products and sales channels are highly relevant for predicting
future demand.

## To reconstruct the solution
- Put all data from Kaggle into data/
- run preprocc_train_submit.py for an optimized solution scoring place 13 on the private leaderboard