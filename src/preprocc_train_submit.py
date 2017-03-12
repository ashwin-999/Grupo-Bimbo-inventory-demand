# Preprocessing

import os, matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', 50)
import numpy as np
import xgboost as xgb
import time
from sklearn.cross_validation import train_test_split

np.random.seed(1337)

def client_anaylsis(client):
    """
    The idea here is to unify the client ID of several different customers to more broad categories.
    """
    # clean duplicate spaces in client names
    client["NombreCliente"] = client["NombreCliente"].str.lower()
    client["NombreCliente"] = client["NombreCliente"].apply(lambda x: " ".join(x.split()))
    client = client.drop_duplicates(subset="Cliente_ID")
    special_list = ["^(yepas)\s.*", "^(oxxo)\s.*", "^(bodega\scomercial)\s.*", "^(bodega\saurrera)\s.*", "^(bodega)\s.*",
                    "^(woolwort|woolworth)\s.*", "^(zona\sexpress)\s.*",
                    "^(zacatecana)\s.*", "^(yza)\s.*",
                    "^(yanet)\s.*", "^(yak)\s.*",
                    "^(wings)\s.*", "^(wendy)\s.*", "^(walmart\ssuper)\s?.*", "^(waldos)\s.*",
                    "^(wal\smart)\s.*", "^(vulcanizadora)\s.*", "^(viveres\sy\sservicios)\s.*",
                    "^(vips)\s.*", "^(vinos\sy\slicores)\s.*", "^(tienda\ssuper\sprecio)\s.*",
                    "^(vinos\sy\sabarrotes)\s.*", "^(vinateria)\s.*", "^(video\sjuegos)\s.*", "^(universidad)\s.*",
                    "^(tiendas\stres\sb)\s.*", "^(toks)\s.*","^(tkt\ssix)\s.*",
                    "^(torteria)\s.*", "^(tortas)\s.*", "^(super\sbara)\s.*",
                    "^(tiendas\sde\ssuper\sprecio)\s.*", "^(ultramarinos)\s.*", "^(tortilleria)\s.*",
                    "^(tienda\sde\sservicio)\s.*", "^(super\sx)\s.*", "^(super\swillys)\s.*",
                    "^(super\ssanchez)\s.*", "^(super\sneto)\s.*", "^(super\skompras)\s.*",
                    "^(super\skiosco)\s.*", "^(super\sfarmacia)\s.*", "^(super\scarnes)\s.*",
                    "^(super\scarniceria)\s.*", "^(soriana)\s.*", "^(super\scenter)\s.*",
                    "^(solo\sun\sprecio)\s.*", "^(super\scity)\s.*", "^(super\sg)\s.*", "^(super\smercado)\s.*",
                    "^(sdn)\s.*", "^(sams\sclub)\s.*", "^(papeleria)\s.*", "^(multicinemas)\s.*",
                    "^(mz)\s.*", "^(motel)\s.*", "^(minisuper)\s.*", "^(mini\stienda)\s.*",
                    "^(mini\ssuper)\s.*", "^(mini\smarket)\s.*", "^(mini\sabarrotes)\s.*", "^(mi\sbodega)\s.*",
                    "^(merza|merzapack)\s.*", "^(mercado\ssoriana)\s.*", "^(mega\scomercial)\s.*",
                    "^(mc\sdonalds)\s.*", "^(mb)\s[^ex].*", "^(maquina\sfma)\s.*", "^(ley\sexpress)\s.*",
                    "^(lavamatica)\s.*", "^(kiosko)\s.*", "^(kesos\sy\skosas)\s.*", "^(issste)\s.*",
                    "^(hot\sdogs\sy\shamburguesas|)\s.*", "^(hamburguesas\sy\shot\sdogs)\s.*", "(hot\sdog)",
                    "^(hospital)\s.*", "^(hiper\ssoriana)\s.*", "^(super\sahorros)\s.*", "^(super\sabarrotes)\s.*",
                    "^(hambuerguesas|hamburguesas|hamburgesas)\s.*", "^(gran\sbodega)\s.*",
                    "^(gran\sd)\s.*", "^(go\smart)\s.*", "^(gasolinera)\s.*", "^(fundacion)\s.*",
                    "^(fruteria)\s.*", "^(frutas\sy\sverduras)\s.*", "^(frutas\sy\slegumbres)\s.*",
                    "^(frutas\sy\sabarrotes)\s.*", "^(fma)\s.*", "^(fiesta\sinn)\s.*", "^(ferreteria)\s.*",
                    "^(farmacon)\s.*", "^(farmacias)\s.*", "^(farmacia\syza)\s.*",
                    "^(farmacia\smoderna)\s.*", "^(farmacia\slopez)\s.*",
                    "^(farmacia\sissste)\s.*", "^(farmacia\sisseg)\s.*", "^(farmacia\sguadalajara)\s.*",
                    "^(farmacia\sesquivar)\s.*", "^(farmacia\scalderon)\s.*", "^(farmacia\sbenavides)\s.*",
                    "^(farmacia\sabc)\s.*", "^(farmacia)\s.*", "^(farm\sguadalajara)\s.*",
                    "^(facultad\sde)\s.*", "^(f\sgdl)\s.*", "^(expendio)\s.*", "^(expendio\sde\span)\s.*",
                    "^(expendio\sde\shuevo)\s.*", "^(expendio\sbimbo)\s.*", "^(expendedoras\sautomaticas)\s.*",
                    "^(estic)\s.*", "^(estancia\sinfantil)\s.*", "^(estacionamiento)\s.*", "^(estanquillo)\s.*",
                    "^(estacion\sde\sservicio)\s.*", "^(establecimientos?)\s.*",
                    "^(escuela\suniversidad|esc\suniversidad)\s.*", "^(escuela\stelesecundaria|esc\stelesecundaria)\s.*",
                    "^(escuela\stecnica|esc\stecnica)\s.*",
                    "^(escuela\ssuperior|esc\ssuperior)\s.*", "^(escuela\ssecundaria\stecnica|esc\ssecundaria\stecnica)\s.*",
                    "^(escuela\ssecundaria\sgeneral|esc\ssecundaria\sgeneral)\s.*",
                    "^(escuela\ssecundaria\sfederal|esc\ssecundaria\sfederal)\s.*",
                    "^(escuela\ssecundaria|esc\ssecundaria)\s.*", "^(escuela\sprimaria|esc\sprimaria)\s.*",
                    "^(escuela\spreparatoria|esc\spreparatoria)\s.*", "^(escuela\snormal|esc\snormal)\s.*",
                    "^(escuela\sinstituto|esc\sinstituto)\s.*", "^(esc\sprepa|esc\sprep)\s.*",
                    "^(escuela\scolegio|esc\scolegio)\s.*", "^(escuela|esc)\s.*", "^(dunosusa)\s.*",
                    "^(ferreteria)\s.*", "^(dulces)\s.*", "^(dulceria)\s.*", "^(dulce)\s.*", "^(distribuidora)\s.*",
                    "^(diconsa)\s.*", "^(deposito)\s.*", "^(del\srio)\s.*", "^(cyber)\s.*", "^(cremeria)\s.*",
                    "^(cosina\seconomica)\s.*", "^(copy).*", "^(consumo|consumos)\s.*","^(conalep)\s.*",
                    "^(comercializadora)\s.*", "^(comercial\ssuper\salianza)\s.*",
                    "^(comercial\smexicana)\s.*", "^(comedor)\s.*", "^(colegio\sde\sbachilleres)\s.*",
                    "^(colegio)\s.*", "^(coffe).*", "^(cocteleria|cockteleria)\s.*", "^(cocina\seconomica)\s.*",
                    "^(cocina)\s.*", "^(cobaev)\s.*", "^(cobaes)\s.*", "^(cobaeh)\s.*", "^(cobach)\s.*",
                    "^(club\sde\sgolf)\s.*", "^(club\scampestre)\s.*", "^(city\sclub)\s.*", "^(circulo\sk)\s.*",
                    "^(cinepolis)\s.*", "^(cinemex)\s.*", "^(cinemas)\s.*", "^(cinemark)\s.*", "^(ciber)\s.*",
                    "^(church|churchs)\s.*", "^(chilis)\s.*", "^(chiles\sy\ssemillas)\s.*", "^(chiles\ssecos)\s.*",
                    "^(chedraui)\s.*", "^(cetis)\s.*", "^(cervefrio)\s.*", "^(cervefiesta)\s.*",
                    "^(cerveceria)\s.*", "^(cervecentro)\s.*", "^(centro\sescolar)\s.*", "^(centro\seducativo)\s.*",
                    "^(centro\sde\sestudios)\s.*", "^(centro\scomercial)\s.*", "^(central\sde\sautobuses)\s.*",
                    "^(cecytem)\s.*", "^(cecytec)\s.*", "^(cecyte)\s.*", "^(cbtis)\s.*", "^(cbta)\s.*", "^(cbt)\s.*",
                    "^(caseta\stelefonica)\s.*", "^(caseta)\s.*", "^(casa\sley)\s.*", "^(casa\shernandez)\s.*",
                    "^(cartonero\scentral)\s.*", "^(carniceria)\s.*", "^(carne\smart)\s.*", "^(calimax)\s.*",
                    "^(cajero)\s.*", "^(cafeteria)\s.*", "^(cafe)\s.*", "^(burritos)\s.*",
                    "^(burguer\sking|burger\sking)\s.*", "^(bip)\s.*", "^(bimbo\sexpendio)\s.*",
                    "^(burguer|burger)\s.*", "^(ba.os)\s.*", "^(bae)\s.*", "^(bachilleres)\s.*", "^(bachillerato)\s.*",
                    "^(autosercivio|auto\sservicio)\s.*", "^(autolavado|auto\slavado)\s.*",
                    "^(autobuses\sla\spiedad|autobuses\sde\sla\piedad)\s.*", "^(arrachera)\s.*",
                    "^(alsuper\sstore)\s.*", "^(alsuper)\s.*", "^(academia)\s.*", "^(abts)\s.*",
                    "^(abarrotera\slagunitas)\s.*", "^(abarrotera)\s.*", "^(abarrotes\sy\svinos)\s.*",
                    "^(abarrotes\sy\sverduras)\s.*",  "^(abarrotes\sy\ssemillas)\s.*",
                    "^(abarrotes\sy\spapeleria)\s.*", "^(abarrotes\sy\snovedades)\s.*", "^(abarrotes\sy\sfruteria)\s.*",
                    "^(abarrotes\sy\sdeposito)\s.*", "^(abarrotes\sy\scremeria)\s.*", "^(abarrotes\sy\scarniceria)\s.*",
                    "^(abarrotes\svinos\sy\slicores)\s.*", "^(abarrote|abarrotes|abarotes|abarr|aba|ab)\s.*",
                    "^(7\seleven)\s.*", "^(7\s24)\s.*"]

    client["NombreCliente2"] = client["NombreCliente"]
    for var in special_list:
        client[var] = client["NombreCliente"].str.extract(var, expand=False).str.upper()
        replace = client.loc[~client[var].isnull(), var]
        client.loc[~client[var].isnull(),"NombreCliente2"] = replace
        client.drop(var, axis=1, inplace=True)
    return client["NombreCliente2"]

def client_anaylsis2(client):
    """
    The idea here is to unify the client ID of several different customers to more broad categories in another
    different way
    """
    # clean duplicate spaces in client names
    client["NombreCliente"] = client["NombreCliente"].str.upper()
    client["NombreCliente"] = client["NombreCliente"].apply(lambda x: " ".join(x.split()))
    client = client.drop_duplicates(subset="Cliente_ID")

    # --- Begin Filtering for specific terms
    # Note that the order of filtering is significant.
    # For example:
    # The regex of .*ERIA.* will assign "FRUITERIA" to 'Eatery' rather than 'Fresh Market'.
    # In other words, the first filters to occur have a bigger priority.

    def filter_specific(vf2):
        # Known Large Company / Special Group Types
        vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*REMISION.*', 'Consignment')
        vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*WAL MART.*', '.*SAMS CLUB.*'], 'Walmart', regex=True)
        vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*OXXO.*', 'Oxxo Store')
        vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*CONASUPO.*', 'Govt Store')
        vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*BIMBO.*', 'Bimbo Store')

        # General term search for a random assortment of words I picked from looking at
        # their frequency of appearance in the data and common spanish words for these categories
        vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*COLEG.*', '.*UNIV.*', '.*ESCU.*', '.*INSTI.*', \
                                                             '.*PREPAR.*'], 'School', regex=True)
        vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*PUESTO.*', 'Post')
        vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*FARMA.*', '.*HOSPITAL.*', '.*CLINI.*', '.*BOTICA.*'],
                                                            'Hospital/Pharmacy', regex=True)
        vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*CAFE.*', '.*CREMERIA.*', '.*DULCERIA.*', \
                                                             '.*REST.*', '.*BURGER.*', '.*TACO.*', '.*TORTA.*', \
                                                             '.*TAQUER.*', '.*HOT DOG.*', '.*PIZZA.*' \
                                                             '.*COMEDOR.*', '.*ERIA.*', '.*BURGU.*'], 'Eatery',
                                                            regex=True)
        vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*SUPER.*', 'Supermarket')
        vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*COMERCIAL.*', '.*BODEGA.*', '.*DEPOSITO.*', \
                                                             '.*ABARROTES.*', '.*MERCADO.*', '.*CAMBIO.*', \
                                                             '.*MARKET.*', '.*MART .*', '.*MINI .*', \
                                                             '.*PLAZA.*', '.*MISC.*', '.*ELEVEN.*', '.*EXP.*', \
                                                             '.*SNACK.*', '.*PAPELERIA.*', '.*CARNICERIA.*', \
                                                             '.*LOCAL.*', '.*COMODIN.*', '.*PROVIDENCIA.*'
                                                             ], 'General Market/Mart' \
                                                            , regex=True)

        vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*VERDU.*', '.*FRUT.*'], 'Fresh Market', regex=True)
        vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*HOTEL.*', '.*MOTEL.*', ".*CASA.*"], 'Hotel', regex=True)
    filter_specific(client)

    # --- Begin filtering for more general terms
    # The idea here is to look for names with particles of speech that would
    # not appear in a person's name.
    # i.e. "Individuals" should not contain any participles or numbers in their names.
    def filter_participle(vf2):
        vf2['NombreCliente'] = vf2['NombreCliente'].replace([
            '.*LA .*', '.*EL .*', '.*DE .*', '.*LOS .*', '.*DEL .*', '.*Y .*', '.*SAN .*', '.*SANTA .*', \
            '.*AG .*', '.*LAS .*', '.*MI .*', '.*MA .*', '.*II.*', '.*[0-9]+.*' \
            ], 'Small Franchise', regex=True)
    filter_participle(client)

    # Any remaining entries should be "Individual" Named Clients, there are some outliers.
    # More specific filters could be used in order to reduce the percentage of outliers in this final set.
    def filter_remaining(vf2):
        def function_word(data):
            # Avoid the single-words created so far by checking for upper-case
            if (data.isupper()) and (data != "NO IDENTIFICADO"):
                return 'Individual'
            else:
                return data

        vf2['NombreCliente'] = vf2['NombreCliente'].map(function_word)
    filter_remaining(client)

    return client["NombreCliente"]

def preprocess():
    dtype_dict = {"Semana": np.uint8, 'Agencia_ID': np.uint16, 'Canal_ID': np.uint8,
                  'Ruta_SAK': np.uint16, 'Cliente_ID': np.uint32, 'Producto_ID': np.uint16,
                  'Demanda_uni_equil': np.uint32, "Venta_hoy": np.float32, "Venta_uni_hoy": np.uint32,
                  "Dev_uni_proxima": np.uint32, "Dev_proxima": np.float32}

    train = pd.read_csv("train.csv.zip", compression="zip", dtype=dtype_dict)
    test = pd.read_csv("test.csv.zip", compression="zip", dtype=dtype_dict)

    mean_dataframes = {}
    mean_dataframes["test"] = train.copy()
    mean_dataframes["train"] = train[train["Semana"]<9].copy()

    print("complete train obs: {}".format(len(train)))
    print("train week 9 obs: {}".format(len(train[train["Semana"] == 9])))
    train = train[train["Semana"] == 9]

    # not used in later stages. Was used to find the right hyperparameters for XGBoost. After finding them and to
    # obtain the best solution the evaluation data was incorporated into the training data and the hyperparameters
    # were used "blindly"
    # eval = train.iloc[int(len(train) * 0.75):, :].copy()
    # print("eval obs: {}".format(len(eval)))
    # mean_dataframes["eval"] = mean_dataframes["test"].iloc[:eval.index.min(), :].copy()
    # train = train.iloc[:int(len(train) * 0.75), :]
    # print("train obs: {}".format(len(train)))

    # read data files and create new client ids
    town = pd.read_csv("../data/town_state.csv.zip", compression="zip", delimiter=";")
    product = pd.read_csv("../data/producto_tabla.csv.zip", compression="zip")
    client = pd.read_csv("../data/cliente_tabla.csv.zip", compression="zip")
    client["NombreCliente2"] = client_anaylsis(client)
    client["NombreCliente3"] = client_anaylsis(client)

    # Feature Engineering
    prod_split = product.NombreProducto.str.split(r"(\s\d+\s?(kg|Kg|g|G|in|ml|pct|p|P|Reb))")
    product["product"] = prod_split.apply(lambda x: x[0])
    product["brand2"] = product.NombreProducto.str.extract("^.+\s(\D+) \d+$", expand=False)
    product['brand'] = prod_split.apply(lambda x: x[-1]).str.split().apply(lambda x: x[:-1])
    product['num_brands'] = product.brand.apply(lambda x: len(x))
    product['brand'] = prod_split.apply(lambda x: x[-1]).str.split().apply(lambda x: x[:-1]).astype("str")
    product['short_name'] = product['product'].str.split(r'[A-Z][A-Z]').apply(lambda x: x[0])
    product["beverage"] = product.NombreProducto.str.extract("\d+(ml)", expand=False)
    product.loc[product["beverage"].notnull(), "beverage"] = 1
    product["beverage"] = pd.to_numeric(product["beverage"])
    product["beverage"] = product["beverage"].fillna(0)
    w = product.NombreProducto.str.extract("(\d+)(kg|Kg|g|G|ml)", expand=True)
    product["weight"] = w[0].astype("float") * w[1].map({"kg": 1000, "Kg": 1000, "G": 1, "g": 1, "ml": 1})
    product["pieces"] = product.NombreProducto.str.extract("(\d+)p\s", expand=False).astype("float")
    product["weight_per_piece"] = product["weight"].fillna(0) / product["pieces"].fillna(1)
    product.loc[product["short_name"] == "", "short_name"] = product.loc[product["short_name"] == "", "product"]
    product.drop(["NombreProducto", "product"], axis=1, inplace=True)

    # Drop duplicate clients
    client = client.drop_duplicates(subset="Cliente_ID")
    # clean duplicate spaces in client names
    client["NombreCliente"] = client["NombreCliente"].apply(lambda x: " ".join(x.split()))

    # Join everything
    dataset_list = ["train", "test"]
    for dataset in dataset_list:
        mean_dataframes[dataset] = mean_dataframes[dataset].merge(town, how="left", on="Agencia_ID")
        mean_dataframes[dataset] = mean_dataframes[dataset].merge(product, how="left", on="Producto_ID")
        mean_dataframes[dataset] = mean_dataframes[dataset].merge(client, how="left", on="Cliente_ID")
    train = train.merge(town, how="left", on="Agencia_ID")
    train = train.merge(product, how="left", on="Producto_ID")
    train = train.merge(client, how="left", on="Cliente_ID")
    # eval = eval.merge(town, how="left", on="Agencia_ID")
    # eval = eval.merge(product, how="left", on="Producto_ID")
    # eval = eval.merge(client, how="left", on="Cliente_ID")
    test = test.merge(town, how="left", on="Agencia_ID")
    test = test.merge(product, how="left", on="Producto_ID")
    test = test.merge(client, how="left", on="Cliente_ID")

    rename_dict = {"Semana": "week", "Agencia_ID": "sales_depot_id", "Canal_ID": "sales_channel_id",
                          "Ruta_SAK": "route_id", "Town": "town", "State": "state",
                          "Cliente_ID": "client_id", "NombreCliente": "client_name", "Producto_ID": "product_id",
                          "NombreProducto": "product_name", "Demanda_uni_equil": "target",
                          "Venta_uni_hoy": "sales_unit_this_week", "Venta_hoy": "sales_this_week",
                          "Dev_uni_proxima": "returns_unit_next_week", "Dev_proxima": "returns_next_week",
                          "NombreCliente2": "client_name2"}

    # rename columns for convenience
    for dataset in dataset_list:
        mean_dataframes[dataset].rename(columns=rename_dict, inplace=True)
    train.rename(columns=rename_dict, inplace=True)
    # eval.rename(columns=rename_dict, inplace=True)
    test.rename(columns=rename_dict, inplace=True)

    # transform target demand to log scale
    for dataset in dataset_list:
        mean_dataframes[dataset]["log_demand"] = np.log1p(mean_dataframes[dataset]["target"])
    train["log_demand"] = np.log1p(train["target"])
    # eval["log_demand"] = np.log1p(eval["target"])
    train_target = train["log_demand"]
    # eval_target = eval["log_demand"]
    train.drop(["target", "log_demand", "sales_unit_this_week", "sales_this_week", "returns_unit_next_week",
                "returns_next_week"], axis=1, inplace=True)
    # eval.drop(["target", "log_demand", "sales_unit_this_week", "sales_this_week", "returns_unit_next_week",
    #             "returns_next_week"], axis=1, inplace=True)

    #TODO Add only_in_train mean vars
    only_in_train = ["sales_unit_this_week", "sales_this_week", "returns_unit_next_week", "returns_next_week"]

    def get_mean(mean_dataset, dataset, columns, target):
        tempTable = mean_dataset[columns + target].groupby(columns).agg(
            ["mean", "std", "median"])[target[0]]
        name = "_".join(columns)
        tempTable = tempTable.rename(columns={
            "count": target[0] + "_count_" + name,
            "mean": target[0] + "_mean_" + name,
            "std": target[0] + "_std_" + name,
            "sum": target[0] + "_sum_" + name,
            "median": target[0] + "_median_" + name})
        tempTable.reset_index(inplace=True)
        dataset = pd.merge(dataset, tempTable, how='left', on=columns)
        return dataset

    column_combinations = [["short_name", "client_id", "sales_channel_id"],
                           ["short_name", "town"], ["route_id", "client_name3", "sales_channel_id", "town", "short_name"],
                           ["product_id", "client_name3", "town", "route_id", "short_name"],
                           ["product_id", "client_name3", "town", "short_name", "route_id"],
                           ["short_name", "client_id", "sales_depot_id", "sales_channel_id", "route_id"],
                           ["product_id", "client_id", "sales_depot_id", "sales_channel_id", "route_id"],
                           ["short_name", "client_id", "town"], ["client_name3", "short_name"],
                           ["client_name3", "short_name", "sales_depot_id"],
                           ["product_id", "client_name3", "sales_depot_id", "short_name", "route_id"],
                           ["client_name3", "short_name", "product_id"],
                           ["client_name3", "short_name", "route_id"],
                           ["client_name2", "short_name", "product_id"], ["client_name2", "short_name", "route_id"],
                           ["product_id", "client_id", "route_id", "short_name", "sales_depot_id"],
                           ["product_id", "client_id", "route_id", "short_name"],
                           ["product_id", "client_id", "sales_depot_id", "short_name"],
                           ["route_id", "product_id", "short_name"], ["route_id", "client_id", "short_name"],
                           ["product_id", "client_id", "short_name"], ["product_id", "short_name"],
                           ["short_name", "sales_depot_id"], ["short_name", "client_id", "sales_depot_id"],
                           ["route_id", "client_id"], ["route_id", "short_name"], ["client_name2", "short_name"],
                           ["product_id", "route_id"], ["product_id", "client_id", "sales_depot_id"],
                           ["product_id", "client_id"], ["product_id", "client_id", "sales_depot_id", "route_id"],
                           ["product_id", "client_id", "route_id"], ["product_id", "client_name"],
                           ["short_name", "client_id"]]

    for columns in column_combinations:
        train = get_mean(mean_dataframes["train"], train, columns, "log_demand")
        test = get_mean(mean_dataframes["test"], test, columns, "log_demand")

    train['null_count'] = train.isnull().sum(axis=1).tolist()
    # eval['null_count'] = eval.isnull().sum(axis=1).tolist()
    test['null_count'] = test.isnull().sum(axis=1).tolist()

    for feat in ["sales_depot_id", "sales_channel_id", "route_id", "town", "state", "client_id", "client_name",
                 "client_name2", "client_name3", "product_id", "brand", "brand2", "short_name"]:
        for dataset in dataset_list:
            mean_train = mean_dataframes[dataset]
            # LOG DEMAND MEANS
            tempTable = mean_train[[feat, "log_demand"]].groupby(feat).agg(["count", "mean", "std", "sum",
                                                                            "median"]).log_demand
            tempTable = tempTable.rename(
                columns={"count": "count_"+feat, "mean": "mean_"+feat, "std": "sd_"+feat, "sum": "sum_"+feat,
                         "median": "median_" + feat})
            tempTable.reset_index(inplace=True)
            if dataset == "train":
                train = pd.merge(train, tempTable, how='left', on=feat)
            # elif dataset == "eval":
            #     eval = pd.merge(eval, tempTable, how='left', on=feat)
            else:
                test = pd.merge(test, tempTable, how='left', on=feat)
        if feat in ["sales_depot_id", "sales_channel_id", "route_id", "client_id", "product_id"]:
            pass
        else:
            del train[feat]
            # del eval[feat]
            del test[feat]

    # drop if feature has small overall gain for small version
    listo = ["median_product_client_short_name", "median_product_client", "mean_product_client_route",
             "sales_unit_this_week_median_product_client_agencia_route",

        "returns_unit_next_week_mean_product_client_agencia_route","sales_channel_id",
             "returns_next_week_sum_product_client_agencia_route","returns_next_week_mean_product_client_agencia_route",
             "returns_next_week_median_product_client_agencia_route","returns_unit_next_week_sd_product_client_agencia_route",
             "returns_unit_next_week_median_product_client_agencia_route","returns_next_week_sd_product_client_agencia_route",

        "median_short_name", "returns_unit_next_week_sum_product_client_agencia_route", "median_client_route_short_name",
      "count_short_name_client_agencia", "returns_unit_next_week_median_product_client","sd_product_client_route_short_name",
      "sales_this_week_median_product_client","median_product_id","count_product_client_agencia_short_name",
      "returns_next_week_mean_product_client","sales_this_week_mean_product_route",
      "sd_short_name_route","returns_unit_next_week_median_short_name_client_agencia","sales_this_week_median_product_client_agencia_route",
      "sd_product_route_short_name","sales_this_week_median_short_name_client_agencia",
      "sales_this_week_median_short_name_client","sales_this_week_sd_product_client_agencia_route","count_product_id",
      "returns_unit_next_week_median_short_name_client","mean_client_name3","sales_this_week_sd_product_route",
      "returns_unit_next_week_mean_short_name_client_agencia","sum_product_client_route_short_name","sales_unit_this_week_count_product_client_agencia_route",
      "returns_next_week_mean_short_name_client_agencia","count_product_client_short_name","returns_unit_next_week_sd_product_route",
      "sum_short_name_route","sd_product_route","median_product_client_route_short_name","returns_next_week_sum_product_route",
      "returns_unit_next_week_sum_product_route","sales_this_week_sum_product_route","median_product_client_route",
      "count_short_name_route","sales_unit_this_week_count_product_route","sum_short_name_client_name3",
      "returns_next_week_sd_product_route","sales_this_week_count_product_client_agencia_route",
      "sales_unit_this_week_sum_product_route","sum_product_route_short_name","count_short_name_client_name3",
      "count_product_client_route_short_name","sum_product_client_route","sd_client_name3","returns_unit_next_week_median_product_route",
      "median_client_name3","sales_this_week_count_product_route","count_product_client","count_client_name3",
      "returns_unit_next_week_count_product_client_agencia_route","returns_next_week_median_product_route",
      "returns_next_week_count_product_client_agencia_route","returns_unit_next_week_count_product_route",
      "returns_next_week_count_product_route","count_product_route_short_name","sum_client_name3",
      "count_product_client_route_short_name_agencia",

             "count_product_client_agencia", "sum_brand2", "num_brands", "median_brand2", "beverage",
             "returns_next_week_median_product", "returns_unit_next_week_median_product",

             "count_state", "week", "returns_unit_next_week_mean_product_client",
             "sd_town", "mean_brand", "mean_sales_channel_id", "returns_unit_next_week_sd_product_client",
             "median_route_id",

             "mean_client_name", "mean_route_id", "mean_town", "sd_product_client_agencia", "sum_brand",
             "null_count",
             "sum_product_route", "returns_next_week_sd_product_client", "mean_client_name2", "count_town",
             "count_sales_depot_id", "sum_state", "sales_this_week_sd_product_client", "sum_sales_depot_id",
             "sd_product_client", "sd_route_id", "sd_product_client_route", "count_product_route",
             "count_short_name_client_name2", "sum_town", "client_id", "sd_product_client_agencia_route",
             "count_brand",

             "count_product_client_name", "median_client_name", "median_sales_channel_id",
             "sum_short_name_client_name2",
             "median_client_name2", "count_product_client_agencia_route", "median_sales_depot_id", "median_brand",
             "mean_product_client_name", "sd_short_name_client_name2",
             "returns_unit_next_week_mean_short_name_client",
             "count_route_id", "sum_route_id", "median_town", "count_product_client_route", "sd_client_name2",
             "returns_next_week_mean_short_name_client", "median_product_client_name", "sum_sales_channel_id",
             "count_client_name", "count_sales_channel_id", "sum_client_name2", "count_client_name2",
             "sd_client_name",
             "sd_sales_channel_id", "sales_unit_this_week_sd_short_name_client", "median_state",
             "sum_client_name",
             "sum_product_client_name", "sd_product_client_name", "sales_this_week_mean_short_name_client",
             "sales_this_week_sd_short_name_client", "returns_unit_next_week_sd_short_name_client",
             "returns_next_week_sd_short_name_client",
             "-5_week_mean_short_name_client", "-4_week_mean_short_name_client", "-6_week_mean_short_name_client",
             "-3_week_mean_product_client",
             "-6_week_mean_product_client", "-4_week_mean_product_client", "-5_week_mean_product_client"
             ]
    train = train.drop(listo, axis=1)
    test = test.drop(listo, axis=1)
    id = pd.DataFrame(test["id"])
    return train, train_target, test, id