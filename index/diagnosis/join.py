import pandas as pd


if __name__ == "__main__":
    shanghai = pd.read_csv("shanghai_7988.csv")
    shanghai = shanghai.loc[:, "path":].join(shanghai.loc[:, "hypertension":"hyperlipidemia"])

    shaoyifu = pd.read_csv("shaoyifu_990.csv")
    shaoyifu = shaoyifu.loc[:, "path":].join(shaoyifu.loc[:, "抑郁":"发作性睡病"])
    shaoyifu = shaoyifu.rename(columns={"帕金森": "pd"})

    ruijin = pd.read_csv("ruijin_477.csv")
    ruijin = ruijin.loc[:, "path":].join(ruijin.loc[:, "pd":"rbd"])

    guangzhou = pd.read_csv("guangzhou_206.csv")
    guangzhou = guangzhou.loc[:, "path":].join(guangzhou.loc[:, "RBD":"PD"])
    guangzhou = guangzhou.rename(columns={"RBD": "rbd", "PD": "pd"})

    join = pd.concat([shanghai, shaoyifu, ruijin, guangzhou], ignore_index=True)
    is_healthy = (join.loc[:, "hypertension":"hyperlipidemia"] == 0).all(axis=1)

    for disease in join.columns.to_list()[3:]:
        positive = join[disease] == 1
        cur = join[positive | is_healthy].fillna(0)
        cur = cur.loc[:, :"split"].join(cur[disease].rename("label").astype(int))
        cur.to_csv(f"{disease}.csv", index=False)
